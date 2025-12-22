"""
DRT Core Module - 핵심 수학 계산 엔진
Based on pyDRTtools methodology (Ciucci's Lab)

Version: 0.1 (MVP Prototype)
"""

import numpy as np
from scipy.optimize import nnls, minimize
from scipy.signal import find_peaks
from scipy.integrate import trapz
from sklearn.linear_model import Ridge, Lasso
import pandas as pd


class DRTCalculator:
    """DRT 계산 클래스 - Streamlit 통합 용이"""
    
    def __init__(self, freq, z_real, z_imag):
        """
        Parameters
        ----------
        freq : array-like
            주파수 (Hz)
        z_real : array-like
            Z' 실수부 (Ω)
        z_imag : array-like
            Z'' 허수부 (Ω, positive)
        """
        self.freq = np.asarray(freq, dtype=float)
        self.z_real = np.asarray(z_real, dtype=float)
        self.z_imag = np.asarray(z_imag, dtype=float)
        
        # 데이터 정렬 (주파수 내림차순)
        sort_idx = np.argsort(-self.freq)
        self.freq = self.freq[sort_idx]
        self.z_real = self.z_real[sort_idx]
        self.z_imag = self.z_imag[sort_idx]
        
        self.omega = 2 * np.pi * self.freq
        
        # 결과 저장
        self.result = None
    
    def compute(self, n_tau=150, lambda_param=1e-3, reg_order=2, method='ridge'):
        """
        DRT 계산 실행
        
        Parameters
        ----------
        n_tau : int
            τ 그리드 포인트 수
        lambda_param : float
            규제화 강도 λ
        reg_order : int
            규제화 차수 (0=Ridge, 1=1차미분, 2=2차미분)
        method : str
            'ridge', 'ridge_nnls', 'lasso', 'nnls'
        
        Returns
        -------
        dict : 결과 딕셔너리
        """
        
        # 1. τ 그리드 설정
        tau = self._setup_tau_grid(n_tau)
        
        # 2. 커널 행렬 A 구성
        A = self._build_kernel_matrix(tau)
        
        # 3. 규제화 행렬 L
        L = self._compute_L_matrix(tau, order=reg_order)
        
        # 4. γ 계산
        gamma = self._solve_tikhonov(A, self.z_imag, L, lambda_param, method)
        
        # 5. 재구성 & 오차
        z_imag_recon = A @ gamma
        residual = self.z_imag - z_imag_recon
        rmse = np.sqrt(np.mean(residual**2))
        rel_error = rmse / np.mean(np.abs(self.z_imag))
        
        # 6. 통계 & 피크
        stats = self._compute_statistics(tau, gamma)
        peaks_info = self._find_peaks(tau, gamma)
        peaks_df = self._peaks_to_dataframe(peaks_info)
        
        # 결과 저장
        self.result = {
            'tau': tau,
            'gamma': gamma,
            'freq': self.freq,
            'omega': self.omega,
            'z_real': self.z_real,
            'z_imag': self.z_imag,
            'z_imag_recon': z_imag_recon,
            'residual': residual,
            'rmse': rmse,
            'rel_error': rel_error,
            'stats': stats,
            'peaks_info': peaks_info,
            'peaks_df': peaks_df,
            'A_matrix': A,
            'L_matrix': L,
            'lambda_param': lambda_param,
            'reg_order': reg_order,
            'method': method,
            'n_tau': n_tau
        }
        
        return self.result
    
    def _setup_tau_grid(self, n_tau):
        """τ 시간상수 그리드 설정 (log space)"""
        f_min, f_max = np.min(self.freq), np.max(self.freq)
        tau_min = 1 / (2 * np.pi * f_max)
        tau_max = 1 / (2 * np.pi * f_min)
        
        tau = np.logspace(np.log10(tau_min), np.log10(tau_max), n_tau)
        return tau
    
    def _build_kernel_matrix(self, tau):
        """
        커널 행렬 A 구성
        A[i,j] = (ω_i * τ_j) / (1 + (ω_i * τ_j)²)
        """
        n_freq = len(self.omega)
        n_tau = len(tau)
        A = np.zeros((n_freq, n_tau))
        
        for i in range(n_freq):
            w_tau = self.omega[i] * tau
            A[i, :] = w_tau / (1 + w_tau**2)
        
        return A
    
    def _compute_L_matrix(self, tau, order=2):
        """
        규제화 행렬 L 구성
        order=0: Ridge (identity)
        order=1: 1차 평탄도
        order=2: 2차 곡률
        """
        n_tau = len(tau)
        d_tau = np.mean(np.diff(tau))
        
        if order == 0:
            L = np.eye(n_tau)
        elif order == 1:
            # 1차 미분
            L = np.diff(np.eye(n_tau), axis=0) / d_tau
        elif order == 2:
            # 2차 미분 (곡률)
            D1 = np.diff(np.eye(n_tau), axis=0) / d_tau
            L = np.diff(D1, axis=0) / d_tau
        else:
            L = np.eye(n_tau)
        
        return L
    
    def _solve_tikhonov(self, A, b, L, lambda_param, method='ridge'):
        """
        Tikhonov 규제화 해결
        min_γ ||A·γ - b||² + λ·||L·γ||²
        """
        
        if method == 'ridge':
            # Ridge 회귀 (음수 허용)
            model = Ridge(alpha=lambda_param, fit_intercept=False, solver='auto')
            gamma = model.fit(A, b).coef_
        
        elif method == 'ridge_nnls':
            # Ridge + 음수 제약 (2단계)
            model = Ridge(alpha=lambda_param, fit_intercept=False)
            gamma = model.fit(A, b).coef_
            gamma = np.maximum(gamma, 0)  # 음수 → 0
        
        elif method == 'lasso':
            # LASSO (희소성)
            model = Lasso(alpha=lambda_param, fit_intercept=False, max_iter=5000)
            gamma = model.fit(A, b).coef_
            gamma = np.maximum(gamma, 0)
        
        elif method == 'nnls':
            # Non-Negative Least Squares
            gamma, _ = nnls(A, b)
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return gamma
    
    def _compute_statistics(self, tau, gamma):
        """DRT 통계량 계산"""
        idx_max = np.argmax(gamma)
        gamma_max = gamma[idx_max]
        tau_at_max = tau[idx_max]
        
        # ⟨τ⟩ 가중평균
        mean_tau = np.sum(tau * gamma) / np.sum(gamma)
        
        # 전체 저항 (log scale 적분)
        total_R = trapz(gamma, x=np.log10(tau))
        
        # FWHM (반최대폭)
        half_height = gamma_max / 2
        above_half = np.where(gamma >= half_height)[0]
        if len(above_half) > 1:
            fwhm_ratio = tau[above_half[-1]] / tau[above_half[0]]
        else:
            fwhm_ratio = np.nan
        
        return {
            'gamma_max': float(gamma_max),
            'tau_at_max': float(tau_at_max),
            'mean_tau': float(mean_tau),
            'total_R': float(total_R),
            'fwhm_ratio': float(fwhm_ratio)
        }
    
    def _find_peaks(self, tau, gamma, prominence_ratio=0.05):
        """DRT에서 피크 자동 탐지"""
        threshold = prominence_ratio * np.max(gamma)
        peaks, properties = find_peaks(
            gamma,
            height=threshold,
            distance=max(1, len(gamma)//20)
        )
        
        peaks_info = []
        for peak_idx in peaks:
            tau_peak = tau[peak_idx]
            gamma_peak = gamma[peak_idx]
            
            # FWHM 범위에서 면적 계산
            half_height = gamma_peak / 2
            above_half = np.where(gamma >= half_height)[0]
            
            if len(above_half) > 1:
                left_idx, right_idx = above_half[0], above_half[-1]
                area = trapz(gamma[left_idx:right_idx+1], 
                           x=np.log10(tau[left_idx:right_idx+1]))
                tau_left, tau_right = tau[left_idx], tau[right_idx]
            else:
                # FWHM 못 찾으면 ±0.5 decade
                tau_left = tau_peak / np.sqrt(10)
                tau_right = tau_peak * np.sqrt(10)
                idx_range = np.where((tau >= tau_left) & (tau <= tau_right))[0]
                if len(idx_range) > 0:
                    area = trapz(gamma[idx_range], x=np.log10(tau[idx_range]))
                else:
                    area = 0
            
            peaks_info.append({
                'tau_peak': tau_peak,
                'gamma_peak': gamma_peak,
                'area': area,
                'tau_left': tau_left if len(above_half) > 1 else tau_peak/np.sqrt(10),
                'tau_right': tau_right if len(above_half) > 1 else tau_peak*np.sqrt(10)
            })
        
        return peaks_info
    
    def _peaks_to_dataframe(self, peaks_info):
        """피크 정보를 DataFrame으로 변환"""
        if not peaks_info:
            return pd.DataFrame()
        
        data = []
        for i, peak in enumerate(peaks_info):
            data.append({
                'Peak #': i + 1,
                'τ_peak (s)': f"{peak['tau_peak']:.2e}",
                'γ_peak (A/Ω)': f"{peak['gamma_peak']:.6f}",
                'ΔR (Ω)': f"{peak['area']:.4f}",
                'log₁₀(τ_peak)': f"{np.log10(peak['tau_peak']):.2f}"
            })
        
        return pd.DataFrame(data)


# ==================== 편의 함수 ====================

def compute_drt(freq, z_real, z_imag, n_tau=150, lambda_param=1e-3, 
                reg_order=2, method='ridge'):
    """
    간단한 DRT 계산 함수 (클래스 래퍼)
    """
    calc = DRTCalculator(freq, z_real, z_imag)
    return calc.compute(n_tau=n_tau, lambda_param=lambda_param,
                       reg_order=reg_order, method=method)


def create_synthetic_eis(elements, freq=None):
    """
    테스트용 합성 EIS 데이터 생성
    
    Parameters
    ----------
    elements : dict
        {'R0': 10, 'R': [100, 50], 'C': [1e-6, 1e-5]}
        R0: 오믹 저항
        R, C: 직렬 RC 쌍
    freq : array, optional
        주파수 (기본: 0.01 ~ 100kHz)
    
    Returns
    -------
    dict : {'freq', 'z_real', 'z_imag'}
    """
    if freq is None:
        freq = np.logspace(-2, 5, 50)
    
    omega = 2 * np.pi * freq
    z = elements.get('R0', 0) * np.ones_like(omega)
    
    R_list = elements.get('R', [])
    C_list = elements.get('C', [])
    
    for R, C in zip(R_list, C_list):
        z_rc = R / (1 + 1j * omega * R * C)
        z += z_rc
    
    z_real = np.real(z)
    z_imag = -np.imag(z)  # 음수로 저장 (표준)
    
    return {
        'freq': freq,
        'z_real': z_real,
        'z_imag': np.abs(z_imag)
    }


if __name__ == "__main__":
    # 테스트: Single ZARC
    print("=" * 60)
    print("Test: Single ZARC (R=100 Ω, C=1 µF)")
    print("=" * 60)
    
    test_data = create_synthetic_eis({'R0': 10, 'R': [100], 'C': [1e-6]})
    
    result = compute_drt(
        freq=test_data['freq'],
        z_real=test_data['z_real'],
        z_imag=test_data['z_imag'],
        n_tau=100,
        lambda_param=1e-3,
        method='ridge'
    )
    
    print(f"\nResults:")
    print(f"  τ_peak = {result['stats']['tau_at_max']:.2e} s")
    print(f"  γ_max = {result['stats']['gamma_max']:.6f} A/Ω")
    print(f"  Total R ≈ {result['stats']['total_R']:.2f} Ω")
    print(f"  RMSE = {result['rmse']:.2e}")
    print(f"  Rel Error = {result['rel_error']*100:.2f}%")
    print(f"\nPeaks found: {len(result['peaks_info'])}")
    if result['peaks_df'] is not None:
        print(result['peaks_df'].to_string())
