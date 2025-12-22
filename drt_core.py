"""
DRT (Distribution of Relaxation Times) Analysis Core Module
===========================================================
Implements Tikhonov regularization-based DRT transformation
from Electrochemical Impedance Spectroscopy (EIS) data.

Theory References:
- Ciucci, F., & Orazov, M. (2014). J. Electrochem. Soc., 160(10), F1422.
- Malifarge, B., et al. (2019). ChemElectroChem, 6(24), 6027–6037.

License: MIT
"""

import numpy as np
from scipy.linalg import solve
from scipy.optimize import fminbound
from scipy.signal import find_peaks
import warnings

warnings.filterwarnings('ignore')


class DRTAnalyzer:
    """Core DRT analyzer using Tikhonov regularization."""
    
    def __init__(self):
        self.frequency = None
        self.Z_real = None
        self.Z_imag = None
        self.omega = None
        self.tau_grid = None
        self.gamma = None
        self.Z_reconst_real = None
        self.Z_reconst_imag = None
        self.residual = None
        self.peaks_info = None
        self.lambda_opt = None
        self.fit_metrics = {}
        
    def load_data(self, frequency, Z_real, Z_imag):
        """Load and preprocess EIS data."""
        frequency = np.asarray(frequency, dtype=float)
        Z_real = np.asarray(Z_real, dtype=float)
        Z_imag = np.asarray(Z_imag, dtype=float)
        
        sort_idx = np.argsort(frequency)
        self.frequency = frequency[sort_idx]
        self.Z_real = Z_real[sort_idx]
        self.Z_imag = np.abs(Z_imag[sort_idx])
        
        unique_f, unique_idx = np.unique(self.frequency, return_index=True)
        self.frequency = unique_f
        self.Z_real = self.Z_real[unique_idx]
        self.Z_imag = self.Z_imag[unique_idx]
        
        self.omega = 2 * np.pi * self.frequency
        self.fit_metrics['n_points'] = len(self.frequency)
        
    def _setup_tau_grid(self, n_tau=100):
        """Setup logarithmic time constant grid."""
        if self.frequency is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        tau_min = 1 / (2 * np.pi * self.frequency.max())
        tau_max = 1 / (2 * np.pi * self.frequency.min())
        
        extension = 0.5
        tau_min_ext = tau_min / (10 ** extension)
        tau_max_ext = tau_max * (10 ** extension)
        
        self.tau_grid = np.logspace(
            np.log10(tau_min_ext), 
            np.log10(tau_max_ext), 
            n_tau
        )
        
    def _build_kernel_matrix(self):
        """Build kernel matrix K(ω, τ) for DRT inversion."""
        n_freq = len(self.omega)
        n_tau = len(self.tau_grid)
        
        A = np.zeros((2 * n_freq, n_tau))
        
        for j, tau in enumerate(self.tau_grid):
            w_tau = self.omega * tau
            denom = 1.0 + w_tau**2
            
            A[:n_freq, j] = tau / denom
            A[n_freq:, j] = self.omega * tau**2 / denom
        
        return A
    
    def _build_data_vector(self):
        """Build combined data vector [Z_real; Z_imag]."""
        return np.concatenate([self.Z_real, self.Z_imag])
    
    def _regularization_matrix(self, n_tau, order=2):
        """Build Tikhonov regularization matrix."""
        if order == 1:
            L = np.diff(np.eye(n_tau), axis=0)
        elif order == 2:
            I = np.eye(n_tau)
            L = np.diff(I, n=2, axis=0)
        else:
            raise ValueError("Only order 1 or 2 supported")
        return L
    
    def _compute_gcv(self, A, z, L, lambda_val):
        """Compute Generalized Cross-Validation score."""
        try:
            ATA = A.T @ A
            LTL = L.T @ L
            ATz = A.T @ z
            
            M = ATA + lambda_val * LTL
            gamma = solve(M, ATz)
            
            r = z - A @ gamma
            rss = np.sum(r**2)
            
            try:
                M_inv = np.linalg.inv(M)
                trace = np.trace(A @ M_inv @ A.T)
            except:
                trace = A.shape[0]
            
            dof = A.shape[0] - trace
            
            if dof <= 0:
                return np.inf
            
            gcv = rss / (dof ** 2)
            return gcv
            
        except:
            return np.inf
    
    def solve_drt(self, n_tau=100, lambda_val=None, lambda_auto=True, 
                  non_negative=False, verbose=True):
        """Solve DRT using Tikhonov regularization."""
        if self.frequency is None:
            raise ValueError("Load data first using load_data()")
        
        self._setup_tau_grid(n_tau)
        if verbose:
            print(f"✓ Tau grid setup: {n_tau} points from {self.tau_grid[0]:.2e} to {self.tau_grid[-1]:.2e} s")
        
        A = self._build_kernel_matrix()
        z = self._build_data_vector()
        L = self._regularization_matrix(n_tau, order=2)
        
        if lambda_auto and lambda_val is None:
            if verbose:
                print("→ Searching optimal λ via GCV...")
            
            lambda_range = np.logspace(-8, 2, 50)
            gcv_scores = []
            
            for lam in lambda_range:
                gcv = self._compute_gcv(A, z, L, lam)
                gcv_scores.append(gcv)
            
            idx_opt = np.nanargmin(gcv_scores)
            self.lambda_opt = lambda_range[idx_opt]
            
            if verbose:
                print(f"✓ Optimal λ = {self.lambda_opt:.2e} (GCV = {gcv_scores[idx_opt]:.2e})")
        else:
            self.lambda_opt = lambda_val if lambda_val is not None else 1e-4
            if verbose:
                print(f"✓ Using λ = {self.lambda_opt:.2e}")
        
        ATA = A.T @ A
        LTL = L.T @ L
        ATz = A.T @ z
        
        M = ATA + self.lambda_opt * LTL
        
        try:
            gamma_raw = solve(M, ATz)
            
            if non_negative:
                gamma = np.maximum(gamma_raw, 0)
            else:
                gamma = gamma_raw
            
            self.gamma = gamma
            
            if verbose:
                print(f"✓ DRT solved, γ range: [{gamma.min():.2e}, {gamma.max():.2e}] Ω")
            
        except Exception as e:
            print(f"✗ Solve failed: {e}")
            return False
        
        self._reconstruct_impedance(A)
        self._calculate_residuals(z)
        self._find_peaks()
        
        return True
    
    def _reconstruct_impedance(self, A):
        """Reconstruct impedance from gamma and kernel matrix."""
        n_freq = len(self.frequency)
        z_reconst = A @ self.gamma
        
        self.Z_reconst_real = z_reconst[:n_freq]
        self.Z_reconst_imag = z_reconst[n_freq:]
    
    def _calculate_residuals(self, z):
        """Calculate fit quality metrics."""
        z_reconst = np.concatenate([self.Z_reconst_real, self.Z_reconst_imag])
        residual = z - z_reconst
        
        mse = np.mean(residual**2)
        rmse = np.sqrt(mse)
        max_err = np.max(np.abs(residual))
        
        z_norm = np.linalg.norm(z)
        relative_error = rmse / z_norm if z_norm > 0 else np.inf
        
        self.residual = {
            'abs': residual,
            'mse': mse,
            'rmse': rmse,
            'max_error': max_err,
            'relative_error': relative_error
        }
        
        self.fit_metrics['rmse'] = rmse
        self.fit_metrics['relative_error'] = relative_error
    
    def _find_peaks(self, height_threshold=0.05):
        """Detect peaks in DRT distribution."""
        if self.gamma is None:
            return
        
        max_height = np.max(self.gamma)
        min_height = height_threshold * max_height
        
        peaks, properties = find_peaks(
            self.gamma, 
            height=min_height,
            distance=max(1, len(self.gamma) // 20)
        )
        
        self.peaks_info = []
        
        for peak_idx in peaks:
            tau_peak = self.tau_grid[peak_idx]
            gamma_peak = self.gamma[peak_idx]
            
            half_height = gamma_peak / 2
            fwhm_left = tau_peak
            fwhm_right = tau_peak
            
            for i in range(peak_idx - 1, -1, -1):
                if self.gamma[i] < half_height:
                    fwhm_left = self.tau_grid[i]
                    break
            
            for i in range(peak_idx + 1, len(self.gamma)):
                if self.gamma[i] < half_height:
                    fwhm_right = self.tau_grid[i]
                    break
            
            dlog_tau = np.log(self.tau_grid[1] / self.tau_grid[0])
            resistance_contrib = np.sum(self.gamma[max(0, peak_idx-5):min(len(self.gamma), peak_idx+6)]) * dlog_tau
            
            self.peaks_info.append({
                'tau': tau_peak,
                'gamma': gamma_peak,
                'resistance': resistance_contrib,
                'index': peak_idx,
                'fwhm_range': (fwhm_left, fwhm_right)
            })
    
    def get_summary(self):
        """Get summary statistics of DRT analysis."""
        if self.gamma is None:
            return None
        
        summary = {
            'n_peaks': len(self.peaks_info) if self.peaks_info else 0,
            'gamma_max': np.max(self.gamma),
            'gamma_min': np.min(self.gamma),
            'total_resistance': np.sum(self.gamma) * np.log(self.tau_grid[1] / self.tau_grid[0]),
            'lambda_opt': self.lambda_opt,
            'fit_metrics': self.fit_metrics,
            'residual_stats': {
                'rmse': self.residual['rmse'],
                'relative_error': self.residual['relative_error'],
                'max_error': self.residual['max_error']
            } if self.residual else None
        }
        
        return summary


def generate_test_eis(circuit_type='RC', frequency=None, seed=42):
    """Generate synthetic EIS data for testing."""
    np.random.seed(seed)
    
    if frequency is None:
        frequency = np.logspace(2, 6, 50)
    
    omega = 2 * np.pi * frequency
    
    if circuit_type == 'RC':
        R0, R1, C1 = 10, 100, 1e-6
        Z_real = R0 + R1 / (1 + (omega * R1 * C1)**2)
        Z_imag = (omega * R1**2 * C1) / (1 + (omega * R1 * C1)**2)
        
    elif circuit_type == 'RC-RC':
        R0, R1, C1, R2, C2 = 10, 50, 1e-6, 50, 1e-5
        Z1_real = R1 / (1 + (omega * R1 * C1)**2)
        Z1_imag = (omega * R1**2 * C1) / (1 + (omega * R1 * C1)**2)
        Z2_real = R2 / (1 + (omega * R2 * C2)**2)
        Z2_imag = (omega * R2**2 * C2) / (1 + (omega * R2 * C2)**2)
        Z_real = R0 + Z1_real + Z2_real
        Z_imag = Z1_imag + Z2_imag
        
    else:  # Randles
        R0, Rct, C_dl = 5, 100, 1e-5
        sigma_w = 50
        Z_w = sigma_w / np.sqrt(omega + 1e-10) * (1 - 1j)
        Z_real = R0 + Rct / (1 + (omega * Rct * C_dl)**2) + Z_w.real
        Z_imag = (omega * Rct**2 * C_dl) / (1 + (omega * Rct * C_dl)**2) + Z_w.imag
    
    noise_level = 0.01
    Z_real += np.random.normal(0, noise_level * np.mean(Z_real), len(Z_real))
    Z_imag += np.random.normal(0, noise_level * np.mean(Z_imag), len(Z_imag))
    
    return frequency, Z_real, np.abs(Z_imag)
