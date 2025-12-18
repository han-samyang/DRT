# EIS to DRT 사용 가이드 (Usage Guide)

## 목차 (Table of Contents)

1. [개요 (Overview)](#개요-overview)
2. [설치 방법 (Installation)](#설치-방법-installation)
3. [기본 사용법 (Basic Usage)](#기본-사용법-basic-usage)
4. [고급 사용법 (Advanced Usage)](#고급-사용법-advanced-usage)
5. [데이터 형식 (Data Format)](#데이터-형식-data-format)
6. [파라미터 튜닝 (Parameter Tuning)](#파라미터-튜닝-parameter-tuning)
7. [결과 해석 (Interpreting Results)](#결과-해석-interpreting-results)
8. [문제 해결 (Troubleshooting)](#문제-해결-troubleshooting)

## 개요 (Overview)

이 도구는 전기화학 임피던스 분광법(EIS) 데이터를 이완시간 분포(DRT)로 변환합니다. DRT는 배터리와 연료전지의 내부 프로세스를 분석하는 강력한 방법입니다.

This tool converts Electrochemical Impedance Spectroscopy (EIS) data to Distribution of Relaxation Times (DRT). DRT is a powerful method for analyzing internal processes in batteries and fuel cells.

## 설치 방법 (Installation)

### 1. 의존성 설치 (Install Dependencies)

```bash
pip install -r requirements.txt
```

### 2. 테스트 (Test Installation)

```bash
python eis_to_drt.py
```

성공하면 `eis_drt_summary.png` 파일이 생성됩니다.

If successful, an `eis_drt_summary.png` file will be generated.

## 기본 사용법 (Basic Usage)

### 1. 합성 데이터로 시작하기 (Start with Synthetic Data)

```python
from eis_to_drt import EIStoDRT, generate_synthetic_eis_data

# 데이터 생성 (Generate data)
frequencies, impedances = generate_synthetic_eis_data()

# 변환기 생성 (Create converter)
converter = EIStoDRT(frequencies, impedances)

# DRT 계산 (Compute DRT)
converter.estimate_R_inf()
tau, gamma = converter.compute_drt(lambda_reg=1e-3)

# 시각화 (Visualize)
converter.plot_summary('results.png')
```

### 2. 실제 데이터 사용 (Use Real Data)

```python
import numpy as np
from eis_to_drt import EIStoDRT

# CSV 파일 로드 (Load CSV file)
# 형식: frequency, Z_real, Z_imag
data = np.loadtxt('my_data.csv', delimiter=',', skiprows=1)
frequencies = data[:, 0]
impedances = data[:, 1] + 1j * data[:, 2]

# 변환 실행 (Run conversion)
converter = EIStoDRT(frequencies, impedances)
converter.estimate_R_inf()

# 최적 λ 찾기 (Find optimal λ)
lambda_opt = converter.find_optimal_lambda()
tau, gamma = converter.compute_drt(lambda_reg=lambda_opt)

# 결과 저장 (Save results)
converter.plot_summary('my_results.png')
```

## 고급 사용법 (Advanced Usage)

### 1. DRT 해상도 조정 (Adjust DRT Resolution)

```python
# 더 높은 해상도 (Higher resolution)
converter = EIStoDRT(frequencies, impedances, n_tau=200)

# 더 낮은 해상도 (Lower resolution, faster)
converter = EIStoDRT(frequencies, impedances, n_tau=50)
```

### 2. 정규화 파라미터 수동 선택 (Manual Regularization Parameter)

```python
# 다양한 λ 값 시도 (Try different λ values)
lambdas = [1e-5, 1e-4, 1e-3, 1e-2]

for lam in lambdas:
    tau, gamma = converter.compute_drt(lambda_reg=lam)
    # 결과 분석 (Analyze results)
```

### 3. 2차 미분 정규화 사용 (Use Second-Order Regularization)

```python
# 더 부드러운 DRT (Smoother DRT)
tau, gamma = converter.compute_drt(lambda_reg=1e-3, derivative_order=2)
```

### 4. 최적화 방법 변경 (Change Optimization Method)

```python
# L-curve 방법 (L-curve method)
lambda_opt = converter.find_optimal_lambda(method='L-curve', n_samples=30)

# GCV 방법 (GCV method)
lambda_opt = converter.find_optimal_lambda(method='gcv')
```

## 데이터 형식 (Data Format)

### CSV 파일 형식 (CSV File Format)

```csv
Frequency(Hz),Z_real(Ohm),Z_imag(Ohm)
0.01,75.2,-12.3
0.1,68.5,-25.6
1.0,55.3,-35.2
...
```

### Python 배열 (Python Arrays)

```python
frequencies = np.array([0.01, 0.1, 1.0, 10.0, ...])  # Hz
Z_real = np.array([75.2, 68.5, 55.3, ...])           # Ohm
Z_imag = np.array([-12.3, -25.6, -35.2, ...])        # Ohm
impedances = Z_real + 1j * Z_imag
```

### 데이터 요구사항 (Data Requirements)

- **주파수 범위 (Frequency range)**: 최소 2-3 decade (예: 0.01 Hz ~ 100 kHz)
- **데이터 포인트 (Data points)**: 최소 20개 권장 (Minimum 20 recommended)
- **품질 (Quality)**: 신호 대 잡음비 > 20 dB 권장

## 파라미터 튜닝 (Parameter Tuning)

### 정규화 파라미터 (Regularization Parameter) λ

| λ 값 | 효과 (Effect) | 사용 시기 (When to Use) |
|------|---------------|------------------------|
| 1e-6 ~ 1e-5 | 많은 피크, 노이즈 민감 (Many peaks, noise-sensitive) | 깨끗한 데이터 (Clean data) |
| 1e-4 ~ 1e-3 | 균형잡힌 결과 (Balanced) | 대부분의 경우 (Most cases) |
| 1e-2 ~ 1e-1 | 부드러운 분포 (Smooth) | 노이즈가 많은 데이터 (Noisy data) |

### DRT 해상도 (DRT Resolution) n_tau

- **50-100**: 빠른 계산, 낮은 해상도 (Fast, low resolution)
- **100-200**: 균형 (권장) (Balanced, recommended)
- **200-500**: 높은 해상도, 느린 계산 (High resolution, slow)

## 결과 해석 (Interpreting Results)

### Nyquist 플롯 (Nyquist Plot)

- **반원 (Semicircles)**: 각 RC 요소 (Each RC element)
- **경사 (Slopes)**: 확산 과정 (Diffusion processes)
- **고주파 점 (High-frequency point)**: R∞ (전해질 저항)

### DRT 플롯 (DRT Plot)

- **피크 위치 (Peak position)**: 이완 시간 τ (Relaxation time)
- **피크 높이 (Peak height)**: 저항 기여도 (Resistance contribution)
- **피크 폭 (Peak width)**: 프로세스 분포 (Process distribution)

### 일반적인 프로세스 (Common Processes)

| τ 범위 | 프로세스 (Process) | 배터리 응용 (Battery Application) |
|--------|-------------------|----------------------------------|
| < 1 ms | 전해질 이온 전도 (Electrolyte ion conduction) | SEI 층 (SEI layer) |
| 1-100 ms | 전하 전달 (Charge transfer) | 전극/전해질 계면 (Electrode/electrolyte interface) |
| 0.1-10 s | 고체 확산 (Solid diffusion) | 리튬 확산 (Li diffusion in active material) |
| > 10 s | 느린 프로세스 (Slow processes) | 농도 분극 (Concentration polarization) |

## 문제 해결 (Troubleshooting)

### 문제 1: DRT에 피크가 보이지 않음

**원인 (Cause)**:
- λ가 너무 큼 (λ too large)
- 데이터 품질 문제 (Data quality issue)

**해결 (Solution)**:
```python
# λ 줄이기 (Reduce λ)
tau, gamma = converter.compute_drt(lambda_reg=1e-5)

# 데이터 확인 (Check data)
converter.plot_eis()  # Nyquist 플롯 확인
```

### 문제 2: DRT가 너무 노이즈가 많음

**원인 (Cause)**:
- λ가 너무 작음 (λ too small)
- 원시 데이터에 노이즈 많음 (Noisy raw data)

**해결 (Solution)**:
```python
# λ 늘리기 (Increase λ)
tau, gamma = converter.compute_drt(lambda_reg=1e-2)

# 또는 자동 선택 (Or auto-select)
lambda_opt = converter.find_optimal_lambda()
```

### 문제 3: 음수 DRT 값

**원인 (Cause)**:
- 데이터가 Kramers-Kronig 관계 위반 (Data violates Kramers-Kronig relations)
- 비선형 시스템 (Non-linear system)

**해결 (Solution)**:
- 측정 조건 확인 (Check measurement conditions)
- 더 작은 진폭으로 재측정 (Remeasure with smaller amplitude)
- DRT는 자동으로 음수 값을 0으로 설정 (DRT automatically clips negative values to 0)

### 문제 4: 메모리 오류

**원인 (Cause)**:
- n_tau가 너무 큼 (n_tau too large)

**해결 (Solution)**:
```python
# n_tau 줄이기 (Reduce n_tau)
converter = EIStoDRT(frequencies, impedances, n_tau=100)
```

## 예제 (Examples)

### 배터리 진단 (Battery Diagnostics)

```python
# 신선한 배터리 vs 노화된 배터리 비교
fresh_converter = EIStoDRT(fresh_freq, fresh_Z)
aged_converter = EIStoDRT(aged_freq, aged_Z)

# DRT 비교
fig, ax = plt.subplots()
fresh_converter.plot_drt(ax)
aged_converter.plot_drt(ax)
ax.legend(['Fresh', 'Aged'])
```

### 온도 의존성 연구 (Temperature Dependence Study)

```python
temperatures = [0, 25, 45]
colors = ['blue', 'green', 'red']

fig, ax = plt.subplots()
for temp, color in zip(temperatures, colors):
    data = load_eis_data(f'data_{temp}C.csv')
    converter = EIStoDRT(data['freq'], data['Z'])
    converter.estimate_R_inf()
    tau, gamma = converter.compute_drt()
    ax.plot(tau, gamma, color=color, label=f'{temp}°C')
    
ax.set_xscale('log')
ax.legend()
ax.set_xlabel('τ (s)')
ax.set_ylabel('γ(τ) (Ω)')
```

## 추가 자료 (Additional Resources)

- **논문 (Papers)**: README.md의 참고문헌 참조
- **도구 (Tools)**: pyDRTtools, DRTtools
- **커뮤니티 (Community)**: GitHub Issues

## 지원 (Support)

질문이나 문제가 있으면 GitHub 이슈를 생성해주세요.

For questions or issues, please create a GitHub issue.
