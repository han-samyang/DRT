# EIS to DRT Conversion Tool

전기화학 임피던스 분광법(EIS) 데이터를 이완시간 분포(DRT)로 변환하는 도구입니다.

A Python tool for converting Electrochemical Impedance Spectroscopy (EIS) data to Distribution of Relaxation Times (DRT) using Tikhonov regularization.

## 개요 (Overview)

이 도구는 배터리, 연료전지 및 기타 전기화학 시스템의 임피던스 데이터를 분석하는 데 사용됩니다. EIS 데이터를 DRT로 변환하면 시스템 내부의 여러 이완 과정을 명확하게 식별할 수 있습니다.

This tool is used for analyzing impedance data from batteries, fuel cells, and other electrochemical systems. Converting EIS data to DRT enables clear identification of multiple relaxation processes within the system.

## 주요 기능 (Features)

- ✅ EIS 데이터 로드 및 전처리 (Load and preprocess EIS data)
- ✅ Tikhonov 정규화를 사용한 DRT 계산 (DRT computation using Tikhonov regularization)
- ✅ 최적 정규화 매개변수 자동 선택 (L-curve method)
- ✅ Nyquist 플롯 및 DRT 시각화 (Visualization with Nyquist and DRT plots)
- ✅ 합성 데이터 생성 (테스트용) (Synthetic data generation for testing)

## 설치 (Installation)

### 요구사항 (Requirements)

```bash
pip install -r requirements.txt
```

필요한 패키지:
- numpy >= 1.21.0
- scipy >= 1.7.0
- matplotlib >= 3.4.0
- pandas >= 1.3.0

## 사용법 (Usage)

### 기본 사용 예제 (Basic Example)

```python
from eis_to_drt import EIStoDRT, generate_synthetic_eis_data

# 1. EIS 데이터 생성 또는 로드
frequencies, impedances = generate_synthetic_eis_data()

# 2. 변환기 초기화
converter = EIStoDRT(frequencies, impedances)

# 3. 고주파 저항 추정
converter.estimate_R_inf()

# 4. 최적 정규화 매개변수 찾기
lambda_opt = converter.find_optimal_lambda(method='L-curve')

# 5. DRT 계산
tau, gamma = converter.compute_drt(lambda_reg=lambda_opt)

# 6. 결과 시각화
converter.plot_summary(save_path='eis_drt_results.png')
```

### 명령줄에서 실행 (Command Line)

```bash
python eis_to_drt.py
```

예제 출력을 생성하고 `eis_drt_summary.png` 파일을 저장합니다.

### 실제 데이터 사용 (Using Real Data)

```python
import numpy as np
from eis_to_drt import EIStoDRT

# CSV 파일에서 데이터 로드
# 형식: frequency(Hz), Z_real(Ω), Z_imag(Ω)
data = np.loadtxt('your_eis_data.csv', delimiter=',', skiprows=1)
frequencies = data[:, 0]
impedances = data[:, 1] + 1j * data[:, 2]

# EIS to DRT 변환
converter = EIStoDRT(frequencies, impedances)
converter.estimate_R_inf()
tau, gamma = converter.compute_drt(lambda_reg=1e-3)

# 결과 플롯
converter.plot_summary('results.png')
```

## 이론 배경 (Theory)

### EIS (Electrochemical Impedance Spectroscopy)

전기화학 임피던스 분광법은 다양한 주파수에서 시스템의 임피던스를 측정하는 기술입니다.

EIS measures the impedance of a system across a range of frequencies.

### DRT (Distribution of Relaxation Times)

이완시간 분포는 임피던스 데이터에서 여러 이완 과정을 추출하는 모델-프리 방법입니다.

DRT is a model-free method for extracting multiple relaxation processes from impedance data.

임피던스와 DRT의 관계:

```
Z(ω) = R_∞ + ∫ γ(τ)/(1 + jωτ) dτ
```

여기서:
- `Z(ω)`: 주파수 ω에서의 임피던스 (Impedance at frequency ω)
- `R_∞`: 고주파 저항 (High-frequency resistance)
- `γ(τ)`: 이완시간 τ에서의 DRT (DRT at relaxation time τ)
- `ω`: 각주파수 (Angular frequency)

### Tikhonov 정규화 (Tikhonov Regularization)

역문제를 해결하기 위해 Tikhonov 정규화를 사용합니다:

```
minimize: ||Ax - b||² + λ||Lx||²
```

여기서:
- `A`: 커널 행렬 (Kernel matrix)
- `x`: DRT (γ)
- `b`: 임피던스 데이터 (Impedance data)
- `λ`: 정규화 매개변수 (Regularization parameter)
- `L`: 정규화 행렬 (미분 연산자) (Regularization matrix - derivative operator)

## 응용 분야 (Applications)

- 🔋 리튬이온 배터리 진단 (Lithium-ion battery diagnostics)
- ⚡ 연료전지 성능 분석 (Fuel cell performance analysis)
- 🔬 전기화학 시스템 특성화 (Electrochemical system characterization)
- 📊 배터리 노화 상태 모니터링 (Battery aging monitoring)
- 🏭 품질 관리 및 생산 테스트 (Quality control and production testing)

## 출력 예제 (Example Output)

실행 시 다음과 같은 플롯이 생성됩니다:

1. **Nyquist 플롯**: 복소 임피던스의 실수부와 허수부
2. **DRT 플롯**: 이완시간에 따른 분포

## 알고리즘 파라미터 (Algorithm Parameters)

### 정규화 매개변수 (Regularization Parameter)

- `lambda_reg`: 평활화 정도 조절 (Controls smoothness)
  - 작은 값: 더 많은 피크, 노이즈에 민감 (More peaks, sensitive to noise)
  - 큰 값: 더 평활한 분포 (Smoother distribution)
  - 권장: L-curve 방법으로 자동 선택 (Recommended: auto-selection via L-curve)

### 정규화 방법 (Regularization Method)

- `derivative_order=1`: 1차 미분 (거칠기 페널티) (First derivative - penalizes roughness)
- `derivative_order=2`: 2차 미분 (곡률 페널티) (Second derivative - penalizes curvature)

## 참고문헌 (References)

- Saccoccio, M., Wan, T. H., Chen, C., & Ciucci, F. (2014). Optimal regularization in distribution of relaxation times applied to electrochemical impedance spectroscopy: ridge regression approach. *Electrochimica Acta*, 147, 470-482.
- Ciucci, F., & Chen, C. (2015). Analysis of electrochemical impedance spectroscopy data using the distribution of relaxation times: A Bayesian and hierarchical Bayesian approach. *Electrochimica Acta*, 167, 439-454.
- Tuncer, E., & Macdonald, J. R. (2006). Comparison of methods for estimating continuous distributions of relaxation times. *Journal of Applied Physics*, 99(7), 074106.

## 라이선스 (License)

MIT License

## 기여 (Contributing)

이슈 리포트 및 풀 리퀘스트를 환영합니다!

Issues and pull requests are welcome!

## 연락처 (Contact)

프로젝트 관련 문의사항이 있으시면 이슈를 생성해주세요.

For any questions, please create an issue in the repository.
