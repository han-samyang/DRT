# DRT Analysis Tool 📊

Distribution of Relaxation Times (DRT) 분석을 위한 **웹 기반 도구**입니다.
EIS(Electrochemical Impedance Spectroscopy) 데이터를 업로드하면 자동으로 DRT 분석을 수행합니다.

**[Ciucci's Lab](https://github.com/ciuccislab/pyDRTtools)의 pyDRTtools 방법론을 기반으로 개발되었습니다.**

---

## ✨ 주요 기능

✅ **EIS 파일 업로드** - CSV 또는 Excel 형식 지원  
✅ **자동 DRT 변환** - Tikhonov 규제화 기반 계산  
✅ **5개 탭 분석**  
- 📊 Nyquist Plot
- 🌊 Bode Plot (Magnitude + Phase)
- 📈 DRT 분포
- ✅ 재구성 검증
- 📋 피크 테이블

✅ **인터랙티브 플롯** - Plotly 기반 (확대, 축소, 저장 가능)  
✅ **결과 다운로드** - Excel 형식으로 저장  
✅ **합성 데이터 테스트** - 미리 준비된 예제로 즉시 테스트  

---

## 🚀 빠른 시작 (5분)

### 1️⃣ 설치

```bash
# 저장소 클론
git clone https://github.com/YOUR_USERNAME/DRT-Tool.git
cd DRT-Tool

# 패키지 설치
pip install -r requirements.txt
```

### 2️⃣ 실행

```bash
streamlit run app.py
```

### 3️⃣ 브라우저에서 접속

```
http://localhost:8501
```

---

## 📖 사용 방법

### 웹 인터페이스 (권장)

```
1. "📁 파일 업로드" 클릭
   → CSV 또는 Excel 파일 선택

2. 컬럼 자동 인식
   → 주파수(Hz), Z'(Ω), Z''(Ω)

3. "DRT 파라미터" 설정 (선택)
   → τ 그리드 포인트: 150 (기본)
   → 규제화 λ: 10^-3 (기본)
   → 규제화 방법: Ridge (기본)

4. "🚀 DRT 분석 시작" 클릭

5. 결과 확인
   ├─ Nyquist/Bode/DRT 플롯
   ├─ 피크 테이블
   └─ 재구성 검증

6. "📊 Excel 다운로드"로 결과 저장
```

### Python 스크립트로 직접 사용

```python
from drt_core import DRTCalculator, create_synthetic_eis

# 합성 데이터 생성 (테스트)
data = create_synthetic_eis({'R0': 10, 'R': [100], 'C': [1e-6]})

# DRT 계산
calc = DRTCalculator(data['freq'], data['z_real'], data['z_imag'])
result = calc.compute(n_tau=150, lambda_param=1e-3, method='ridge')

# 결과 확인
print(f"τ_peak: {result['stats']['tau_at_max']:.2e} s")
print(f"γ_max: {result['stats']['gamma_max']:.6f} A/Ω")
print(f"Total R: {result['stats']['total_R']:.2f} Ω")
print(f"RMSE: {result['rmse']:.2e}")
print(f"Relative Error: {result['rel_error']*100:.2f}%")

# 피크 정보
print(result['peaks_df'])
```

---

## 📁 프로젝트 구조

```
DRT-Tool/
├── app.py                           # Streamlit 웹 앱 (메인)
├── drt_core.py                      # DRT 계산 엔진
├── requirements.txt                 # 필요 패키지
├── README.md                        # 이 파일
├── LICENSE                          # MIT 라이센스
│
└── docs/                            # 문서
    ├── QUICK_START_ONE_PAGE.md      # 5분 시작 가이드
    ├── AUTO_LINKING_EXPLANATION.md  # 코드 연동 원리
    ├── CODE_STRUCTURE_EXPLAINED.md   # 코드 구조 설명
    ├── FINAL_STRATEGY_SUMMARY.md    # 전체 전략
    ├── ciuccislab_utilization_strategy.md
    └── pyDRTtools_analysis_guide.md
```

---

## 🔧 파라미터 설명

### τ (시간상수) 그리드 포인트
- **범위**: 50~300
- **기본값**: 150
- **의미**: DRT 해상도. 클수록 정교하지만 느림

### 규제화 강도 (λ = 10^x)
- **범위**: 10^-6 ~ 10^0
- **기본값**: 10^-3
- **팁**:
  - 작음 (10^-4): 데이터 적합성 높음, 노이즈 민감
  - 중간 (10^-3): 표준 설정 ⭐
  - 큼 (10^-2): 평탄함, 정보 손실

### 규제화 방법
- **Ridge (L2)**: 표준, 음수 허용 ⭐
- **Ridge + NNLS**: 음수 제약 (물리적)
- **LASSO**: 희소성 강조
- **NNLS**: 순수 음수 없는 최소제곱

---

## 📊 출력 해석

### Nyquist Plot
- 원래 EIS 데이터의 복소 임피던스 표시
- 반원: 한 개의 RC 쌍 또는 프로세스

### Bode Plot
- 주파수에 따른 |Z| (크기)와 Phase 변화
- 로그-로그 스케일

### DRT (Distribution of Relaxation Times)
- γ(τ) 분포: 시간상수별 저항 기여도
- 피크: 특정 프로세스 식별
- Area: 해당 프로세스의 저항값

### 재구성 검증
- 원본 Z''과 재구성 Z''의 비교
- RMSE & Relative Error
- Relative Error < 5%면 양호

### 피크 테이블
- τ_peak: 시간상수 값
- γ_peak: 피크 높이
- ΔR: 저항 기여도
- log₁₀(τ): 로그 시간상수

---

## 🧪 테스트

### 단위 테스트 (drt_core.py)
```bash
python drt_core.py
```

**출력 예시:**
```
Test: Single ZARC (R=100 Ω, C=1 µF)
Results:
  τ_peak = 1.00e-04 s
  γ_max = 0.009999 A/Ω
  Total R ≈ 100.00 Ω
  RMSE = 1.23e-15
  Rel Error = 0.00%

Peaks found: 1
  Peak #  τ_peak (s)  γ_peak (A/Ω)  ΔR (Ω)  log₁₀(τ_peak)
0     1     1.00e-04      0.009999    100.0000      -4.00
```

✅ 이 정도면 설치 성공

---

## 🔗 참고 자료

### 학술 논문 (DRT 방법론)

1. **Wan et al. (2015)** - DRTtools 원조
   - "Influence of the discretization methods..."
   - *Electrochimica Acta*, 184: 483-499
   - DOI: https://doi.org/10.1016/j.electacta.2015.09.097

2. **Liu & Ciucci (2019)** - GP-DRT
   - "Gaussian process distribution of relaxation times..."
   - *Electrochimica Acta*, 135316
   - DOI: https://doi.org/10.1016/j.electacta.2019.135316

3. **Ciucci (2022)** - 최신 종설
   - "Distribution of Relaxation Times Analysis"
   - *Joule*, 6: 1172-1198
   - DOI: https://doi.org/10.1016/j.joule.2022.04.003

### 오픈소스 기반

- **pyDRTtools**: https://github.com/ciuccislab/pyDRTtools
- **Ciucci's Lab**: https://github.com/ciuccislab
- **DRTtools (MATLAB)**: https://github.com/ciuccislab/DRTtools

---

## ❓ FAQ

### Q1: "CSV 파일 포맷이 뭔가요?"

**A:** 다음 컬럼이 필요합니다:
```
frequency (Hz), Z_real (Ohm), Z_imag (Ohm)
0.01,          100,          50
0.1,           100,          40
1,             100,          30
...
```

또는 다른 이름도 자동 인식:
- 주파수: freq, f, frequency_Hz
- Z': Z', Re(Z), Zreal
- Z'': Z'', -Z'', Im(Z), Zimag

### Q2: "두 개 이상의 파일 한 번에 분석하고 싶어요"

**A:** 배치 스크립트 만들기:
```python
# batch_analysis.py
from drt_core import DRTCalculator
import pandas as pd
import glob

for csv_file in glob.glob("*.csv"):
    df = pd.read_csv(csv_file)
    calc = DRTCalculator(df['freq'], df['z_real'], df['z_imag'])
    result = calc.compute()
    print(f"{csv_file}: τ_peak={result['stats']['tau_at_max']:.2e}")
```

실행:
```bash
python batch_analysis.py
```

### Q3: "재구성 오차가 20% 이상인데요?"

**A:** 
1. λ 값 조정 (작은 값으로)
2. n_tau 증가
3. 원본 데이터 확인 (노이즈, 이상치)

### Q4: "다른 사람과 공유하고 싶어요"

**A:** 클라우드 배포:
```bash
# Streamlit Cloud에 배포
# https://streamlit.io/cloud

# 또는 Docker화
docker build -t drt-tool .
docker run -p 8501:8501 drt-tool
```

---

## 🛠️ 개발 환경

### 요구사항
- Python 3.8+
- Streamlit 1.28+
- Plotly 5.17+
- Pandas 2.0+
- NumPy 1.24+
- SciPy 1.11+
- scikit-learn 1.3+

### 설치 확인
```bash
python --version        # 3.8 이상?
pip list | grep streamlit
```

---

## 📝 라이센스

MIT License - 자유롭게 수정 및 배포 가능

```
MIT License © 2024

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files...
```

기반: [pyDRTtools](https://github.com/ciuccislab/pyDRTtools) (MIT License)

---

## 🙏 인용 (학술 사용)

이 도구를 학술 논문에 사용하셨다면:

```bibtex
@software{DRTTool2024,
  title={DRT Analysis Tool: Web-based Distribution of Relaxation Times Analysis},
  author={[Your Name]},
  year={2024},
  url={https://github.com/YOUR_USERNAME/DRT-Tool}
}

% 그리고 원본 논문도 인용:
@article{Wan2015,
  title={Influence of the discretization methods on the distribution 
         of relaxation times deconvolution},
  author={Wan, T. H. and Saccoccio, M. and Chen, C. and Ciucci, F.},
  journal={Electrochimica Acta},
  volume={184},
  pages={483--499},
  year={2015}
}
```

---

## 🐛 버그 리포트 & 기능 제안

### Issue 작성
1. GitHub → Issues → New Issue
2. 제목: "[Bug] 문제" 또는 "[Feature] 기능 제안"
3. 설명: 상세한 재현 방법

### Pull Request
```bash
git fork https://github.com/YOUR_USERNAME/DRT-Tool.git
git checkout -b feature/my-feature
git commit -am "Add my feature"
git push origin feature/my-feature
# GitHub에서 PR 생성
```

---

## 📧 문의

- **Issues**: https://github.com/YOUR_USERNAME/DRT-Tool/issues
- **Email**: your_email@example.com
- **Lab**: [Your Lab Name]

---

## 🔄 버전 히스토리

### v0.1.0 (2024-12-18)
- ✨ 초기 릴리스
- ✅ Tikhonov 규제화 기반 DRT
- ✅ Streamlit 웹 인터페이스
- ✅ Plotly 인터랙티브 플롯
- ✅ Excel 다운로드

### 예정 (v0.2.0)
- 🔄 자동 λ 선택 (GCV/L-curve)
- 🔄 배치 처리 기능
- 🔄 PDF 리포트 생성
- 🔄 고급 규제화 (GP-DRT)

---

## 🌟 Acknowledgments

- **Francesco Ciucci** (HKUST) - DRT 방법론 개발
- **pyDRTtools 개발팀** - 기반 코드
- **Streamlit 팀** - 웹 프레임워크
- **[Contributor Names]** - 피드백 및 개선

---

**Happy DRT Analysis! 🚀**

*마지막 업데이트: 2024-12-18*
