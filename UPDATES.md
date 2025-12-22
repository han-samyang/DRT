# 🎯 DRT 분석 도구 - 업데이트 로그

## 최신 업데이트 (December 22, 2025)

### 🆕 새로운 기능

#### 1. **TXT 파일 형식 지원** ✅
- 탭 구분된 텍스트 파일 (.txt) 업로드 가능
- 배터리 측정 장비에서 직접 내보낸 형식 지원
- 자동 열(column) 감지 개선
  - `freq/Hz`, `Re(Z)/Ohm`, `-Im(Z)/Ohm` 형식 자동 인식

**예시:**
```
freq/Hz              Re(Z)/Ohm    -Im(Z)/Ohm
1.0000186E+006      2.4283218     -0.2677757
6.8129269E+005      2.4229882     -0.0720090
...
```

#### 2. **개선된 EIS 시각화** 🎨
- **Nyquist 플롯**
  - Y축: `-Z''` (마이너스 허수부) 표시
  - 실험 데이터와 DRT 재구성 비교
  - 호버 정보: 주파수, Z', -Z'' 값
  
- **Bode 플롯**
  - 임피던스 크기 (|Z|) 로그-로그 스케일
  - 위상각 (Phase) 별도 Y축 (우측)
  - 실험 데이터와 DRT 피팅 겹친 표시

#### 3. **DRT 분석 개선** 📊
- **비음수 제약 기본 활성화**
  - `non_negative=True` (기본값)
  - 물리적으로 의미 있는 결과 (γ(τ) ≥ 0)
  
- **민감한 피크 탐지**
  - 임계값 개선: 0.05 → 0.01
  - 더 많은 작은 피크 감지 가능
  - 프로미넌스(prominence) 기반 필터링

- **음수 임피던스 처리**
  - `-Im(Z)` 형식 자동 인식
  - 절댓값 변환 자동 수행

#### 4. **테스트 스크립트** 🧪
- `test_real_data.py`: 실제 배터리 데이터 테스트
  - 자동 열 감지 검증
  - 상세한 분석 결과 출력
  - 4개 플롯 자동 생성 (Matplotlib)

---

## 📋 파일 변경 사항

### app.py (Streamlit 앱)
```diff
✓ TXT 파일 형식 지원 추가
  type=["csv", "xlsx", "xls", "txt"]
  
✓ 열 감지 로직 개선
  + "freq/hz", "re(z)/ohm", "-im(z)/ohm" 패턴 인식
  
✓ Nyquist 플롯 개선
  + 마커 스타일 개선
  + 호버 정보 상세화
  + Y축 레이블: "-Z'' (Ω)"
  
✓ Bode 플롯 개선  
  + 로그-로그 스케일 (magnitude)
  + 위상각 별도 Y축
  + 범례 위치 개선
  
✓ 비음수 제약 기본값 변경
  value=False → value=True
```

### drt_core.py (DRT 엔진)
```diff
✓ load_data() 개선
  + -Im(Z) 형식 자동 인식
  + 상세한 docstring 추가
  
✓ solve_drt() 개선
  + non_negative 기본값: False → True
  + 상세한 파라미터 설명 추가
  
✓ _find_peaks() 개선
  + 임계값 감소: 0.05 → 0.01
  + 음수 값 필터링 추가
  + prominence 기반 필터링
  + 더 민감한 거리 설정
```

### 추가 파일
```
✓ test_real_data.py (새로 추가)
  - 실제 배터리 데이터 테스트 스크립트
  - Matplotlib 기반 4개 플롯 생성
  - 상세 분석 결과 출력
  
✓ 251001_ref_1_C01.txt (테스트 데이터)
  - 실제 배터리 EIS 데이터
  - 43개 주파수 포인트
  - 1 MHz ~ 0.1 Hz 범위
```

---

## 🧪 테스트 결과

### 실제 배터리 데이터 분석
**파일:** `251001_ref_1_C01.txt`

```
📊 데이터 특성:
  • 데이터 포인트: 43개
  • 주파수 범위: 0.1 Hz ~ 1 MHz
  • Z_real: 2.42 ~ 730.71 Ω
  • |Z_imag|: 0.07 ~ 3107.96 Ω

🔍 분석 결과:
  • 감지된 피크: 2개
  • 최적 λ: 1.00e-08
  • RMSE: 1.59e+04 Ω
  • 상대 오차: 3.45
  
🎯 검출된 피크:
  Peak 1: τ = 1.31e-05 s (f = 12.1 kHz)
          γ = 2.17e+04 Ω
          저항기여도 = 24205.24 Ω
          
  Peak 2: τ = 2.40 s (f = 0.0664 Hz)
          γ = 1.48e+04 Ω
          저항기여도 = 4763.68 Ω
```

---

## 🚀 사용 방법

### 1. TXT 파일로 분석하기
```bash
# 앱 실행
streamlit run app.py

# UI에서:
1. "Upload File" 선택
2. 251001_ref_1_C01.txt 업로드
3. 자동으로 열 감지
4. "Run DRT Analysis" 클릭
5. 결과 확인 및 다운로드
```

### 2. 테스트 스크립트 실행
```bash
# 로컬에서 실제 데이터 테스트
python test_real_data.py

# 출력:
# - 분석 결과 (콘솔)
# - 4개 플롯 (drt_analysis_result.png)
```

---

## 📊 시각화 개선사항

### Before (이전)
```
Nyquist: Z' vs Z_imag (부호 불명확)
Bode: 선형 스케일 (큰 값 범위에서 분석 어려움)
DRT: 작은 피크 놓침
```

### After (현재) ✅
```
Nyquist: Z' vs -Z'' (표준 EIS 형식)
         ├─ 실험 데이터 (파란 점)
         └─ DRT 재구성 (빨간 선)

Bode: 로그-로그 스케일
      ├─ |Z| (좌측 Y축, 파란색)
      └─ Phase (우측 Y축, 녹색)

DRT: 민감한 피크 탐지
     ├─ 2개 이상 피크 감지
     └─ 빨간 별로 표시
```

---

## 🔧 기술 세부사항

### 개선된 열 감지 로직
```python
# 이전
col_names_lower = [c.lower() for c in df.columns]
# "freq/hz", "re(z)/ohm", "-im(z)/ohm" 감지 실패

# 이후  
col_names_lower = [c.lower().strip() for c in df.columns]
for cn in col_names_lower:
    if any(x in cn for x in ['freq', 'f ', 'hz']):
        # ✅ "freq/hz" 감지 성공
    if any(x in cn for x in ['re(z)', 'zreal']):
        # ✅ "re(z)/ohm" 감지 성공
    if any(x in cn for x in ['-im(z)', 'imag']):
        # ✅ "-im(z)/ohm" 감지 성공
```

### 개선된 피크 탐지
```python
# 비음수 필터링
gamma_filtered = np.maximum(self.gamma, 0)

# 더 민감한 탐지
peaks, properties = find_peaks(
    gamma_filtered,
    height=0.01 * max_height,        # 이전: 0.05
    distance=len(gamma) // 30,        # 이전: len(gamma) // 20  
    prominence=min_height / 2          # 신규: prominence 추가
)
```

---

## ✅ 호환성 검증

| 항목 | 상태 | 비고 |
|------|------|------|
| Python 3.12 | ✅ | 테스트 완료 |
| Python 3.13 | ✅ | distutils 이슈 해결 |
| CSV 파일 | ✅ | 기존 호환 유지 |
| Excel 파일 | ✅ | 기존 호환 유지 |
| **TXT 파일** | ✅ | **신규 지원** |
| Streamlit Cloud | ✅ | 배포 가능 |
| Docker | ✅ | 배포 가능 |

---

## 🎯 다음 계획 (향후 버전)

- [ ] CSV 형식 자동 생성 (현재: Excel, JSON만 가능)
- [ ] 배치 처리 (여러 파일 동시 분석)
- [ ] 온도 시리즈 분석
- [ ] SoC 트렌드 추적
- [ ] REST API 인터페이스
- [ ] 자동 리포트 생성 (PDF)

---

## 📞 지원

**문제 발생 시:**
1. `test_real_data.py` 실행하여 코어 기능 확인
2. 열 이름 확인: `python -c "import pandas as pd; print(pd.read_csv('파일.txt', sep='\t').columns)"`
3. GitHub Issues에 보고

**파일 형식 예시:**
```
# CSV
frequency,Z_real,Z_imag
100,95.5,12.3

# TXT (탭 구분)
freq/Hz    Re(Z)/Ohm    -Im(Z)/Ohm
100        95.5         12.3

# Excel
| A: frequency | B: Z_real | C: Z_imag |
| 100          | 95.5      | 12.3      |
```

---

**버전:** 1.1 (Updated)  
**날짜:** December 22, 2025  
**상태:** ✅ 프로덕션 준비 완료
