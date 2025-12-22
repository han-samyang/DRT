"""
DRT Analysis Web Application using Streamlit
Based on pyDRTtools methodology (Ciucci's Lab)

Version: 0.1 (MVP)
"""

import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# DRT 핵심 모듈 임포트
from drt_core import DRTCalculator, create_synthetic_eis

# ==================== 페이지 설정 ====================
st.set_page_config(
    page_title="DRT Analysis Tool",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일
st.markdown("""
<style>
    .metric-box {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# ==================== 세션 상태 초기화 ====================
if 'eis_data' not in st.session_state:
    st.session_state.eis_data = None
if 'drt_result' not in st.session_state:
    st.session_state.drt_result = None
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False

# ==================== 사이드바: 입력 설정 ====================
with st.sidebar:
    st.title("📋 DRT 분석 설정")
    
    # 데이터 입력 방식 선택
    input_mode = st.radio(
        "데이터 입력 방식",
        ["📁 파일 업로드", "🧪 합성 데이터 (테스트)"]
    )
    
    # ===== 파일 업로드 모드 =====
    if input_mode == "📁 파일 업로드":
        st.subheader("EIS 파일 업로드")
        
        uploaded_file = st.file_uploader(
            "CSV 또는 Excel 파일 선택",
            type=['csv', 'xlsx', 'xls'],
            help="필수 컬럼: 주파수(Hz), Z'(Ω), Z''(Ω)"
        )
        
        if uploaded_file is not None:
            # 파일 로드
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.success(f"✅ 파일 로드 성공: {len(df)} 포인트")
                
                # 컬럼 매핑
                st.subheader("컬럼 선택")
                
                freq_col = st.selectbox(
                    "주파수 컬럼",
                    df.columns,
                    help="Hz 단위"
                )
                
                zreal_col = st.selectbox(
                    "Z' (실수부) 컬럼",
                    df.columns,
                    help="Ω 단위"
                )
                
                zimag_col = st.selectbox(
                    "Z'' (허수부) 컬럼",
                    df.columns,
                    help="Ω 단위"
                )
                
                # Zimag 부호 처리
                zimag_sign = st.radio(
                    "Z'' 부호 확인",
                    ["-Z'' (표준, 음수로 저장)", "Z'' (양수로 저장)"],
                    help="대부분의 경우 음수로 저장됨"
                )
                
                # 데이터 추출
                freq = df[freq_col].values.astype(float)
                z_real = df[zreal_col].values.astype(float)
                z_imag = df[zimag_col].values.astype(float)
                
                # 부호 처리
                if zimag_sign == "-Z'' (표준, 음수로 저장)":
                    z_imag = np.abs(z_imag)  # 절댓값 취함
                else:
                    z_imag = np.abs(z_imag)
                
                st.session_state.eis_data = {
                    'freq': freq,
                    'z_real': z_real,
                    'z_imag': z_imag
                }
                
                # 데이터 미리보기
                with st.expander("📊 데이터 미리보기"):
                    preview_df = pd.DataFrame({
                        'Frequency (Hz)': freq[:5],
                        "Z' (Ω)": z_real[:5],
                        "Z'' (Ω)": z_imag[:5]
                    })
                    st.dataframe(preview_df)
                    st.caption(f"... 총 {len(freq)} 포인트")
            
            except Exception as e:
                st.error(f"❌ 파일 로드 실패: {e}")
    
    # ===== 합성 데이터 모드 =====
    else:
        st.subheader("합성 EIS 데이터 (테스트)")
        
        test_case = st.selectbox(
            "테스트 케이스",
            [
                "Single ZARC (R=100Ω, C=1µF)",
                "Two ZARC Series (100Ω+50Ω)",
                "Custom"
            ]
        )
        
        if test_case == "Single ZARC (R=100Ω, C=1µF)":
            synthetic = create_synthetic_eis(
                {'R0': 10, 'R': [100], 'C': [1e-6]}
            )
        elif test_case == "Two ZARC Series (100Ω+50Ω)":
            synthetic = create_synthetic_eis(
                {'R0': 10, 'R': [100, 50], 'C': [1e-6, 1e-5]}
            )
        else:
            R0 = st.number_input("R₀ (Ω)", value=10.0)
            R1 = st.number_input("R₁ (Ω)", value=100.0)
            C1 = st.number_input("C₁ (F)", value=1e-6, format="%.2e")
            
            synthetic = create_synthetic_eis(
                {'R0': R0, 'R': [R1], 'C': [C1]}
            )
        
        st.session_state.eis_data = synthetic
        st.success("✅ 합성 데이터 생성 완료")
    
    # ===== DRT 파라미터 설정 =====
    if st.session_state.eis_data is not None:
        st.divider()
        st.subheader("DRT 파라미터")
        
        n_tau = st.slider(
            "τ 그리드 포인트 수",
            min_value=50,
            max_value=300,
            value=150,
            step=10,
            help="시간상수 그리드의 해상도"
        )
        
        # 규제화 강도 (로그 스케일)
        lambda_exp = st.slider(
            "규제화 λ = 10^x",
            min_value=-6,
            max_value=0,
            value=-3,
            step=1,
            help="작을수록: 데이터 적합성 ↑, 노이즈 민감\n"
                 "클수록: 평탄함 ↑, 정보 손실"
        )
        lambda_param = 10 ** lambda_exp
        
        # 규제화 방법
        reg_method = st.radio(
            "규제화 방법",
            [
                ("Ridge (L2) - 표준", "ridge"),
                ("Ridge + 음수제약 (NNLS)", "ridge_nnls"),
                ("LASSO (희소성)", "lasso"),
                ("순수 NNLS", "nnls")
            ],
            format_func=lambda x: x[0],
            help="Ridge: 안정적\n"
                 "NNLS: 음수 제약\n"
                 "LASSO: 희소한 피크"
        )
        reg_method = reg_method[1]  # 투플에서 값 추출
        
        # 규제화 차수
        reg_order = st.selectbox(
            "규제화 차수",
            [(0, "0차 - Ridge (L2)"),
             (1, "1차 - 평탄도"),
             (2, "2차 - 곡률 (표준)")],
            format_func=lambda x: x[1],
            index=2
        )
        reg_order = reg_order[0]
        
        # 분석 실행 버튼
        st.divider()
        if st.button("🚀 DRT 분석 시작", key='run_analysis', use_container_width=True):
            with st.spinner("계산 중... ⏳"):
                try:
                    calculator = DRTCalculator(
                        st.session_state.eis_data['freq'],
                        st.session_state.eis_data['z_real'],
                        st.session_state.eis_data['z_imag']
                    )
                    
                    st.session_state.drt_result = calculator.compute(
                        n_tau=n_tau,
                        lambda_param=lambda_param,
                        reg_order=reg_order,
                        method=reg_method
                    )
                    
                    st.session_state.analysis_done = True
                    st.success("✅ 분석 완료!")
                
                except Exception as e:
                    st.error(f"❌ 분석 실패: {e}")
                    st.session_state.analysis_done = False

# ==================== 메인 영역: 결과 표시 ====================
if st.session_state.drt_result is not None and st.session_state.analysis_done:
    
    result = st.session_state.drt_result
    
    # ===== 상단: 요약 통계 =====
    st.title("⚡ DRT 분석 결과")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "τ_peak (s)",
            f"{result['stats']['tau_at_max']:.2e}"
        )
    
    with col2:
        st.metric(
            "γ_max (A/Ω)",
            f"{result['stats']['gamma_max']:.6f}"
        )
    
    with col3:
        st.metric(
            "Total R (Ω)",
            f"{result['stats']['total_R']:.2f}"
        )
    
    with col4:
        st.metric(
            "Rel. Error (%)",
            f"{result['rel_error']*100:.2f}%"
        )
    
    # ===== 탭 구성 =====
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Nyquist",
        "🌊 Bode",
        "📈 DRT",
        "✅ 재구성",
        "📋 피크"
    ])
    
    # ===== Tab 1: Nyquist Plot =====
    with tab1:
        st.subheader("Nyquist Plot: -Z'' vs Z'")
        
        fig_nyquist = go.Figure()
        
        # 데이터 포인트
        fig_nyquist.add_trace(go.Scatter(
            x=result['z_real'],
            y=result['z_imag'],
            mode='markers',
            name='Measured',
            marker=dict(size=8, color='blue', opacity=0.7),
            hovertemplate='Z\'=%{x:.1f} Ω<br>Z\'\'=%{y:.1f} Ω<extra></extra>'
        ))
        
        # 피팅선 (선택)
        fig_nyquist.add_trace(go.Scatter(
            x=result['z_real'],
            y=result['z_imag'],
            mode='lines',
            name='Trend',
            line=dict(color='blue', width=1, dash='dash'),
            hoverinfo='skip'
        ))
        
        fig_nyquist.update_layout(
            title="Nyquist Plot",
            xaxis_title="Z' (Ω)",
            yaxis_title="-Z'' (Ω)",
            template="plotly_white",
            height=500,
            hovermode='closest',
            showlegend=True
        )
        
        st.plotly_chart(fig_nyquist, use_container_width=True)
    
    # ===== Tab 2: Bode Plot =====
    with tab2:
        st.subheader("Bode Plot")
        
        zmag = np.sqrt(result['z_real']**2 + result['z_imag']**2)
        phase = np.arctan2(-result['z_imag'], result['z_real']) * 180 / np.pi
        
        fig_bode = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Magnitude", "Phase"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Magnitude (log-log)
        fig_bode.add_trace(
            go.Scatter(
                x=result['freq'],
                y=zmag,
                mode='lines+markers',
                name='|Z|',
                line=dict(color='green', width=2),
                marker=dict(size=5),
                hovertemplate='f=%{x:.2e} Hz<br>|Z|=%{y:.1f} Ω<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Phase
        fig_bode.add_trace(
            go.Scatter(
                x=result['freq'],
                y=phase,
                mode='lines+markers',
                name='Phase',
                line=dict(color='red', width=2),
                marker=dict(size=5),
                hovertemplate='f=%{x:.2e} Hz<br>φ=%{y:.1f}°<extra></extra>'
            ),
            row=1, col=2
        )
        
        # 로그 스케일
        fig_bode.update_xaxes(type='log', row=1, col=1)
        fig_bode.update_xaxes(type='log', row=1, col=2)
        fig_bode.update_yaxes(type='log', row=1, col=1)
        
        # 레이아웃
        fig_bode.update_xaxes(title_text="Frequency (Hz)", row=1, col=1)
        fig_bode.update_xaxes(title_text="Frequency (Hz)", row=1, col=2)
        fig_bode.update_yaxes(title_text="|Z| (Ω)", row=1, col=1)
        fig_bode.update_yaxes(title_text="Phase (°)", row=1, col=2)
        
        fig_bode.update_layout(
            template="plotly_white",
            height=500,
            showlegend=True,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_bode, use_container_width=True)
    
    # ===== Tab 3: DRT =====
    with tab3:
        st.subheader("Distribution of Relaxation Times (DRT)")
        
        fig_drt = go.Figure()
        
        fig_drt.add_trace(go.Scatter(
            x=result['tau'],
            y=result['gamma'],
            mode='lines+markers',
            name='γ(τ)',
            line=dict(color='purple', width=2),
            marker=dict(size=4),
            fill='tozeroy',
            fillcolor='rgba(128, 0, 128, 0.2)',
            hovertemplate='τ=%{x:.2e} s<br>γ=%{y:.6f} A/Ω<extra></extra>'
        ))
        
        # 피크 표시
        for i, peak in enumerate(result['peaks_info']):
            fig_drt.add_vline(
                x=peak['tau_peak'],
                line_dash='dash',
                line_color='red',
                annotation_text=f"Peak {i+1}",
                annotation_position="top"
            )
        
        fig_drt.update_layout(
            title="Distribution of Relaxation Times",
            xaxis_title="τ (s)",
            yaxis_title="γ(τ) (A/Ω)",
            xaxis_type='log',
            template="plotly_white",
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_drt, use_container_width=True)
    
    # ===== Tab 4: 재구성 검증 =====
    with tab4:
        st.subheader("원본 vs 재구성 Z'' 비교")
        
        fig_recon = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Z'' 비교", "잔차 (Residual)"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Z'' 비교
        fig_recon.add_trace(
            go.Scatter(
                x=result['freq'],
                y=result['z_imag'],
                mode='markers',
                name='Measured',
                marker=dict(size=6, color='blue'),
                hovertemplate='f=%{x:.2e} Hz<br>Z\'\'=%{y:.1f} Ω<extra></extra>'
            ),
            row=1, col=1
        )
        
        fig_recon.add_trace(
            go.Scatter(
                x=result['freq'],
                y=result['z_imag_recon'],
                mode='lines',
                name='Reconstructed',
                line=dict(color='red', dash='dash', width=2),
                hovertemplate='f=%{x:.2e} Hz<br>Z\'\'=%{y:.1f} Ω<extra></extra>'
            ),
            row=1, col=1
        )
        
        # 잔차
        fig_recon.add_trace(
            go.Scatter(
                x=result['freq'],
                y=result['residual'],
                mode='markers',
                name='Residual',
                marker=dict(size=6, color='green'),
                hovertemplate='f=%{x:.2e} Hz<br>Residual=%{y:.2e}<extra></extra>'
            ),
            row=1, col=2
        )
        
        # 로그 스케일
        fig_recon.update_xaxes(type='log', row=1, col=1)
        fig_recon.update_xaxes(type='log', row=1, col=2)
        
        # 레이아웃
        fig_recon.update_xaxes(title_text="Frequency (Hz)", row=1, col=1)
        fig_recon.update_xaxes(title_text="Frequency (Hz)", row=1, col=2)
        fig_recon.update_yaxes(title_text="Z'' (Ω)", row=1, col=1)
        fig_recon.update_yaxes(title_text="Residual", row=1, col=2)
        
        fig_recon.update_layout(
            template="plotly_white",
            height=500,
            showlegend=True,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_recon, use_container_width=True)
        
        # 오차 통계
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("RMSE", f"{result['rmse']:.2e}")
        with col2:
            st.metric("Rel. Error", f"{result['rel_error']*100:.2f}%")
        with col3:
            st.metric("Mean |Residual|", f"{np.mean(np.abs(result['residual'])):.2e}")
    
    # ===== Tab 5: 피크 테이블 =====
    with tab5:
        st.subheader("탐지된 피크")
        
        if result['peaks_df'] is not None and len(result['peaks_df']) > 0:
            st.dataframe(result['peaks_df'], use_container_width=True)
            
            # 피크별 해석
            with st.expander("📝 피크 해석"):
                for i, peak in enumerate(result['peaks_info']):
                    st.write(f"**Peak {i+1}:**")
                    st.write(f"  - τ = {peak['tau_peak']:.2e} s (log₁₀ = {np.log10(peak['tau_peak']):.2f})")
                    st.write(f"  - γ = {peak['gamma_peak']:.6f} A/Ω")
                    st.write(f"  - ΔR (저항기여) ≈ {peak['area']:.4f} Ω")
                    st.write(f"  - τ 범위: {peak['tau_left']:.2e} ~ {peak['tau_right']:.2e} s")
        else:
            st.info("🔍 탐지된 피크가 없습니다. 규제화 파라미터를 조정해보세요.")
    
    # ===== 다운로드 섹션 =====
    st.divider()
    st.subheader("📥 결과 다운로드")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Excel 다운로드
        output = BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Sheet 1: 요약
            summary_df = pd.DataFrame({
                'Parameter': [
                    'τ_peak (s)',
                    'γ_max (A/Ω)',
                    'Total R (Ω)',
                    'RMSE',
                    'Rel. Error (%)',
                    'λ',
                    'Reg. Order',
                    'Method',
                    'n_tau'
                ],
                'Value': [
                    result['stats']['tau_at_max'],
                    result['stats']['gamma_max'],
                    result['stats']['total_R'],
                    result['rmse'],
                    result['rel_error']*100,
                    result['lambda_param'],
                    result['reg_order'],
                    result['method'],
                    result['n_tau']
                ]
            })
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Sheet 2: 피크
            if result['peaks_df'] is not None and len(result['peaks_df']) > 0:
                result['peaks_df'].to_excel(writer, sheet_name='Peaks', index=False)
            
            # Sheet 3: 원본 데이터
            data_df = pd.DataFrame({
                'Frequency (Hz)': result['freq'],
                "Z' (Ω)": result['z_real'],
                "Z'' (Ω)": result['z_imag'],
                "Z'' Recon (Ω)": result['z_imag_recon'],
                'Residual': result['residual']
            })
            data_df.to_excel(writer, sheet_name='Data', index=False)
            
            # Sheet 4: DRT
            drt_df = pd.DataFrame({
                'τ (s)': result['tau'],
                'log₁₀(τ)': np.log10(result['tau']),
                'γ(τ) (A/Ω)': result['gamma']
            })
            drt_df.to_excel(writer, sheet_name='DRT', index=False)
        
        output.seek(0)
        st.download_button(
            label="📊 Excel 다운로드",
            data=output.getvalue(),
            file_name="drt_result.xlsx",
            mime="application/vnd.ms-excel",
            use_container_width=True
        )
    
    with col2:
        st.info("💾 CSV 형식은 추가 개발 예정")

# ===== 도움말 =====
else:
    st.title("⚡ DRT 분석 도구")
    st.write("""
    ### 👋 사용 가이드
    
    1. **데이터 업로드**: 좌측 사이드바에서 EIS 파일(CSV/Excel)을 업로드하세요
    2. **파라미터 설정**: τ 그리드, 규제화 강도(λ) 등을 조정합니다
    3. **분석 실행**: "🚀 분석 시작" 버튼을 클릭합니다
    4. **결과 확인**: Nyquist, Bode, DRT, 재구성 등 5개 탭에서 결과를 확인합니다
    5. **결과 저장**: Excel 형식으로 다운로드합니다
    
    ### 📚 기본 개념
    
    **DRT (Distribution of Relaxation Times)**는 EIS 데이터를 주파수 영역에서 시간상수(τ) 영역으로 변환하는 분석 방법입니다.
    
    - **τ (시간상수)**: 각 프로세스의 특성 시간 스케일
    - **γ(τ)**: 특정 τ에서의 저항 기여도
    - **λ (규제화 강도)**: 작을수록 데이터 적합성 ↑, 클수록 노이즈 내성 ↑
    
    ### 🧪 테스트
    
    처음 사용하시면 좌측에서 "합성 데이터"를 선택해 테스트해보세요!
    """)
    
    st.info("💡 사이드바에서 파일을 업로드하면 분석을 시작할 수 있습니다.")


# ===== Footer =====
st.divider()
st.caption("""
**DRT Analysis Tool v0.1**  
Based on pyDRTtools (Ciucci's Lab, HKUST)  
Reference: Wan et al. (2015), Liu & Ciucci (2019)  
[GitHub](https://github.com/ciuccislab/pyDRTtools)
""")
