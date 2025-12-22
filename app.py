"""
DRT Analysis Tool - Streamlit Web Application
==============================================
Interactive tool for EIS data analysis using DRT transformation.

License: MIT
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
import json
from datetime import datetime

from drt_core import DRTAnalyzer, generate_test_eis

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="DRT Analysis Tool",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR - INPUT SETTINGS
# ============================================================================

st.sidebar.markdown("## ⚙️ Input Data")

data_source = st.sidebar.radio(
    "Data Source",
    ["Upload File", "Sample Data"],
    help="Choose between uploading your own EIS data or using sample test data"
)

freq = None
z_real = None
z_imag = None
metadata = {}

if data_source == "Upload File":
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV or Excel file",
        type=["csv", "xlsx", "xls"],
        help="Format: frequency(Hz) | Z_real(Ω) | Z_imag(Ω)"
    )
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Auto-detect column names
            freq_col = None
            z_real_col = None
            z_imag_col = None
            
            col_names_lower = [c.lower() for c in df.columns]
            
            for i, cn in enumerate(col_names_lower):
                if any(x in cn for x in ['freq', 'f', 'ω']):
                    freq_col = df.columns[i]
                elif any(x in cn for x in ["z'", 'zreal', 'zr', "z'", 're']):
                    z_real_col = df.columns[i]
                elif any(x in cn for x in ['z"', 'zimag', 'zi', "z''", 'im']):
                    z_imag_col = df.columns[i]
            
            # Fallback to first 3 columns if auto-detect fails
            if freq_col is None or z_real_col is None or z_imag_col is None:
                st.sidebar.warning("Could not auto-detect columns. Using first 3 columns.")
                freq_col, z_real_col, z_imag_col = df.columns[0], df.columns[1], df.columns[2]
            
            freq = df[freq_col].values
            z_real = df[z_real_col].values
            z_imag = np.abs(df[z_imag_col].values)  # Ensure positive
            
            st.sidebar.success(f"✓ Loaded {len(freq)} points")
            
            # Metadata
            if 'date' in df.columns or 'Date' in df.columns:
                metadata['date'] = str(df['Date' if 'Date' in df.columns else 'date'].iloc[0])
            
        except Exception as e:
            st.sidebar.error(f"Error loading file: {e}")

else:  # Sample Data
    sample_type = st.sidebar.selectbox(
        "Sample Circuit Type",
        ["RC", "RC-RC", "Randles"],
        help="Type of equivalent circuit to simulate"
    )
    
    n_points = st.sidebar.slider(
        "Number of frequency points",
        20, 200, 50
    )
    
    freq_range = st.sidebar.slider(
        "Frequency range (log scale)",
        0.0, 8.0, (1.0, 6.0),
        help="Min and max frequency in powers of 10 (Hz)"
    )
    
    frequency = np.logspace(freq_range[0], freq_range[1], n_points)
    freq, z_real, z_imag = generate_test_eis(sample_type, frequency)
    metadata['sample_type'] = sample_type
    st.sidebar.success(f"✓ Generated {sample_type} sample data")

# ============================================================================
# SIDEBAR - DRT PARAMETERS
# ============================================================================

st.sidebar.markdown("## 🔧 DRT Parameters")

n_tau = st.sidebar.slider(
    "τ grid points",
    30, 300, 100,
    help="Number of time constant points"
)

lambda_auto = st.sidebar.checkbox(
    "Auto-select λ (GCV)",
    value=True,
    help="Automatically find optimal regularization parameter"
)

if not lambda_auto:
    lambda_val = st.sidebar.slider(
        "Regularization λ",
        -8, 2, -4,
        help="Log10(λ) value"
    )
    lambda_val = 10.0 ** lambda_val
else:
    lambda_val = None

non_negative = st.sidebar.checkbox(
    "Non-negative γ(τ)",
    value=False,
    help="Enforce γ(τ) ≥ 0 (physical constraint)"
)

# ============================================================================
# MAIN CONTENT
# ============================================================================

st.markdown('<p class="main-header">⚡ DRT Analysis Tool</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Distribution of Relaxation Times from Electrochemical Impedance Spectroscopy</p>', unsafe_allow_html=True)

if freq is None:
    st.info("👈 Select data source from sidebar to get started")
    st.stop()

# ============================================================================
# RUN DRT ANALYSIS
# ============================================================================

col1, col2, col3 = st.columns(3)

with col1:
    run_button = st.button("▶️ Run DRT Analysis", use_container_width=True, type="primary")

with col2:
    clear_button = st.button("🔄 Clear Cache", use_container_width=True)

with col3:
    st.write("")  # Spacer

if clear_button:
    st.cache_data.clear()
    st.rerun()

if run_button or 'drt_analyzer' not in st.session_state:
    
    # Run analysis
    with st.spinner("⏳ Analyzing... (this may take a few seconds)"):
        analyzer = DRTAnalyzer()
        analyzer.load_data(freq, z_real, z_imag)
        success = analyzer.solve_drt(
            n_tau=n_tau,
            lambda_val=lambda_val,
            lambda_auto=lambda_auto,
            non_negative=non_negative,
            verbose=False
        )
    
    if success:
        st.session_state.drt_analyzer = analyzer
        st.session_state.freq = freq
        st.session_state.z_real = z_real
        st.session_state.z_imag = z_imag
        st.session_state.metadata = metadata
        st.rerun()

if 'drt_analyzer' in st.session_state:
    analyzer = st.session_state.drt_analyzer
    freq = st.session_state.freq
    z_real = st.session_state.z_real
    z_imag = st.session_state.z_imag
    
    st.success("✅ Analysis completed successfully!")
    
    # ========================================================================
    # RESULTS TABS
    # ========================================================================
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 DRT Plot", 
        "🌀 Nyquist/Bode",
        "📈 Fit Quality",
        "🎯 Peaks",
        "💾 Export"
    ])
    
    # TAB 1: DRT PLOT
    # ====================================================================
    
    with tab1:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            fig_drt = go.Figure()
            
            fig_drt.add_trace(go.Scatter(
                x=analyzer.tau_grid,
                y=analyzer.gamma,
                mode='lines',
                name='γ(τ)',
                line=dict(color='#1f77b4', width=3),
                fill='tozeroy',
                fillcolor='rgba(31, 119, 180, 0.3)'
            ))
            
            # Mark peaks
            if analyzer.peaks_info:
                peak_taus = [p['tau'] for p in analyzer.peaks_info]
                peak_gammas = [p['gamma'] for p in analyzer.peaks_info]
                
                fig_drt.add_trace(go.Scatter(
                    x=peak_taus,
                    y=peak_gammas,
                    mode='markers',
                    name='Peaks',
                    marker=dict(color='red', size=10, symbol='star')
                ))
            
            fig_drt.update_xaxes(type='log', title='Time Constant τ (s)')
            fig_drt.update_yaxes(title='γ(τ) (Ω/log(s))')
            fig_drt.update_layout(
                title='Distribution of Relaxation Times (DRT)',
                height=500,
                hovermode='x unified',
                template='plotly_white'
            )
            
            st.plotly_chart(fig_drt, use_container_width=True)
        
        with col2:
            summary = analyzer.get_summary()
            
            st.markdown("### Summary")
            st.metric("Peaks", summary['n_peaks'])
            st.metric("λ (optimal)", f"{summary['lambda_opt']:.2e}")
            st.metric("γ_max", f"{summary['gamma_max']:.2e} Ω")
            st.metric("Total R", f"{summary['total_resistance']:.1f} Ω")
    
    # TAB 2: NYQUIST & BODE
    # ====================================================================
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            # Nyquist Plot
            fig_nyquist = go.Figure()
            
            fig_nyquist.add_trace(go.Scatter(
                x=z_real,
                y=z_imag,
                mode='markers',
                name='Experimental',
                marker=dict(color='blue', size=8),
                text=[f'{f:.0f} Hz' for f in freq],
                hoverinfo='text'
            ))
            
            fig_nyquist.add_trace(go.Scatter(
                x=analyzer.Z_reconst_real,
                y=analyzer.Z_reconst_imag,
                mode='lines',
                name='Reconstructed',
                line=dict(color='red', width=2, dash='dash')
            ))
            
            fig_nyquist.update_xaxes(title="Z' (Ω)")
            fig_nyquist.update_yaxes(title="-Z'' (Ω)")
            fig_nyquist.update_layout(
                title='Nyquist Plot',
                height=500,
                hovermode='closest',
                template='plotly_white'
            )
            
            st.plotly_chart(fig_nyquist, use_container_width=True)
        
        with col2:
            # Bode Plot
            mag = np.sqrt(z_real**2 + z_imag**2)
            phase = np.arctan2(z_imag, z_real) * 180 / np.pi
            
            mag_reconst = np.sqrt(analyzer.Z_reconst_real**2 + analyzer.Z_reconst_imag**2)
            phase_reconst = np.arctan2(analyzer.Z_reconst_imag, analyzer.Z_reconst_real) * 180 / np.pi
            
            fig_bode = go.Figure()
            
            fig_bode.add_trace(go.Scatter(
                x=freq, y=mag,
                mode='markers',
                name='|Z| (exp)',
                marker=dict(color='blue', size=6),
                yaxis='y1'
            ))
            
            fig_bode.add_trace(go.Scatter(
                x=freq, y=mag_reconst,
                mode='lines',
                name='|Z| (fit)',
                line=dict(color='red', dash='dash'),
                yaxis='y1'
            ))
            
            fig_bode.add_trace(go.Scatter(
                x=freq, y=phase,
                mode='markers',
                name='Phase (exp)',
                marker=dict(color='green', size=6),
                yaxis='y2'
            ))
            
            fig_bode.add_trace(go.Scatter(
                x=freq, y=phase_reconst,
                mode='lines',
                name='Phase (fit)',
                line=dict(color='orange', dash='dash'),
                yaxis='y2'
            ))
            
            fig_bode.update_xaxes(type='log', title='Frequency (Hz)')
            fig_bode.update_yaxes(title='|Z| (Ω)', secondary_y=False)
            fig_bode.update_layout(
                yaxis2=dict(title='Phase (°)', overlaying='y', side='right'),
                title='Bode Plot',
                height=500,
                hovermode='x unified',
                template='plotly_white'
            )
            
            st.plotly_chart(fig_bode, use_container_width=True)
    
    # TAB 3: FIT QUALITY
    # ====================================================================
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Fit Statistics")
            metrics = analyzer.get_summary()['fit_metrics']
            metrics_residual = analyzer.get_summary()['residual_stats']
            
            st.metric("Data Points", metrics['n_points'])
            st.metric("RMSE", f"{metrics_residual['rmse']:.2e} Ω")
            st.metric("Relative Error", f"{metrics_residual['relative_error']:.4f}")
            st.metric("Max Error", f"{metrics_residual['max_error']:.2e} Ω")
        
        with col2:
            # Residual plot
            n_freq = len(freq)
            residual_real = analyzer.residual['abs'][:n_freq]
            residual_imag = analyzer.residual['abs'][n_freq:]
            
            fig_residual = go.Figure()
            
            fig_residual.add_trace(go.Scatter(
                x=freq, y=residual_real,
                mode='markers',
                name="Residual Z'",
                marker=dict(color='blue')
            ))
            
            fig_residual.add_trace(go.Scatter(
                x=freq, y=residual_imag,
                mode='markers',
                name='Residual Z"',
                marker=dict(color='red')
            ))
            
            fig_residual.update_xaxes(type='log', title='Frequency (Hz)')
            fig_residual.update_yaxes(title='Residual (Ω)')
            fig_residual.update_layout(
                title='Fit Residuals',
                height=400,
                template='plotly_white'
            )
            
            st.plotly_chart(fig_residual, use_container_width=True)
    
    # TAB 4: PEAKS ANALYSIS
    # ====================================================================
    
    with tab4:
        if analyzer.peaks_info:
            st.markdown("### Detected Peaks in DRT")
            
            peaks_df = pd.DataFrame([
                {
                    'Peak #': i+1,
                    'τ (s)': f"{p['tau']:.2e}",
                    'γ (Ω)': f"{p['gamma']:.2e}",
                    'Resistance (Ω)': f"{p['resistance']:.2f}",
                    'f @ τ (Hz)': f"{1/(2*np.pi*p['tau']):.2e}"
                }
                for i, p in enumerate(analyzer.peaks_info)
            ])
            
            st.dataframe(peaks_df, use_container_width=True)
            
            # Peak decomposition chart
            fig_peaks = go.Figure()
            
            fig_peaks.add_trace(go.Scatter(
                x=analyzer.tau_grid,
                y=analyzer.gamma,
                mode='lines',
                name='Total DRT',
                line=dict(color='black', width=2)
            ))
            
            colors = px.colors.qualitative.Plotly
            for i, peak in enumerate(analyzer.peaks_info):
                tau_range = np.logspace(
                    np.log10(peak['fwhm_range'][0]),
                    np.log10(peak['fwhm_range'][1]),
                    50
                )
                idx_range = [np.argmin(np.abs(analyzer.tau_grid - t)) for t in tau_range]
                
                fig_peaks.add_trace(go.Scatter(
                    x=tau_range,
                    y=analyzer.gamma[idx_range],
                    mode='lines',
                    name=f"Peak {i+1} (τ={peak['tau']:.2e}s)",
                    line=dict(color=colors[i % len(colors)]),
                    fill='tozeroy'
                ))
            
            fig_peaks.update_xaxes(type='log', title='Time Constant τ (s)')
            fig_peaks.update_yaxes(title='γ(τ) (Ω/log(s))')
            fig_peaks.update_layout(
                title='Peak Decomposition',
                height=500,
                template='plotly_white'
            )
            
            st.plotly_chart(fig_peaks, use_container_width=True)
        
        else:
            st.info("No significant peaks detected")
    
    # TAB 5: EXPORT
    # ====================================================================
    
    with tab5:
        st.markdown("### Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        # Export Excel
        with col1:
            if st.button("📥 Download Excel", use_container_width=True):
                excel_buffer = BytesIO()
                
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    # Sheet 1: Summary
                    summary = analyzer.get_summary()
                    summary_df = pd.DataFrame({
                        'Parameter': ['Measurement Date', 'Num Data Points', 'Num Tau Points',
                                    'Num Peaks', 'Optimal λ', 'RMSE', 'Relative Error', 'Max Error'],
                        'Value': [
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            summary['fit_metrics']['n_points'],
                            len(analyzer.tau_grid),
                            summary['n_peaks'],
                            f"{summary['lambda_opt']:.2e}",
                            f"{summary['residual_stats']['rmse']:.2e}",
                            f"{summary['residual_stats']['relative_error']:.4f}",
                            f"{summary['residual_stats']['max_error']:.2e}"
                        ]
                    })
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)
                    
                    # Sheet 2: DRT
                    drt_df = pd.DataFrame({
                        'tau (s)': analyzer.tau_grid,
                        'gamma (Ω)': analyzer.gamma
                    })
                    drt_df.to_excel(writer, sheet_name='DRT', index=False)
                    
                    # Sheet 3: Raw EIS
                    eis_df = pd.DataFrame({
                        'Frequency (Hz)': freq,
                        'Z_real (Ω)': z_real,
                        'Z_imag (Ω)': z_imag,
                        'Z_real_reconst (Ω)': analyzer.Z_reconst_real,
                        'Z_imag_reconst (Ω)': analyzer.Z_reconst_imag
                    })
                    eis_df.to_excel(writer, sheet_name='EIS Data', index=False)
                    
                    # Sheet 4: Peaks
                    if analyzer.peaks_info:
                        peaks_export = pd.DataFrame([
                            {
                                'Peak #': i+1,
                                'tau (s)': p['tau'],
                                'gamma (Ω)': p['gamma'],
                                'Resistance (Ω)': p['resistance'],
                                'Frequency @ tau (Hz)': 1/(2*np.pi*p['tau'])
                            }
                            for i, p in enumerate(analyzer.peaks_info)
                        ])
                        peaks_export.to_excel(writer, sheet_name='Peaks', index=False)
                
                excel_buffer.seek(0)
                st.download_button(
                    label="✅ Download Excel",
                    data=excel_buffer,
                    file_name=f"DRT_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
        
        # Export JSON
        with col2:
            if st.button("📥 Download JSON", use_container_width=True):
                export_data = {
                    'metadata': {
                        'analysis_date': datetime.now().isoformat(),
                        'n_data_points': len(freq),
                        'n_tau_points': len(analyzer.tau_grid)
                    },
                    'parameters': {
                        'lambda_opt': float(analyzer.lambda_opt),
                        'n_peaks': len(analyzer.peaks_info) if analyzer.peaks_info else 0
                    },
                    'fit_quality': {
                        'rmse': float(analyzer.residual['rmse']),
                        'relative_error': float(analyzer.residual['relative_error']),
                        'max_error': float(analyzer.residual['max_error'])
                    },
                    'drt': {
                        'tau': analyzer.tau_grid.tolist(),
                        'gamma': analyzer.gamma.tolist()
                    },
                    'peaks': [
                        {
                            'tau': float(p['tau']),
                            'gamma': float(p['gamma']),
                            'resistance': float(p['resistance'])
                        }
                        for p in (analyzer.peaks_info or [])
                    ]
                }
                
                json_str = json.dumps(export_data, indent=2)
                st.download_button(
                    label="✅ Download JSON",
                    data=json_str,
                    file_name=f"DRT_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
        
        with col3:
            st.info("💾 Formats: Excel, JSON")
        
        # Data table
        st.markdown("### DRT Data Table")
        
        drt_table = pd.DataFrame({
            'τ (s)': analyzer.tau_grid,
            'γ(τ) (Ω)': analyzer.gamma,
            'log(τ)': np.log10(analyzer.tau_grid)
        })
        
        st.dataframe(
            drt_table.style.format({
                'τ (s)': '{:.2e}',
                'γ(τ) (Ω)': '{:.2e}',
                'log(τ)': '{:.2f}'
            }),
            use_container_width=True
        )

# ============================================================================
# FOOTER
# ============================================================================

st.divider()

st.markdown("""
**DRT Analysis Tool v1.0**  
⚡ Electrochemical Impedance Spectroscopy Analysis  
📄 Built with Streamlit | Powered by Tikhonov Regularization  
📚 Theory: Ciucci et al., Joule (2022) | ChemElectroChem (2019)  
📖 Docs: [GitHub](https://github.com) | License: MIT
""")
