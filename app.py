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
    uploaded_files = st.sidebar.file_uploader(
        "Upload CSV, Excel, or TXT files (multiple files for overlay comparison)",
        type=["csv", "xlsx", "xls", "txt"],
        accept_multiple_files=True,  # Enable multiple file upload
        help="Format: frequency(Hz) | Z_real(Ω) | -Z_imag(Ω)"
    )
    
    # Store all datasets
    datasets = []
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith('.txt'):
                    df = pd.read_csv(uploaded_file, sep='\t', skiprows=0)
                else:
                    df = pd.read_excel(uploaded_file)
                
                # Auto-detect column names
                freq_col = None
                z_real_col = None
                z_imag_col = None
                
                col_names_lower = [c.lower().strip() for c in df.columns]
                
                for i, cn in enumerate(col_names_lower):
                    if any(x in cn for x in ['freq', 'f ', 'ω', 'hz']):
                        freq_col = df.columns[i]
                    elif any(x in cn for x in ["z'", 'zreal', 'zr', "z'", 're(z)', 'real']):
                        z_real_col = df.columns[i]
                    elif any(x in cn for x in ['z"', 'zimag', 'zi', "z''", 'im(z)', 'imag', '-im(z)']):
                        z_imag_col = df.columns[i]
                
                # Fallback to first 3 columns if auto-detect fails
                if freq_col is None or z_real_col is None or z_imag_col is None:
                    freq_col, z_real_col, z_imag_col = df.columns[0], df.columns[1], df.columns[2]
                
                freq_data = df[freq_col].values
                z_real_data = df[z_real_col].values
                z_imag_data = np.abs(df[z_imag_col].values)  # Ensure positive
                
                datasets.append({
                    'filename': uploaded_file.name,
                    'freq': freq_data,
                    'z_real': z_real_data,
                    'z_imag': z_imag_data,
                    'n_points': len(freq_data)
                })
                
            except Exception as e:
                st.sidebar.error(f"Error loading {uploaded_file.name}: {e}")
        
        if datasets:
            st.sidebar.success(f"✓ Loaded {len(datasets)} file(s)")
            # For backward compatibility, use first dataset as primary
            freq = datasets[0]['freq']
            z_real = datasets[0]['z_real']
            z_imag = datasets[0]['z_imag']
            metadata['num_datasets'] = len(datasets)
            # Store all datasets in session state for overlay
            st.session_state.datasets = datasets

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

# τ grid points: Number input instead of slider
n_tau = st.sidebar.number_input(
    "τ grid points",
    min_value=30,
    max_value=500,
    value=100,
    step=10,
    help="Number of time constant grid points for DRT analysis"
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
    value=True,
    help="Enforce γ(τ) ≥ 0 (physical constraint for impedance) - Recommended: ON"
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

# ============================================================================
# MULTI-FILE ANALYSIS
# ============================================================================

if run_button or 'drt_analyzers' not in st.session_state:
    
    # Prepare datasets
    if data_source == "Upload File" and 'datasets' in st.session_state and st.session_state.datasets:
        datasets_to_analyze = st.session_state.datasets
    elif freq is not None:
        datasets_to_analyze = [{
            'filename': metadata.get('sample_type', 'Data'),
            'freq': freq,
            'z_real': z_real,
            'z_imag': z_imag
        }]
    else:
        st.error("No data to analyze")
        datasets_to_analyze = []
    
    if datasets_to_analyze:
        analyzers = []
        
        with st.spinner(f"⏳ Analyzing {len(datasets_to_analyze)} dataset(s)... (this may take a few seconds)"):
            for dataset in datasets_to_analyze:
                analyzer = DRTAnalyzer()
                analyzer.load_data(dataset['freq'], dataset['z_real'], dataset['z_imag'])
                success = analyzer.solve_drt(
                    n_tau=n_tau,
                    lambda_val=lambda_val,
                    lambda_auto=lambda_auto,
                    non_negative=non_negative,
                    verbose=False
                )
                
                if success:
                    analyzers.append({
                        'filename': dataset['filename'],
                        'analyzer': analyzer,
                        'freq': dataset['freq'],
                        'z_real': dataset['z_real'],
                        'z_imag': dataset['z_imag']
                    })
        
        if analyzers:
            st.session_state.drt_analyzers = analyzers
            st.session_state.datasets_analyzed = datasets_to_analyze
            st.rerun()

if 'drt_analyzers' in st.session_state:
    analyzers_data = st.session_state.drt_analyzers
    
    st.success(f"✅ Analysis completed successfully! ({len(analyzers_data)} dataset(s))")
    
    # ========================================================================
    # RESULTS TABS
    # ========================================================================
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 DRT Plot (Overlay)", 
        "🌀 Nyquist/Bode",
        "📈 Fit Quality",
        "🎯 Peaks",
        "💾 Export"
    ])
    
    # TAB 1: DRT PLOT - MULTI-FILE OVERLAY
    # ====================================================================
    
    with tab1:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Axis range controls
            st.markdown("### 🔍 Axis Range (for peak comparison)")
            axis_col1, axis_col2 = st.columns(2)
            
            with axis_col1:
                st.markdown("**X-Axis (τ range)**")
                x_min_exp = st.number_input("X-min (10^x s)", value=-6, step=1, help="Minimum log10(τ)")
                x_max_exp = st.number_input("X-max (10^x s)", value=0, step=1, help="Maximum log10(τ)")
            
            with axis_col2:
                st.markdown("**Y-Axis (γ' range)**")
                y_min = st.number_input("Y-min (Ω)", value=0, step=1000, help="Minimum γ'(ln τ)")
                y_max = st.number_input("Y-max (Ω)", value=0, step=1000, help="Maximum γ'(ln τ) (0=auto)")
            
            # Create overlay plot with all datasets
            fig_drt = go.Figure()
            
            # Color palette for multiple files
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
            
            # Add all DRT curves
            for idx, analyzer_info in enumerate(analyzers_data):
                analyzer = analyzer_info['analyzer']
                filename = analyzer_info['filename']
                color = colors[idx % len(colors)]
                
                # Convert γ(τ) to differential form: γ'(ln τ) = γ(τ)/ln(10)
                gamma_differential = analyzer.gamma / np.log(10)
                
                fig_drt.add_trace(go.Scatter(
                    x=analyzer.tau_grid,
                    y=gamma_differential,
                    mode='lines',
                    name=f"{filename.replace('.csv', '').replace('.txt', '').replace('.xlsx', '')}",
                    line=dict(color=color, width=2.5),
                    fill='tozeroy' if idx == 0 else None,
                    fillcolor='rgba(31, 119, 180, 0.2)' if idx == 0 else None,
                    hovertemplate='<b>' + filename + '</b><br>τ=%{x:.2e}s<br>γ\'=%{y:.2e}Ω<extra></extra>'
                ))
                
                # Mark peaks for this dataset
                if analyzer.peaks_info:
                    peak_taus = [p['tau'] for p in analyzer.peaks_info]
                    peak_gammas = [p['gamma'] / np.log(10) for p in analyzer.peaks_info]
                    
                    fig_drt.add_trace(go.Scatter(
                        x=peak_taus,
                        y=peak_gammas,
                        mode='markers',
                        name=f"{filename.replace('.csv', '').replace('.txt', '').replace('.xlsx', '')} - Peaks",
                        marker=dict(color=color, size=8, symbol='star', line=dict(color='black', width=1)),
                        hovertemplate='<b>' + filename + '</b><br>τ=%{x:.2e}s<br>γ\'=%{y:.2e}Ω<extra></extra>',
                        showlegend=True
                    ))
            
            fig_drt.update_xaxes(
                type='log',
                title='Time Constant τ (s)',
                titlefont=dict(size=12),
                range=[x_min_exp, x_max_exp]
            )
            
            y_range = [y_min, y_max if y_max > 0 else None]
            fig_drt.update_yaxes(
                title="γ'(ln τ) (Ω)",
                titlefont=dict(size=12),
                range=y_range
            )
            
            fig_drt.update_layout(
                title=f'Distribution of Relaxation Times - {len(analyzers_data)} Dataset(s) Overlay',
                height=600,
                hovermode='x unified',
                template='plotly_white',
                font=dict(size=11),
                legend=dict(x=1.02, y=1, xanchor='left', yanchor='top')
            )
            
            st.plotly_chart(fig_drt, use_container_width=True)
        
        with col2:
            st.markdown("### 📊 Summary")
            st.metric("Datasets", len(analyzers_data))
            
            for idx, analyzer_info in enumerate(analyzers_data):
                analyzer = analyzer_info['analyzer']
                filename = analyzer_info['filename']
                summary = analyzer.get_summary()
                
                st.markdown(f"**{idx+1}. {filename}**")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Peaks", summary['n_peaks'], label_visibility="collapsed")
                with col_b:
                    st.metric("λ opt", f"{summary['lambda_opt']:.0e}", label_visibility="collapsed")
                col_c, col_d = st.columns(2)
                with col_c:
                    st.metric("γ_max", f"{summary['gamma_max']:.0e}Ω", label_visibility="collapsed")
                with col_d:
                    st.metric("Total R", f"{summary['total_resistance']:.0f}Ω", label_visibility="collapsed")
                st.divider()
    
    # TAB 2: NYQUIST & BODE - MULTI-FILE OVERLAY
    # ====================================================================
    
    with tab2:
        st.markdown(f"### 📊 EIS Comparison - {len(analyzers_data)} Dataset(s)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Nyquist Plot - All datasets overlay
            fig_nyquist = go.Figure()
            
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
            
            for idx, analyzer_info in enumerate(analyzers_data):
                analyzer = analyzer_info['analyzer']
                freq_data = analyzer_info['freq']
                z_real_data = analyzer_info['z_real']
                z_imag_data = analyzer_info['z_imag']
                filename = analyzer_info['filename']
                color = colors[idx % len(colors)]
                
                # Experimental data
                fig_nyquist.add_trace(go.Scatter(
                    x=z_real_data,
                    y=z_imag_data,
                    mode='markers',
                    name=f'{filename.replace(".csv", "").replace(".txt", "").replace(".xlsx", "")} (Exp)',
                    marker=dict(color=color, size=6, opacity=0.7),
                    text=[f'{f:.0f} Hz' for f in freq_data],
                    hovertemplate='<b>' + filename + ' (Exp)</b><br>Z\'=%{x:.2f}Ω<br>-Z"=%{y:.2f}Ω<extra></extra>'
                ))
                
                # DRT Reconstructed
                fig_nyquist.add_trace(go.Scatter(
                    x=analyzer.Z_reconst_real,
                    y=analyzer.Z_reconst_imag,
                    mode='lines',
                    name=f'{filename.replace(".csv", "").replace(".txt", "").replace(".xlsx", "")} (DRT)',
                    line=dict(color=color, width=2.5, dash='dash'),
                    hovertemplate='<b>' + filename + ' (DRT)</b><br>Z\'=%{x:.2f}Ω<br>-Z"=%{y:.2f}Ω<extra></extra>'
                ))
            
            fig_nyquist.update_xaxes(title="Z' (Ω)", titlefont=dict(size=12))
            fig_nyquist.update_yaxes(title="-Z'' (Ω)", titlefont=dict(size=12))
            fig_nyquist.update_layout(
                title='Nyquist Plot: All Datasets Overlay',
                height=500,
                hovermode='closest',
                template='plotly_white',
                font=dict(size=11),
                legend=dict(x=0.02, y=0.98, font=dict(size=9))
            )
            
            st.plotly_chart(fig_nyquist, use_container_width=True)
        
        with col2:
            # Bode Plot - All datasets overlay
            fig_bode = go.Figure()
            
            for idx, analyzer_info in enumerate(analyzers_data):
                analyzer = analyzer_info['analyzer']
                freq_data = analyzer_info['freq']
                z_real_data = analyzer_info['z_real']
                z_imag_data = analyzer_info['z_imag']
                filename = analyzer_info['filename']
                color = colors[idx % len(colors)]
                
                # Magnitude
                mag = np.sqrt(z_real_data**2 + z_imag_data**2)
                mag_reconst = np.sqrt(analyzer.Z_reconst_real**2 + analyzer.Z_reconst_imag**2)
                
                # Experimental magnitude
                fig_bode.add_trace(go.Scatter(
                    x=freq_data, y=mag,
                    mode='markers',
                    name=f'{filename.replace(".csv", "").replace(".txt", "").replace(".xlsx", "")} |Z|',
                    marker=dict(color=color, size=5, opacity=0.7),
                    yaxis='y',
                    hovertemplate='<b>' + filename + '</b><br>f=%{x:.0f} Hz<br>|Z|=%{y:.2f}Ω<extra></extra>'
                ))
                
                # DRT reconstructed magnitude
                fig_bode.add_trace(go.Scatter(
                    x=freq_data, y=mag_reconst,
                    mode='lines',
                    name=f'{filename.replace(".csv", "").replace(".txt", "").replace(".xlsx", "")} |Z| (DRT)',
                    line=dict(color=color, width=2, dash='dash'),
                    yaxis='y',
                    hovertemplate='<b>' + filename + ' (DRT)</b><br>f=%{x:.0f} Hz<br>|Z|=%{y:.2f}Ω<extra></extra>'
                ))
            
            fig_bode.update_xaxes(type='log', title='Frequency (Hz)', titlefont=dict(size=12))
            fig_bode.update_yaxes(type='log', title='|Z| (Ω)', titlefont=dict(size=12))
            fig_bode.update_layout(
                title='Bode Plot: All Datasets Overlay',
                height=500,
                hovermode='x unified',
                template='plotly_white',
                font=dict(size=11),
                legend=dict(x=0.02, y=0.98, font=dict(size=9))
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
