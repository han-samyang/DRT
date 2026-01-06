import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize
from GP_DRT import NMLL_fct, grad_NMLL_fct, matrix_L2_im_K, is_PD, nearest_PD

# -------------------------------------------------------------------
# Fixed figure size for BOTH DRT and Nyquist (independent of scale)
# -------------------------------------------------------------------
FIG_W, FIG_H = 7, 7  # inches

st.set_page_config(layout="wide")
st.title("⚡ GP-DRT 분석")

uploaded_files = st.file_uploader("파일 선택 (여러 파일 가능)", type=["txt", "csv"], accept_multiple_files=True)
if not uploaded_files:
    st.stop()

def analyze_one(uploaded_file):
    file_id = getattr(uploaded_file, 'name', 'file').replace('.', '_')
    st.markdown(f"### 📄 {uploaded_file.name}")
    # -------------------------------------------------------------------
    # Load file
    # -------------------------------------------------------------------
    if uploaded_file.name.endswith(".txt"):
        df = pd.read_csv(uploaded_file, sep="\t", engine="python")
    else:
        df = pd.read_csv(uploaded_file)

    df.columns = [c.strip() for c in df.columns]
    st.write("📋 컬럼:", df.columns.tolist())

    freq = df.iloc[:, 0].values.astype(float)
    z_real = df.iloc[:, 1].values.astype(float)
    z_imag_in = df.iloc[:, 2].values.astype(float)

    mask = np.isfinite(freq) & np.isfinite(z_real) & np.isfinite(z_imag_in)
    freq = freq[mask]
    z_real = z_real[mask]
    z_imag_in = z_imag_in[mask]

    # Complex impedance for GP-DRT (input column is -Im(Z))
    Z_exp = z_real + 1j * (-z_imag_in)

    # -------------------------------------------------------------------
    # (Nyquist plot removed)
    # -------------------------------------------------------------------

    st.success(f"✅ {len(freq)} 포인트 로드됨")

    # -------------------------------------------------------------------
    # GP-DRT optimization
    # -------------------------------------------------------------------
    st.info("⏳ 분석 중...")

    xi_vec = np.log(2 * np.pi * freq)

    result = optimize.minimize(
        NMLL_fct,
        [0.01, 10.0, 1.0],
        args=(Z_exp, xi_vec),
        method="L-BFGS-B",
        jac=grad_NMLL_fct,
        bounds=[(0.001, 1.0), (0.1, 100.0), (0.01, 10.0)],
        options={"ftol": 1e-6, "gtol": 1e-6},
    )

    sigma_n_opt, sigma_f_opt, ell_opt = result.x

    L2_im_K = matrix_L2_im_K(xi_vec, xi_vec, sigma_f_opt, ell_opt)
    Sigma = (sigma_n_opt**2) * np.eye(len(xi_vec))
    K = L2_im_K + Sigma
    if not is_PD(K):
        K = nearest_PD(K)

    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L, Z_exp.imag)
    alpha = np.linalg.solve(L.T, alpha)

    tau = 1 / (2 * np.pi * freq)  # tau = 1/(2πf); note freq is sorted asc so tau desc before re-sorting in plot
    gamma = 2 * np.pi * np.exp(xi_vec) * alpha

    st.success("✅ 분석 완료!")

    # -------------------------------------------------------------------
    # Helpers: positive peak segmentation on y>0 in log-x space
    # -------------------------------------------------------------------
    def positive_peak_segments(logx: np.ndarray, y: np.ndarray):
        """Return list of (logx_seg, y_seg) where y>0 continuously.
        Adds interpolated y=0 crossing points at segment boundaries for clean fill/area.
        """
        segs = []
        n = len(y)
        i = 0
        while i < n:
            if not np.isfinite(y[i]) or y[i] <= 0:
                i += 1
                continue

            start = i
            while i < n and np.isfinite(y[i]) and y[i] > 0:
                i += 1
            end = i - 1

            # build with boundary crossings
            lx = []
            yy = []

            # left boundary
            if start - 1 >= 0 and np.isfinite(y[start - 1]) and y[start - 1] <= 0 < y[start]:
                x0, x1 = logx[start - 1], logx[start]
                y0, y1 = y[start - 1], y[start]
                xc = x0 + (0 - y0) * (x1 - x0) / (y1 - y0)
                lx.append(xc); yy.append(0.0)

            lx.append(logx[start]); yy.append(y[start])

            for k in range(start + 1, end + 1):
                lx.append(logx[k]); yy.append(y[k])

            # right boundary
            if end + 1 < n and np.isfinite(y[end + 1]) and y[end] > 0 >= y[end + 1]:
                x0, x1 = logx[end], logx[end + 1]
                y0, y1 = y[end], y[end + 1]
                xc = x0 + (0 - y0) * (x1 - x0) / (y1 - y0)
                lx.append(xc); yy.append(0.0)

            segs.append((np.array(lx, dtype=float), np.array(yy, dtype=float)))

        return segs

    # -------------------------------------------------------------------
    # UI options
    # -------------------------------------------------------------------
    st.markdown("---")
    st.markdown("## 📊 분석 결과")

    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        y_scale = st.radio("Y축 스케일:", ["선형", "로그 (절댓값)", "대칭 로그"], horizontal=True, key=f"y_scale_{file_id}")
    with col_opt2:
        x_axis = st.radio("DRT X축:", ["τ (시간상수)", "f (주파수)"], horizontal=True, key=f"x_axis_{file_id}")

    # 표시 방식 전환: g(τ) vs γ(ln τ)
    show_gamma_ln = st.checkbox("γ(ln τ)로 표시 (γ(ln τ) = τ · g(τ))", value=False, key=f"gamma_ln_{file_id}")


    col1 = st.container()  # single column (Nyquist removed)

    # -------------------------------------------------------------------
    # DRT plot
    # -------------------------------------------------------------------
    with col1:
        st.subheader("DRT")
        st.write(f"**Gamma 범위**: {np.nanmin(gamma):.2e} ~ {np.nanmax(gamma):.2e} Ω")

        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), constrained_layout=False)

        # x selection
        if x_axis == "τ (시간상수)":
            x_data = tau.copy()
            x_label = "τ (s)"
        else:
            x_data = freq.copy()
            x_label = "f (Hz)"

        # ensure monotonically increasing x for stable line + integration
        order = np.argsort(x_data)
        x_data = x_data[order]
        gamma_plot = gamma[order]
        # g(τ) / γ(ln τ) 표시 전환 (τ 축일 때만 적용)
        if x_axis == "τ (시간상수)" and show_gamma_ln:
            gamma_plot = gamma_plot * x_data  # γ(ln τ) = τ · g(τ)

        ax.set_xscale("log")

        # keep original blue point-line style (no legend)
        ax.plot(x_data, gamma_plot, "b-o", linewidth=2, markersize=4, label="_nolegend_")
        ax.axhline(0, color="k", linestyle="-", linewidth=0.5)

        # y-scale options (keep existing behavior)
        if y_scale == "선형":
            ax.set_yscale("linear")
        elif y_scale == "로그 (절댓값)":
            ax.set_yscale("log")
            ax.plot(x_data, np.abs(gamma_plot), "r--", linewidth=1, alpha=0.5, label="_nolegend_")
        else:
            ax.set_yscale("symlog", linthresh=1e15)

        # Positive peaks: fill & area on y>0 only (integrate in ln(x))
        logx = np.log(x_data)
        segs = positive_peak_segments(logx, gamma_plot)

        # area for ALL positive segments (total)
        all_pos_areas = []
        seg_info = []
        for lx_seg, y_seg in segs:
            if len(lx_seg) < 2:
                continue
            # sort by logx
            o = np.argsort(lx_seg)
            lx2 = lx_seg[o]
            y2 = y_seg[o]
            a = float(np.trapz(y2, lx2))  # should be positive
            if a > 0 and np.isfinite(a):
                all_pos_areas.append(a)
                seg_info.append((a, lx2, y2))

        total_pos_area = float(np.sum(all_pos_areas)) if all_pos_areas else 0.0

        # ONLY show peaks that are actually colored in legend:
        # To avoid tiny/noisy segments blowing up the legend, color only the top 3 areas.
        TOP_K = 3
        seg_info.sort(key=lambda x: x[0], reverse=True)
        seg_info_colored = seg_info[:TOP_K]

        colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3", "C4"])
        shown_peak_areas = []

        for i, (a, lx2, y2) in enumerate(seg_info_colored, start=1):
            shown_peak_areas.append(a)
            xx = np.exp(lx2)
            ax.fill_between(
                xx, 0, y2,
                alpha=0.25,
                color=colors[(i - 1) % len(colors)],
                label=f"Peak {i} area={a:.2e}",
            )

        # label text: total positive area + how many colored peaks shown
        if x_axis == "τ (시간상수)":
            area_label = f"∑(γ>0) ∫γ d(ln τ) = {total_pos_area:.2e}  (shown={len(shown_peak_areas)})"
        else:
            area_label = f"∑(γ>0) ∫γ d(ln f) = {total_pos_area:.2e}  (shown={len(shown_peak_areas)})"

        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel("γ(ln τ) (Ω)" if (x_axis=="τ (시간상수)" and show_gamma_ln) else "g(τ) (Ω)", fontsize=12)
        ax.grid(True, alpha=0.3)

        # legend: ONLY colored peaks
        if len(shown_peak_areas) > 0:
            ax.legend(loc="upper right")
        else:
            ax.legend([], [], frameon=False)

        ax.text(
            0.05, 0.95, area_label,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        st.pyplot(fig)

    # -------------------------------------------------------------------
    # Stats
    # -------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 📊 통계 정보")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Gamma 최대값", f"{np.nanmax(gamma):.2e} Ω")
    with c2:
        st.metric("Gamma 최소값", f"{np.nanmin(gamma):.2e} Ω")
    with c3:
        st.metric("면적 (∑(γ>0) ∫γ d(ln x))", f"{total_pos_area:.2e}")

    # -------------------------------------------------------------------
    # Download
    # -------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 📥 결과 다운로드")

    c1, c2 = st.columns(2)
    with c1:
        drt_csv = pd.DataFrame({"tau (s)": tau, "gamma (Ω)": gamma}).to_csv(index=False)
        st.download_button("📥 DRT.csv", drt_csv, "DRT.csv", "text/csv", key=f"download_drt_{file_id}")
    with c2:
        eis_csv = pd.DataFrame({"freq (Hz)": freq, "Z_real (Ω)": z_real, "col3 (as loaded)": z_imag_in}).to_csv(index=False)
        st.download_button("📥 EIS.csv", eis_csv, "EIS.csv", "text/csv", key=f"download_eis_{file_id}")

    st.success("✅ 완료!")

# -------------------------------------------------------------------
# Run analysis only when user clicks the button
# -------------------------------------------------------------------
if st.button("🚀 분석 시작", type="primary"):
    for _uf in uploaded_files:
        analyze_one(_uf)
else:
    st.info("파일을 업로드한 뒤, **분석 시작**을 눌러주세요.")
