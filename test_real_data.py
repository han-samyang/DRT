"""
Test DRT Analysis with Real Battery Data
========================================
Quick test script using the real battery EIS data you provided
(251001_ref_1_C01.txt)

Run: python test_real_data.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from drt_core import DRTAnalyzer

# Load real battery data
print("📊 Loading real battery EIS data...")
print("=" * 70)

data_file = "251001_ref_1_C01.txt"

try:
    # Read the tab-separated txt file
    df = pd.read_csv(data_file, sep='\t', skiprows=0)
    
    print(f"✅ Loaded data with {len(df)} frequency points")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    # Auto-detect columns
    col_names = [c.lower().strip() for c in df.columns]
    print(f"\nColumn names (lowercased): {col_names}")
    
    # Extract data
    freq = df.iloc[:, 0].values  # First column: frequency
    z_real = df.iloc[:, 1].values  # Second column: Re(Z)
    z_imag = np.abs(df.iloc[:, 2].values)  # Third column: -Im(Z) -> |Im(Z)|
    
    print(f"\n📈 Data ranges:")
    print(f"  Frequency: {freq.min():.3e} to {freq.max():.3e} Hz")
    print(f"  Z_real: {z_real.min():.2f} to {z_real.max():.2f} Ω")
    print(f"  Z_imag: {z_imag.min():.2f} to {z_imag.max():.2f} Ω")
    
except FileNotFoundError:
    print(f"❌ Error: Could not find {data_file}")
    print("Make sure the file is in the same directory as this script")
    exit(1)

# Initialize and run DRT analyzer
print("\n" + "=" * 70)
print("🔧 Running DRT Analysis with Tikhonov Regularization...")
print("=" * 70)

analyzer = DRTAnalyzer()
analyzer.load_data(freq, z_real, z_imag)

# Solve with automatic λ selection (GCV)
success = analyzer.solve_drt(
    n_tau=100,
    lambda_auto=True,
    non_negative=False,
    verbose=True
)

if success:
    print("\n✅ Analysis completed successfully!")
    
    # Get summary
    summary = analyzer.get_summary()
    
    print("\n" + "=" * 70)
    print("📊 ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"Number of peaks detected: {summary['n_peaks']}")
    print(f"Optimal regularization λ: {summary['lambda_opt']:.2e}")
    print(f"γ(τ) range: [{summary['gamma_min']:.2e}, {summary['gamma_max']:.2e}] Ω")
    print(f"Total resistance: {summary['total_resistance']:.2f} Ω")
    
    print("\n📈 FIT QUALITY:")
    print(f"RMSE: {summary['residual_stats']['rmse']:.2e} Ω")
    print(f"Relative error: {summary['residual_stats']['relative_error']:.4f} (4% = 0.04)")
    print(f"Max error: {summary['residual_stats']['max_error']:.2e} Ω")
    
    # Peak information
    if analyzer.peaks_info:
        print("\n🎯 DETECTED PEAKS:")
        print(f"{'Peak':<6} {'τ (s)':<15} {'γ (Ω)':<15} {'R (Ω)':<12} {'f (Hz)':<12}")
        print("-" * 65)
        for i, peak in enumerate(analyzer.peaks_info, 1):
            f_peak = 1 / (2 * np.pi * peak['tau'])
            print(f"{i:<6} {peak['tau']:<15.3e} {peak['gamma']:<15.3e} "
                  f"{peak['resistance']:<12.3f} {f_peak:<12.3e}")
    
    # Create visualization
    print("\n" + "=" * 70)
    print("📊 Creating plots...")
    print("=" * 70)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('DRT Analysis - Real Battery EIS Data (251001_ref_1_C01.txt)', 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: DRT Distribution
    ax = axes[0, 0]
    gamma_differential = analyzer.gamma / np.log(10)  # γ'(ln τ) = γ(τ)/ln(10)
    ax.semilogx(analyzer.tau_grid, gamma_differential, 'b-', linewidth=2.5, label="γ'(ln τ)")
    ax.fill_between(analyzer.tau_grid, 0, gamma_differential, alpha=0.3, color='blue')
    if analyzer.peaks_info:
        peaks_tau = [p['tau'] for p in analyzer.peaks_info]
        peaks_gamma = [p['gamma'] / np.log(10) for p in analyzer.peaks_info]
        ax.plot(peaks_tau, peaks_gamma, 'r*', markersize=15, label='Peaks')
    ax.set_xlabel('Time Constant τ (s)', fontsize=11)
    ax.set_ylabel("γ'(ln τ) (Ω)", fontsize=11)
    ax.set_title('Distribution of Relaxation Times (DRT)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # Plot 2: Nyquist
    ax = axes[0, 1]
    ax.plot(z_real, z_imag, 'bo', markersize=5, label='Experimental EIS', alpha=0.7)
    ax.plot(analyzer.Z_reconst_real, analyzer.Z_reconst_imag, 'r-', 
            linewidth=2.5, label='DRT Reconstructed')
    ax.set_xlabel("Z' (Ω)", fontsize=11)
    ax.set_ylabel("-Z'' (Ω)", fontsize=11)
    ax.set_title('Nyquist Plot: EIS vs DRT Reconstruction', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_aspect('equal', adjustable='box')
    
    # Plot 3: Bode - Magnitude
    ax = axes[1, 0]
    mag = np.sqrt(z_real**2 + z_imag**2)
    mag_reconst = np.sqrt(analyzer.Z_reconst_real**2 + analyzer.Z_reconst_imag**2)
    ax.loglog(freq, mag, 'bo', markersize=5, label='Experimental', alpha=0.7)
    ax.loglog(freq, mag_reconst, 'r-', linewidth=2.5, label='DRT Fit')
    ax.set_xlabel('Frequency (Hz)', fontsize=11)
    ax.set_ylabel('|Z| (Ω)', fontsize=11)
    ax.set_title('Bode Plot: Impedance Magnitude', fontsize=12, fontweight='bold')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=10)
    
    # Plot 4: Residuals
    ax = axes[1, 1]
    n_freq = len(freq)
    residual_real = analyzer.residual['abs'][:n_freq]
    residual_imag = analyzer.residual['abs'][n_freq:]
    ax.semilogx(freq, residual_real, 'bo-', markersize=4, label="Residual Z'", alpha=0.7)
    ax.semilogx(freq, residual_imag, 'rs-', markersize=4, label='Residual Z"', alpha=0.7)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Frequency (Hz)', fontsize=11)
    ax.set_ylabel('Residual (Ω)', fontsize=11)
    ax.set_title('Fit Residuals', fontsize=12, fontweight='bold')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig('drt_analysis_result.png', dpi=150, bbox_inches='tight')
    print("✅ Plot saved as 'drt_analysis_result.png'")
    
    # Optional: Show plot
    plt.show()
    
else:
    print("❌ Analysis failed!")

print("\n" + "=" * 70)
print("✨ Test completed!")
print("=" * 70)
