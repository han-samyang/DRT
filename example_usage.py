"""
Example usage of the EIS to DRT conversion tool.
This script demonstrates how to use the tool with both synthetic and real data.
"""

import numpy as np
import matplotlib.pyplot as plt
from eis_to_drt import EIStoDRT, generate_synthetic_eis_data


def example_synthetic_data():
    """Example 1: Using synthetic EIS data."""
    print("=" * 60)
    print("Example 1: EIS to DRT Conversion with Synthetic Data")
    print("=" * 60)
    
    # Generate synthetic data
    print("\nStep 1: Generating synthetic EIS data...")
    frequencies, impedances = generate_synthetic_eis_data(n_points=60)
    print(f"  Generated {len(frequencies)} frequency points")
    print(f"  Frequency range: {frequencies[0]:.2e} Hz to {frequencies[-1]:.2e} Hz")
    
    # Create converter
    print("\nStep 2: Creating EIS to DRT converter...")
    converter = EIStoDRT(frequencies, impedances)
    
    # Estimate high-frequency resistance
    print("\nStep 3: Estimating high-frequency resistance...")
    R_inf = converter.estimate_R_inf()
    print(f"  R_∞ = {R_inf:.2f} Ω")
    
    # Find optimal regularization parameter
    print("\nStep 4: Finding optimal regularization parameter...")
    print("  Using L-curve method...")
    lambda_opt = converter.find_optimal_lambda(lambda_range=(1e-6, 1e-1), method='L-curve')
    print(f"  Optimal λ = {lambda_opt:.3e}")
    
    # Compute DRT
    print("\nStep 5: Computing DRT...")
    tau, gamma = converter.compute_drt(lambda_reg=lambda_opt)
    print(f"  DRT computed over {len(tau)} relaxation times")
    print(f"  τ range: {tau[0]:.2e} s to {tau[-1]:.2e} s")
    
    # Find peaks in DRT
    peak_indices = []
    for i in range(1, len(gamma)-1):
        if gamma[i] > gamma[i-1] and gamma[i] > gamma[i+1] and gamma[i] > 0.5 * np.max(gamma):
            peak_indices.append(i)
    
    if peak_indices:
        print(f"\n  Found {len(peak_indices)} significant peaks:")
        for idx in peak_indices:
            print(f"    τ = {tau[idx]:.3e} s, γ = {gamma[idx]:.2f} Ω")
    
    # Plot results
    print("\nStep 6: Creating visualization...")
    fig = converter.plot_summary(save_path='example_synthetic_drt.png')
    print("  Saved plot to 'example_synthetic_drt.png'")
    
    print("\n" + "=" * 60)
    print("Example 1 Complete!")
    print("=" * 60 + "\n")
    
    return converter


def example_comparison():
    """Example 2: Compare different regularization parameters."""
    print("=" * 60)
    print("Example 2: Effect of Regularization Parameter")
    print("=" * 60)
    
    # Generate data
    frequencies, impedances = generate_synthetic_eis_data(n_points=50)
    converter = EIStoDRT(frequencies, impedances)
    converter.estimate_R_inf()
    
    # Try different lambda values
    lambdas = [1e-5, 1e-4, 1e-3, 1e-2]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, lam in enumerate(lambdas):
        tau, gamma = converter.compute_drt(lambda_reg=lam)
        
        axes[i].plot(tau, gamma, 'b-', linewidth=2)
        axes[i].set_xlabel('τ (s)')
        axes[i].set_ylabel('γ(τ) (Ω)')
        axes[i].set_title(f'DRT with λ = {lam:.0e}')
        axes[i].set_xscale('log')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('example_lambda_comparison.png', dpi=300, bbox_inches='tight')
    print("\nSaved comparison plot to 'example_lambda_comparison.png'")
    
    print("\nObservations:")
    print("  - Small λ: More peaks, sensitive to noise")
    print("  - Large λ: Smoother distribution, may miss details")
    print("  - Optimal λ: Balance between fit and smoothness")
    
    print("\n" + "=" * 60)
    print("Example 2 Complete!")
    print("=" * 60 + "\n")


def example_save_load_data():
    """Example 3: Save and load EIS data."""
    print("=" * 60)
    print("Example 3: Saving and Loading EIS Data")
    print("=" * 60)
    
    # Generate and save data
    print("\nGenerating and saving EIS data...")
    frequencies, impedances = generate_synthetic_eis_data()
    
    # Save to CSV format
    data = np.column_stack([
        frequencies,
        np.real(impedances),
        np.imag(impedances)
    ])
    
    header = "Frequency(Hz),Z_real(Ohm),Z_imag(Ohm)"
    np.savetxt('example_eis_data.csv', data, delimiter=',', 
               header=header, comments='', fmt='%.6e')
    print("  Saved to 'example_eis_data.csv'")
    
    # Load data
    print("\nLoading EIS data from file...")
    loaded_data = np.loadtxt('example_eis_data.csv', delimiter=',', skiprows=1)
    loaded_freq = loaded_data[:, 0]
    loaded_Z = loaded_data[:, 1] + 1j * loaded_data[:, 2]
    print(f"  Loaded {len(loaded_freq)} data points")
    
    # Process loaded data
    print("\nProcessing loaded data...")
    converter = EIStoDRT(loaded_freq, loaded_Z)
    converter.estimate_R_inf()
    lambda_opt = converter.find_optimal_lambda()
    tau, gamma = converter.compute_drt(lambda_reg=lambda_opt)
    
    converter.plot_summary(save_path='example_loaded_data_drt.png')
    print("  Results saved to 'example_loaded_data_drt.png'")
    
    print("\n" + "=" * 60)
    print("Example 3 Complete!")
    print("=" * 60 + "\n")


def print_drt_statistics(converter):
    """Print statistics about the computed DRT."""
    if converter.gamma is None:
        print("No DRT computed yet.")
        return
    
    print("\nDRT Statistics:")
    print("-" * 40)
    print(f"  Total polarization resistance: {np.sum(converter.gamma) * np.mean(np.diff(np.log(converter.tau))):.2f} Ω")
    print(f"  Maximum γ value: {np.max(converter.gamma):.2f} Ω")
    print(f"  τ at maximum γ: {converter.tau[np.argmax(converter.gamma)]:.3e} s")
    print(f"  Number of τ points: {len(converter.tau)}")
    print("-" * 40)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("EIS to DRT Conversion Tool - Examples")
    print("=" * 60 + "\n")
    
    # Run examples
    converter = example_synthetic_data()
    print_drt_statistics(converter)
    
    print("\n")
    example_comparison()
    
    print("\n")
    example_save_load_data()
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - example_synthetic_drt.png")
    print("  - example_lambda_comparison.png")
    print("  - example_eis_data.csv")
    print("  - example_loaded_data_drt.png")
    print("\nCheck these files to see the results!")
