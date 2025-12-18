"""
EIS to DRT Conversion Tool
Converts Electrochemical Impedance Spectroscopy (EIS) data to 
Distribution of Relaxation Times (DRT) using Tikhonov regularization.
"""

import numpy as np
from scipy.linalg import solve
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt


class EIStoDRT:
    """
    Convert EIS impedance data to DRT (Distribution of Relaxation Times).
    
    Uses Tikhonov regularization to solve the inverse problem of extracting
    the distribution of relaxation times from frequency-domain impedance data.
    """
    
    # Constants
    DEFAULT_N_TAU = 100  # Default number of tau (relaxation time) points
    CURVATURE_EPSILON = 1e-10  # Small value to prevent division by zero in curvature calculation
    
    def __init__(self, frequencies, impedances, n_tau=None):
        """
        Initialize the EIS to DRT converter.
        
        Parameters:
        -----------
        frequencies : array-like
            Frequencies in Hz where impedance was measured
        impedances : array-like (complex)
            Complex impedance values (Z = Z' + jZ'')
        n_tau : int, optional
            Number of relaxation time points for DRT discretization (default: 100)
        """
        self.frequencies = np.array(frequencies)
        self.impedances = np.array(impedances, dtype=complex)
        self.omega = 2 * np.pi * self.frequencies
        
        # Initialize tau grid (relaxation times)
        tau_min = 1 / (2 * np.pi * np.max(self.frequencies))
        tau_max = 1 / (2 * np.pi * np.min(self.frequencies))
        n_tau_points = n_tau if n_tau is not None else self.DEFAULT_N_TAU
        self.tau = np.logspace(np.log10(tau_min), np.log10(tau_max), n_tau_points)
        
        self.gamma = None  # DRT distribution
        self.R_inf = None  # High-frequency resistance
        self.lambda_reg = None  # Regularization parameter
        
    def build_A_matrix(self):
        """
        Build the discretized kernel matrix A for the DRT integral equation.
        
        The DRT is related to impedance by:
        Z(ω) = R_∞ + ∫ γ(τ)/(1 + jωτ) dτ
        """
        N_freq = len(self.omega)
        N_tau = len(self.tau)
        
        A_re = np.zeros((N_freq, N_tau))
        A_im = np.zeros((N_freq, N_tau))
        
        for i, omega_i in enumerate(self.omega):
            for j, tau_j in enumerate(self.tau):
                denom = 1 + (omega_i * tau_j)**2
                A_re[i, j] = 1.0 / denom
                A_im[i, j] = -omega_i * tau_j / denom
        
        # Scale by log-spacing of tau (trapezoidal rule in log-space)
        delta_ln_tau = np.diff(np.log(self.tau))
        delta_ln_tau = np.append(delta_ln_tau, delta_ln_tau[-1])
        
        for j in range(N_tau):
            A_re[:, j] *= delta_ln_tau[j]
            A_im[:, j] *= delta_ln_tau[j]
        
        return A_re, A_im
    
    def build_L_matrix(self, derivative_order=1):
        """
        Build the regularization matrix L for Tikhonov regularization.
        
        Parameters:
        -----------
        derivative_order : int
            Order of derivative to regularize (1 or 2)
        """
        N = len(self.tau)
        
        if derivative_order == 1:
            # First derivative (penalize roughness)
            L = np.zeros((N-1, N))
            delta_ln_tau = np.diff(np.log(self.tau))
            for i in range(N-1):
                L[i, i] = -1.0 / delta_ln_tau[i]
                L[i, i+1] = 1.0 / delta_ln_tau[i]
        elif derivative_order == 2:
            # Second derivative (penalize curvature)
            L = np.zeros((N-2, N))
            delta_ln_tau = np.diff(np.log(self.tau))
            for i in range(N-2):
                h1 = delta_ln_tau[i]
                h2 = delta_ln_tau[i+1]
                L[i, i] = 2.0 / (h1 * (h1 + h2))
                L[i, i+1] = -2.0 / (h1 * h2)
                L[i, i+2] = 2.0 / (h2 * (h1 + h2))
        else:
            raise ValueError("derivative_order must be 1 or 2")
        
        return L
    
    def estimate_R_inf(self):
        """
        Estimate the high-frequency resistance R_∞.
        Uses the real part of impedance at highest frequency.
        """
        idx_max_freq = np.argmax(self.frequencies)
        self.R_inf = np.real(self.impedances[idx_max_freq])
        return self.R_inf
    
    def compute_drt(self, lambda_reg=1e-3, derivative_order=1):
        """
        Compute the DRT using Tikhonov regularization.
        
        Parameters:
        -----------
        lambda_reg : float
            Regularization parameter (controls smoothness)
        derivative_order : int
            Order of derivative for regularization (1 or 2)
        
        Returns:
        --------
        tau : array
            Relaxation times
        gamma : array
            DRT distribution values
        """
        # Estimate R_inf if not already done
        if self.R_inf is None:
            self.estimate_R_inf()
        
        # Build system matrices
        A_re, A_im = self.build_A_matrix()
        L = self.build_L_matrix(derivative_order)
        
        # Prepare data vector (subtract R_inf from real part)
        Z_re = np.real(self.impedances) - self.R_inf
        Z_im = np.imag(self.impedances)
        
        # Stack real and imaginary parts
        A = np.vstack([A_re, A_im])
        b = np.hstack([Z_re, Z_im])
        
        # Tikhonov regularization: minimize ||Ax - b||^2 + λ||Lx||^2
        # Solution: (A^T A + λ L^T L) x = A^T b
        ATA = A.T @ A
        ATb = A.T @ b
        LTL = L.T @ L
        
        # Solve regularized system
        self.gamma = solve(ATA + lambda_reg * LTL, ATb)
        self.lambda_reg = lambda_reg
        
        # Ensure non-negative DRT
        self.gamma = np.maximum(self.gamma, 0)
        
        return self.tau, self.gamma
    
    def find_optimal_lambda(self, lambda_range=(1e-6, 1e-1), method='L-curve', n_samples=20):
        """
        Find optimal regularization parameter using L-curve method or cross-validation.
        
        Parameters:
        -----------
        lambda_range : tuple
            Range of lambda values to search
        method : str
            Method to use ('L-curve' or 'gcv' for generalized cross-validation)
        n_samples : int
            Number of lambda values to test (default: 20)
        
        Returns:
        --------
        lambda_opt : float
            Optimal regularization parameter
        """
        if method == 'L-curve':
            return self._find_lambda_lcurve(lambda_range, n_samples)
        elif method == 'gcv':
            return self._find_lambda_gcv(lambda_range)
        else:
            raise ValueError("method must be 'L-curve' or 'gcv'")
    
    def _find_lambda_lcurve(self, lambda_range, n_samples):
        """Find optimal lambda using L-curve criterion."""
        lambdas = np.logspace(np.log10(lambda_range[0]), np.log10(lambda_range[1]), n_samples)
        
        residuals = []
        smoothness = []
        
        A_re, A_im = self.build_A_matrix()
        L = self.build_L_matrix()
        
        Z_re = np.real(self.impedances) - self.R_inf
        Z_im = np.imag(self.impedances)
        A = np.vstack([A_re, A_im])
        b = np.hstack([Z_re, Z_im])
        
        for lam in lambdas:
            gamma = solve(A.T @ A + lam * L.T @ L, A.T @ b)
            gamma = np.maximum(gamma, 0)
            
            residual = np.linalg.norm(A @ gamma - b)
            smooth = np.linalg.norm(L @ gamma)
            
            residuals.append(residual)
            smoothness.append(smooth)
        
        # Find corner of L-curve (maximum curvature)
        residuals = np.array(residuals)
        smoothness = np.array(smoothness)
        
        # Normalize for curvature calculation
        log_res = np.log(residuals)
        log_smooth = np.log(smoothness)
        
        # Compute curvature
        curvatures = []
        for i in range(1, len(lambdas)-1):
            dx1 = log_res[i] - log_res[i-1]
            dy1 = log_smooth[i] - log_smooth[i-1]
            dx2 = log_res[i+1] - log_res[i]
            dy2 = log_smooth[i+1] - log_smooth[i]
            
            # Approximate curvature
            curv = abs(dx1*dy2 - dx2*dy1) / ((dx1**2 + dy1**2)**1.5 + self.CURVATURE_EPSILON)
            curvatures.append(curv)
        
        # Find lambda with maximum curvature
        idx_opt = np.argmax(curvatures) + 1
        return lambdas[idx_opt]
    
    def _find_lambda_gcv(self, lambda_range):
        """Find optimal lambda using Generalized Cross-Validation."""
        # Simplified GCV implementation
        def gcv_score(log_lambda):
            lam = 10**log_lambda
            _, gamma = self.compute_drt(lam)
            
            A_re, A_im = self.build_A_matrix()
            A = np.vstack([A_re, A_im])
            Z_re = np.real(self.impedances) - self.R_inf
            Z_im = np.imag(self.impedances)
            b = np.hstack([Z_re, Z_im])
            
            residual = np.linalg.norm(A @ gamma - b)
            n = len(b)
            p = len(gamma)
            
            # GCV score with protection against division by zero
            if n >= p:
                # When n >= p, use large penalty
                score = residual**2 * 1e10
            else:
                score = residual**2 / (n * (1 - n/p)**2)
            return score
        
        result = minimize_scalar(gcv_score, 
                                bounds=(np.log10(lambda_range[0]), 
                                       np.log10(lambda_range[1])),
                                method='bounded')
        
        return 10**result.x
    
    def plot_eis(self, ax=None):
        """
        Plot Nyquist plot of EIS data.
        
        Parameters:
        -----------
        ax : matplotlib axis, optional
            Axis to plot on
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        Z_re = np.real(self.impedances)
        Z_im = -np.imag(self.impedances)  # Convention: -Z'' on y-axis
        
        ax.plot(Z_re, Z_im, 'o-', label='EIS Data')
        ax.set_xlabel("Z' (Ω)")
        ax.set_ylabel("-Z'' (Ω)")
        ax.set_title('Nyquist Plot')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.axis('equal')
        
        return ax
    
    def plot_drt(self, ax=None):
        """
        Plot the computed DRT.
        
        Parameters:
        -----------
        ax : matplotlib axis, optional
            Axis to plot on
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        if self.gamma is None:
            raise ValueError("DRT not computed yet. Run compute_drt() first.")
        
        ax.plot(self.tau, self.gamma, 'b-', linewidth=2)
        ax.set_xlabel('τ (s)')
        ax.set_ylabel('γ(τ) (Ω)')
        ax.set_title('Distribution of Relaxation Times')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_summary(self, save_path=None):
        """
        Create a summary plot with EIS data and DRT.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        self.plot_eis(ax1)
        self.plot_drt(ax2)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


def generate_synthetic_eis_data(n_points=50, noise_level=0.5):
    """
    Generate synthetic EIS data for testing.
    
    Simulates impedance with multiple relaxation processes.
    
    Parameters:
    -----------
    n_points : int
        Number of frequency points (default: 50)
    noise_level : float
        Standard deviation of additive Gaussian noise (default: 0.5)
    
    Returns:
    --------
    frequencies : array
        Frequencies in Hz
    impedances : array (complex)
        Complex impedance values
    """
    # Frequency range
    frequencies = np.logspace(-2, 6, n_points)  # 0.01 Hz to 1 MHz
    omega = 2 * np.pi * frequencies
    
    # Simulate impedance with multiple RC elements
    R_inf = 10.0  # High-frequency resistance
    
    # Multiple relaxation processes
    R_values = [20.0, 30.0, 15.0]  # Resistance values
    tau_values = [1e-3, 1e-1, 1e1]  # Relaxation times
    
    impedances = np.ones(len(frequencies), dtype=complex) * R_inf
    
    for R, tau in zip(R_values, tau_values):
        impedances += R / (1 + 1j * omega * tau)
    
    # Add noise
    impedances += (np.random.randn(len(frequencies)) + 
                   1j * np.random.randn(len(frequencies))) * noise_level
    
    return frequencies, impedances


if __name__ == "__main__":
    # Example usage
    print("EIS to DRT Conversion Tool")
    print("=" * 50)
    
    # Generate synthetic data
    print("\n1. Generating synthetic EIS data...")
    frequencies, impedances = generate_synthetic_eis_data()
    print(f"   Generated {len(frequencies)} frequency points")
    
    # Create converter
    print("\n2. Initializing EIS to DRT converter...")
    converter = EIStoDRT(frequencies, impedances)
    
    # Estimate R_inf
    converter.estimate_R_inf()
    print(f"   Estimated R_∞ = {converter.R_inf:.2f} Ω")
    
    # Find optimal regularization parameter
    print("\n3. Finding optimal regularization parameter...")
    lambda_opt = converter.find_optimal_lambda(method='L-curve')
    print(f"   Optimal λ = {lambda_opt:.2e}")
    
    # Compute DRT
    print("\n4. Computing DRT...")
    tau, gamma = converter.compute_drt(lambda_reg=lambda_opt)
    print(f"   DRT computed with {len(tau)} relaxation times")
    
    # Plot results
    print("\n5. Generating plots...")
    converter.plot_summary(save_path='eis_drt_summary.png')
    print("   Plots saved to 'eis_drt_summary.png'")
    
    print("\n" + "=" * 50)
    print("Conversion complete!")
