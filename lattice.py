"""D2Q9 Lattice for 2D Lattice Boltzmann Method.

This module defines the D2Q9 lattice structure with 9 velocity directions,
providing equilibrium distributions and macroscopic quantity computations.
"""

import numpy as np
from typing import Tuple


class D2Q9Lattice:
    """D2Q9 lattice for 2D Lattice Boltzmann Method.
    
    The D2Q9 lattice has 9 velocity directions:
    - 1 rest particle (0,0)
    - 4 cardinal directions (±1,0) and (0,±1)
    - 4 diagonal directions (±1,±1)
    
    Attributes:
        c: Lattice velocities (9, 2) - [x, y] components
        w: Lattice weights (9,) - normalized to sum to 1
        opp: Opposite directions (9,) - indices for bounce-back
        cs: Speed of sound (1/√3)
        cs2: Speed of sound squared (1/3)
    """
    
    def __init__(self):
        """Initialize D2Q9 lattice structure."""
        # Lattice velocities (9 directions)
        # Index layout:
        #   6   2   5
        #     ↖ ↑ ↗
        #   3 ← 0 → 1
        #     ↙ ↓ ↘
        #   7   4   8
        self.c = np.array([
            [0,  0],   # 0: rest
            [1,  0],   # 1: east
            [0,  1],   # 2: north
            [-1, 0],   # 3: west
            [0, -1],   # 4: south
            [1,  1],   # 5: northeast
            [-1, 1],   # 6: northwest
            [-1,-1],   # 7: southwest
            [1, -1]    # 8: southeast
        ], dtype=float)
        
        # Lattice weights (normalized)
        # w[0] = 4/9 for rest particle
        # w[1-4] = 1/9 for cardinal directions
        # w[5-8] = 1/36 for diagonal directions
        self.w = np.array([
            4/9,   # 0: rest
            1/9,   # 1-4: cardinal
            1/9,
            1/9,
            1/9,
            1/36,  # 5-8: diagonal
            1/36,
            1/36,
            1/36
        ])
        
        # Opposite directions (for bounce-back boundary condition)
        # opp[i] gives the index of the direction opposite to i
        self.opp = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])
        
        # Speed of sound in lattice units
        self.cs = 1.0 / np.sqrt(3.0)
        self.cs2 = self.cs**2
        
        # Verify lattice properties
        self._verify_lattice_properties()
    
    def _verify_lattice_properties(self):
        """Verify lattice satisfies required properties."""
        # Check weights sum to 1
        weight_sum = np.sum(self.w)
        assert abs(weight_sum - 1.0) < 1e-12, f"Weights sum to {weight_sum}, not 1.0"
        
        # Check opposite directions
        for i in range(9):
            opp_i = self.opp[i]
            assert np.allclose(self.c[i], -self.c[opp_i]), \
                f"Direction {i} and opposite {opp_i} don't cancel"
        
        print("✓ D2Q9 lattice properties verified")
    
    def equilibrium(self, rho: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Compute equilibrium distribution function.
        
        The equilibrium distribution is given by:
        f_i^eq = w_i * ρ * [1 + (c_i·u)/cs² + (c_i·u)²/(2cs⁴) - u²/(2cs²)]
        
        Args:
            rho: Density field (Nx, Ny)
            u: x-velocity component (Nx, Ny)
            v: y-velocity component (Nx, Ny)
            
        Returns:
            f_eq: Equilibrium distributions (Nx, Ny, 9)
            
        Example:
            >>> lattice = D2Q9Lattice()
            >>> rho = np.ones((10, 10))
            >>> u = 0.1 * np.ones((10, 10))
            >>> v = np.zeros((10, 10))
            >>> f_eq = lattice.equilibrium(rho, u, v)
            >>> f_eq.shape
            (10, 10, 9)
        """
        Nx, Ny = rho.shape
        f_eq = np.zeros((Nx, Ny, 9))
        
        # Velocity squared
        u2 = u**2 + v**2
        
        # Compute equilibrium for each direction
        for i in range(9):
            # Lattice velocity dotted with macroscopic velocity
            cu = self.c[i, 0] * u + self.c[i, 1] * v
            
            # Equilibrium distribution
            f_eq[:, :, i] = self.w[i] * rho * (
                1.0
                + cu / self.cs2
                + cu**2 / (2 * self.cs2**2)
                - u2 / (2 * self.cs2)
            )
        
        return f_eq
    
    def compute_macroscopic(self, f: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute macroscopic density and velocity from distribution functions.
        
        The macroscopic quantities are computed as moments of the distribution:
        ρ = Σ f_i
        ρ u = Σ f_i * c_i
        
        Args:
            f: Distribution functions (Nx, Ny, 9)
            
        Returns:
            rho: Density (Nx, Ny)
            u: x-velocity component (Nx, Ny)
            v: y-velocity component (Nx, Ny)
            
        Example:
            >>> lattice = D2Q9Lattice()
            >>> f = np.random.rand(10, 10, 9)
            >>> rho, u, v = lattice.compute_macroscopic(f)
            >>> rho.shape, u.shape, v.shape
            ((10, 10), (10, 10), (10, 10))
        """
        # Density: sum of all distributions
        rho = np.sum(f, axis=2)
        
        # Avoid division by zero
        rho_safe = np.where(rho > 1e-12, rho, 1.0)
        
        # Momentum: sum of distributions weighted by lattice velocities
        u = np.sum(f * self.c[:, 0], axis=2) / rho_safe
        v = np.sum(f * self.c[:, 1], axis=2) / rho_safe
        
        return rho, u, v
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"D2Q9Lattice(directions=9, cs={self.cs:.6f}, "
                f"weights_sum={np.sum(self.w):.6f})")


# Utility functions for advanced usage

def compute_vorticity(u: np.ndarray, v: np.ndarray, dx: float = 1.0) -> np.ndarray:
    """Compute vorticity field: ω = ∂v/∂x - ∂u/∂y.
    
    Args:
        u: x-velocity component (Nx, Ny)
        v: y-velocity component (Nx, Ny)
        dx: Grid spacing (default: 1.0)
        
    Returns:
        omega: Vorticity field (Nx, Ny)
    """
    dv_dx = np.gradient(v, dx, axis=0)
    du_dy = np.gradient(u, dx, axis=1)
    return dv_dx - du_dy


def compute_stream_function(u: np.ndarray, v: np.ndarray, dx: float = 1.0) -> np.ndarray:
    """Compute stream function ψ from velocity field.
    
    The stream function satisfies:
    u = ∂ψ/∂y, v = -∂ψ/∂x
    
    Args:
        u: x-velocity component (Nx, Ny)
        v: y-velocity component (Nx, Ny)
        dx: Grid spacing (default: 1.0)
        
    Returns:
        psi: Stream function (Nx, Ny)
        
    Note:
        This uses a simple cumulative integration. For better accuracy,
        solve the Poisson equation: ∇²ψ = ω
    """
    # Simple integration approach (less accurate but fast)
    psi = np.cumsum(u, axis=1) * dx
    return psi
