"""LBM Core Engine for collision and streaming operations.

This module implements the core LBM algorithm with Numba JIT optimization.
"""

import numpy as np
from numba import jit, prange
from lattice import D2Q9Lattice
from typing import Optional


class LBMCore:
    """Core LBM collision and streaming operations."""
    
    def __init__(self, config_manager, lattice: D2Q9Lattice, geometry_builder):
        """Initialize LBM core engine.
        
        Args:
            config_manager: ConfigManager instance
            lattice: D2Q9Lattice instance
            geometry_builder: GeometryBuilder instance
        """
        self.config = config_manager
        self.lattice = lattice
        self.geom = geometry_builder
        
        # Grid dimensions
        self.Nx = config_manager.get('grid.Nx')
        self.Ny = config_manager.get('grid.Ny')
        
        # Physics parameters
        self.tau = config_manager.get('solver.tau')
        self.omega = 1.0 / self.tau  # Relaxation frequency
        
        # Distribution functions
        self.f = np.zeros((self.Nx, self.Ny, 9), dtype=np.float64)
        self.f_post_collision = np.zeros_like(self.f)
        
        # Macroscopic fields
        self.rho = np.ones((self.Nx, self.Ny), dtype=np.float64)
        self.u = np.zeros((self.Nx, self.Ny), dtype=np.float64)
        self.v = np.zeros((self.Nx, self.Ny), dtype=np.float64)
        
        # Cache lattice properties for JIT functions
        self.c = lattice.c.copy()
        self.w = lattice.w.copy()
        self.opp = lattice.opp.copy()
        self.cs2 = lattice.cs2
        
        # Solid mask
        self.is_solid = geometry_builder.is_solid.copy()
        
        print(f"✓ LBM Core initialized:")
        print(f"  - Grid: {self.Nx} × {self.Ny}")
        print(f"  - Relaxation time (tau): {self.tau:.6f}")
        print(f"  - Relaxation frequency (omega): {self.omega:.6f}")
    
    def initialize(self, u_init: Optional[float] = None):
        """Initialize distribution functions to equilibrium.
        
        Args:
            u_init: Initial x-velocity (default: from config)
        """
        # Set initial velocity
        if u_init is None:
            u_init = self.config.get('physics.characteristic_velocity', 0.0)
        
        self.u[:] = u_init
        self.v[:] = 0.0
        self.rho[:] = 1.0
        
        # Compute equilibrium
        self.f = self.lattice.equilibrium(self.rho, self.u, self.v)
        
        # Zero out solid nodes
        for i in range(9):
            self.f[self.is_solid, i] = 0.0
        
        print(f"✓ Distributions initialized to equilibrium")
        print(f"  - Initial velocity: u={u_init:.4f}, v=0.0")
        print(f"  - Initial density: rho=1.0")
    
    def collision(self):
        """BGK collision step (Numba-optimized)."""
        self.f_post_collision = collision_bgk_numba(
            self.f, self.rho, self.u, self.v,
            self.omega, self.w, self.c, self.cs2,
            self.is_solid
        )
    
    def streaming(self):
        """Streaming step with bounce-back (Numba-optimized)."""
        self.f = streaming_numba(
            self.f_post_collision, self.c, self.is_solid, self.opp,
            self.Nx, self.Ny
        )
    
    def compute_macroscopic(self):
        """Update macroscopic fields from distributions."""
        self.rho, self.u, self.v = self.lattice.compute_macroscopic(self.f)
        
        # Zero out velocities in solid nodes
        self.u[self.is_solid] = 0.0
        self.v[self.is_solid] = 0.0
    
    def step(self):
        """Execute one full LBM time step.
        
        Sequence:
        1. Collision
        2. Streaming (includes bounce-back)
        3. Compute macroscopic quantities
        """
        self.collision()
        self.streaming()
        self.compute_macroscopic()
    
    def get_vorticity(self, dx: float = 1.0) -> np.ndarray:
        """Compute vorticity field.
        
        Args:
            dx: Grid spacing
            
        Returns:
            omega: Vorticity field (Nx, Ny)
        """
        dv_dx = np.gradient(self.v, dx, axis=0)
        du_dy = np.gradient(self.u, dx, axis=1)
        return dv_dx - du_dy
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"LBMCore(grid={self.Nx}×{self.Ny}, tau={self.tau:.4f}, "
                f"omega={self.omega:.4f})")


# Numba-compiled functions for performance
@jit(nopython=True, parallel=True)
def collision_bgk_numba(f, rho, u, v, omega, w, c, cs2, is_solid):
    """BGK collision operator (JIT-compiled).
    
    Args:
        f: Distribution functions (Nx, Ny, 9)
        rho: Density (Nx, Ny)
        u, v: Velocity components (Nx, Ny)
        omega: Relaxation frequency (1/tau)
        w: Lattice weights (9,)
        c: Lattice velocities (9, 2)
        cs2: Speed of sound squared
        is_solid: Solid mask (Nx, Ny)
    
    Returns:
        f_post: Post-collision distributions (Nx, Ny, 9)
    """
    Nx, Ny, _ = f.shape
    f_post = np.empty_like(f)
    
    for x in prange(Nx):
        for y in range(Ny):
            # Skip solid nodes
            if is_solid[x, y]:
                for i in range(9):
                    f_post[x, y, i] = f[x, y, i]
                continue
            
            # Velocity squared
            u2 = u[x, y]**2 + v[x, y]**2
            
            # Collision for each direction
            for i in range(9):
                # Lattice velocity · macroscopic velocity
                cu = c[i, 0] * u[x, y] + c[i, 1] * v[x, y]
                
                # Equilibrium distribution
                f_eq = w[i] * rho[x, y] * (
                    1.0 
                    + cu / cs2 
                    + cu**2 / (2.0 * cs2**2)
                    - u2 / (2.0 * cs2)
                )
                
                # BGK collision: f_post = f - ω(f - f_eq)
                f_post[x, y, i] = f[x, y, i] - omega * (f[x, y, i] - f_eq)
    
    return f_post


@jit(nopython=True)
def streaming_numba(f_post, c, is_solid, opp, Nx, Ny):
    """Streaming step with bounce-back (JIT-compiled).
    
    Args:
        f_post: Post-collision distributions (Nx, Ny, 9)
        c: Lattice velocities (9, 2)
        is_solid: Solid mask (Nx, Ny)
        opp: Opposite direction indices (9,)
        Nx, Ny: Grid dimensions
    
    Returns:
        f_new: Post-streaming distributions (Nx, Ny, 9)
    """
    f_new = np.zeros_like(f_post)
    
    for x in range(Nx):
        for y in range(Ny):
            # Skip solid nodes (will be handled by bounce-back)
            if is_solid[x, y]:
                continue
            
            for i in range(9):
                # Stream from neighbor in opposite direction
                x_prev = x - int(c[i, 0])
                y_prev = y - int(c[i, 1])
                
                # Periodic boundary conditions (wrap around)
                x_prev = (x_prev + Nx) % Nx
                y_prev = (y_prev + Ny) % Ny
                
                # Check if source node is solid
                if is_solid[x_prev, y_prev]:
                    # Bounce back: use opposite direction from current node
                    f_new[x, y, i] = f_post[x, y, opp[i]]
                else:
                    # Normal streaming
                    f_new[x, y, i] = f_post[x_prev, y_prev, i]
    
    return f_new
