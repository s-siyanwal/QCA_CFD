"""Unit tests for the D2Q9 lattice module.

Run with: pytest tests/test_lattice.py
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add parent directory to path to import lattice module
sys.path.insert(0, str(Path(__file__).parent.parent))

from lattice import D2Q9Lattice, compute_vorticity, compute_stream_function


def test_lattice_initialization():
    """Test that lattice is properly initialized."""
    lattice = D2Q9Lattice()
    
    # Check shapes
    assert lattice.c.shape == (9, 2), "Lattice velocities should be (9, 2)"
    assert lattice.w.shape == (9,), "Lattice weights should be (9,)"
    assert lattice.opp.shape == (9,), "Opposite directions should be (9,)"
    
    # Check speed of sound
    expected_cs = 1.0 / np.sqrt(3.0)
    assert np.isclose(lattice.cs, expected_cs), f"cs should be {expected_cs}"
    assert np.isclose(lattice.cs2, expected_cs**2), "cs2 should be cs^2"


def test_weight_normalization():
    """Test that lattice weights sum to 1."""
    lattice = D2Q9Lattice()
    weight_sum = np.sum(lattice.w)
    assert np.isclose(weight_sum, 1.0, atol=1e-12), "Weights should sum to 1.0"


def test_opposite_directions():
    """Test that opposite direction mapping is correct."""
    lattice = D2Q9Lattice()
    
    for i in range(9):
        opp_i = lattice.opp[i]
        # Check that c[i] + c[opp[i]] = 0
        sum_vel = lattice.c[i] + lattice.c[opp_i]
        assert np.allclose(sum_vel, [0, 0]), \
            f"Direction {i} and opposite {opp_i} should cancel"


def test_equilibrium_mass_conservation():
    """Test that equilibrium preserves mass."""
    lattice = D2Q9Lattice()
    
    # Create test fields
    Nx, Ny = 10, 10
    rho = np.ones((Nx, Ny))
    u = 0.1 * np.ones((Nx, Ny))
    v = 0.05 * np.ones((Nx, Ny))
    
    # Compute equilibrium
    f_eq = lattice.equilibrium(rho, u, v)
    
    # Check that sum of f_eq equals rho
    rho_computed = np.sum(f_eq, axis=2)
    assert np.allclose(rho_computed, rho), "Equilibrium should conserve mass"


def test_macroscopic_recovery():
    """Test that we can recover macroscopic quantities from equilibrium."""
    lattice = D2Q9Lattice()
    
    # Create test fields
    Nx, Ny = 10, 10
    rho_in = 1.2 * np.ones((Nx, Ny))
    u_in = 0.1 * np.ones((Nx, Ny))
    v_in = 0.05 * np.ones((Nx, Ny))
    
    # Compute equilibrium
    f_eq = lattice.equilibrium(rho_in, u_in, v_in)
    
    # Recover macroscopic quantities
    rho_out, u_out, v_out = lattice.compute_macroscopic(f_eq)
    
    # Check recovery
    assert np.allclose(rho_out, rho_in, atol=1e-10), "Should recover density"
    assert np.allclose(u_out, u_in, atol=1e-10), "Should recover u velocity"
    assert np.allclose(v_out, v_in, atol=1e-10), "Should recover v velocity"


def test_zero_velocity_equilibrium():
    """Test equilibrium at zero velocity."""
    lattice = D2Q9Lattice()
    
    # Create test fields with zero velocity
    Nx, Ny = 10, 10
    rho = np.ones((Nx, Ny))
    u = np.zeros((Nx, Ny))
    v = np.zeros((Nx, Ny))
    
    # Compute equilibrium
    f_eq = lattice.equilibrium(rho, u, v)
    
    # At zero velocity, f_eq[i] should be w[i] * rho
    for i in range(9):
        expected = lattice.w[i] * rho
        assert np.allclose(f_eq[:, :, i], expected), \
            f"At zero velocity, f_eq[{i}] should be w[{i}] * rho"


def test_vorticity_computation():
    """Test vorticity computation for a known flow."""
    # Create a simple vortex flow: u = -y, v = x
    Nx, Ny = 20, 20
    x = np.linspace(-1, 1, Nx)
    y = np.linspace(-1, 1, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    u = -Y
    v = X
    
    # Compute vorticity
    omega = compute_vorticity(u, v, dx=x[1]-x[0])
    
    # For this flow, vorticity should be constant = 2
    # (dv/dx - du/dy = 1 - (-1) = 2)
    expected_vorticity = 2.0
    
    # Check interior points (boundaries may have edge effects)
    interior = omega[2:-2, 2:-2]
    assert np.allclose(interior, expected_vorticity, atol=0.1), \
        "Vorticity should be approximately 2.0 for this flow"


def test_stream_function_computation():
    """Test stream function computation."""
    Nx, Ny = 20, 20
    u = 0.1 * np.ones((Nx, Ny))
    v = np.zeros((Nx, Ny))
    
    psi = compute_stream_function(u, v)
    
    # Check shape
    assert psi.shape == (Nx, Ny), "Stream function should have same shape as velocity"
    
    # For uniform flow in x, psi should increase linearly in y direction
    # Check that differences are approximately constant
    psi_diff = np.diff(psi, axis=1)
    assert np.allclose(psi_diff, psi_diff[0, 0], atol=1e-10), \
        "Stream function should increase uniformly for uniform flow"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
