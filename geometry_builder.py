"""Geometry Builder for LBM CFD Solver.

This module generates obstacle masks and classifies lattice nodes.
Supports circles, rectangles, and NACA airfoils.
"""

import numpy as np
from typing import Tuple, List, Dict, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath


@dataclass
class Obstacle:
    """Base class for obstacles."""
    
    def is_inside(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return boolean mask: True where (x,y) is inside obstacle.
        
        Args:
            x: x-coordinates (Nx, Ny)
            y: y-coordinates (Nx, Ny)
            
        Returns:
            mask: Boolean array (Nx, Ny)
        """
        raise NotImplementedError


class CircleObstacle(Obstacle):
    """Circular obstacle (cylinder in 2D)."""
    
    def __init__(self, center: Tuple[float, float], radius: float):
        """Initialize circle obstacle.
        
        Args:
            center: (x, y) center coordinates
            radius: Circle radius
        """
        self.center = np.array(center)
        self.radius = radius
    
    def is_inside(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Check if points are inside circle."""
        dx = x - self.center[0]
        dy = y - self.center[1]
        return (dx**2 + dy**2) <= self.radius**2


class RectangleObstacle(Obstacle):
    """Rectangular obstacle."""
    
    def __init__(self, lower_left: Tuple[float, float], 
                 upper_right: Tuple[float, float]):
        """Initialize rectangle obstacle.
        
        Args:
            lower_left: (x, y) lower-left corner
            upper_right: (x, y) upper-right corner
        """
        self.ll = np.array(lower_left)
        self.ur = np.array(upper_right)
    
    def is_inside(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Check if points are inside rectangle."""
        return ((x >= self.ll[0]) & (x <= self.ur[0]) &
                (y >= self.ll[1]) & (y <= self.ur[1]))


class NACAObstacle(Obstacle):
    """NACA 4-digit airfoil obstacle."""
    
    def __init__(self, center: Tuple[float, float], chord: float,
                 naca_digits: str = "0012", angle_deg: float = 0.0):
        """Initialize NACA airfoil obstacle.
        
        Args:
            center: (x, y) center coordinates (leading edge)
            chord: Chord length
            naca_digits: 4-digit NACA designation (e.g., "0012")
            angle_deg: Angle of attack in degrees
        """
        self.center = np.array(center)
        self.chord = chord
        self.naca = naca_digits
        self.angle = np.deg2rad(angle_deg)
        self._compute_airfoil_points()
    
    def _compute_airfoil_points(self):
        """Generate NACA airfoil surface points."""
        # Parse NACA digits: MPXX
        # M = max camber (% of chord)
        # P = position of max camber (tenths of chord)
        # XX = max thickness (% of chord)
        m = int(self.naca[0]) / 100.0
        p = int(self.naca[1]) / 10.0
        t = int(self.naca[2:]) / 100.0
        
        # Generate x coordinates (cosine spacing)
        n_points = 200
        beta = np.linspace(0, np.pi, n_points)
        x_c = 0.5 * (1 - np.cos(beta))  # Normalized [0, 1]
        
        # Thickness distribution (symmetric airfoil)
        y_t = 5 * t * (0.2969*np.sqrt(x_c) - 0.1260*x_c - 0.3516*x_c**2 
                       + 0.2843*x_c**3 - 0.1015*x_c**4)
        
        # Camber line
        if m == 0:
            y_c = np.zeros_like(x_c)
            dyc_dx = np.zeros_like(x_c)
        else:
            y_c = np.where(x_c < p,
                          m/p**2 * (2*p*x_c - x_c**2),
                          m/(1-p)**2 * ((1-2*p) + 2*p*x_c - x_c**2))
            dyc_dx = np.where(x_c < p,
                             2*m/p**2 * (p - x_c),
                             2*m/(1-p)**2 * (p - x_c))
        
        # Surface points
        theta = np.arctan(dyc_dx)
        x_upper = x_c - y_t * np.sin(theta)
        y_upper = y_c + y_t * np.cos(theta)
        x_lower = x_c + y_t * np.sin(theta)
        y_lower = y_c - y_t * np.cos(theta)
        
        # Combine (LE → upper → TE → lower → LE)
        x_contour = np.concatenate([x_upper, x_lower[::-1]])
        y_contour = np.concatenate([y_upper, y_lower[::-1]])
        
        # Scale to chord length
        x_scaled = x_contour * self.chord
        y_scaled = y_contour * self.chord
        
        # Rotate around origin
        x_rot = x_scaled * np.cos(self.angle) - y_scaled * np.sin(self.angle)
        y_rot = x_scaled * np.sin(self.angle) + y_scaled * np.cos(self.angle)
        
        # Translate to center
        self.contour_x = x_rot + self.center[0]
        self.contour_y = y_rot + self.center[1]
    
    def is_inside(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Check if points are inside airfoil using ray-casting."""
        contour_points = np.column_stack([self.contour_x, self.contour_y])
        path = MplPath(contour_points)
        points = np.column_stack([x.ravel(), y.ravel()])
        mask = path.contains_points(points)
        return mask.reshape(x.shape)


class GeometryBuilder:
    """Build simulation geometry with obstacles."""
    
    def __init__(self, config_manager):
        """Initialize geometry builder.
        
        Args:
            config_manager: ConfigManager instance
        """
        self.config = config_manager
        self.Nx = config_manager.get('grid.Nx')
        self.Ny = config_manager.get('grid.Ny')
        self.obstacles: List[Obstacle] = []
        self.is_solid = None
        self._build_obstacles()
    
    def _build_obstacles(self):
        """Parse configuration and instantiate obstacle objects."""
        obstacle_configs = self.config.get('obstacles', [])
        
        for obs_cfg in obstacle_configs:
            obs_type = obs_cfg['type']
            
            if obs_type == 'circle':
                obs = CircleObstacle(obs_cfg['center'], obs_cfg['radius'])
            elif obs_type == 'rectangle':
                obs = RectangleObstacle(obs_cfg['lower_left'], 
                                       obs_cfg['upper_right'])
            elif obs_type == 'naca':
                obs = NACAObstacle(
                    obs_cfg['center'], 
                    obs_cfg['chord'],
                    obs_cfg.get('naca_digits', '0012'),
                    obs_cfg.get('angle_deg', 0.0)
                )
            else:
                raise ValueError(f"Unknown obstacle type: {obs_type}")
            
            self.obstacles.append(obs)
        
        print(f"✓ Loaded {len(self.obstacles)} obstacle(s)")
    
    def generate_mask(self) -> np.ndarray:
        """Generate boolean mask: True = solid, False = fluid.
        
        Returns:
            is_solid: Boolean array (Nx, Ny)
        """
        # Create coordinate grid
        x_grid, y_grid = np.meshgrid(
            np.arange(self.Nx), 
            np.arange(self.Ny), 
            indexing='ij'
        )
        
        # Start with all fluid
        self.is_solid = np.zeros((self.Nx, self.Ny), dtype=bool)
        
        # Mark obstacle nodes as solid
        for obstacle in self.obstacles:
            self.is_solid |= obstacle.is_inside(x_grid, y_grid)
        
        # Statistics
        n_solid = np.sum(self.is_solid)
        n_fluid = np.sum(~self.is_solid)
        total = self.Nx * self.Ny
        
        print(f"✓ Generated geometry mask:")
        print(f"  - Grid size: {self.Nx} × {self.Ny} = {total:,} nodes")
        print(f"  - Fluid nodes: {n_fluid:,} ({n_fluid/total*100:.1f}%)")
        print(f"  - Solid nodes: {n_solid:,} ({n_solid/total*100:.1f}%)")
        
        return self.is_solid
    
    def get_fluid_mask(self) -> np.ndarray:
        """Return fluid mask (inverse of solid mask)."""
        return ~self.is_solid
    
    def visualize_geometry(self, save_path: str = None, figsize=(12, 6)):
        """Plot geometry for verification.
        
        Args:
            save_path: Path to save figure (optional)
            figsize: Figure size (width, height)
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot domain (white = fluid, black = solid)
        ax.imshow(self.is_solid.T, origin='lower', cmap='gray_r', 
                 aspect='equal', interpolation='nearest')
        
        # Overlay obstacle contours if available
        for obstacle in self.obstacles:
            if isinstance(obstacle, NACAObstacle):
                ax.plot(obstacle.contour_x, obstacle.contour_y, 
                       'r-', linewidth=2, alpha=0.7)
            elif isinstance(obstacle, CircleObstacle):
                circle = plt.Circle(obstacle.center, obstacle.radius, 
                                   fill=False, edgecolor='r', 
                                   linewidth=2, alpha=0.7)
                ax.add_patch(circle)
        
        ax.set_xlabel('x (grid points)')
        ax.set_ylabel('y (grid points)')
        ax.set_title('Computational Domain\n(white = fluid, black = solid)')
        ax.set_xlim(0, self.Nx)
        ax.set_ylim(0, self.Ny)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Geometry visualization saved to {save_path}")
        
        plt.show()
    
    def __repr__(self) -> str:
        """String representation."""
        n_obstacles = len(self.obstacles)
        if self.is_solid is not None:
            n_solid = np.sum(self.is_solid)
            return (f"GeometryBuilder(grid={self.Nx}×{self.Ny}, "
                   f"obstacles={n_obstacles}, solid_nodes={n_solid})")
        else:
            return f"GeometryBuilder(grid={self.Nx}×{self.Ny}, obstacles={n_obstacles})"
