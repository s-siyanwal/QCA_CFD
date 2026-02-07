"""Configuration Manager for LBM CFD Solver.

This module handles loading, validating, and distributing simulation parameters.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
import jsonschema


class ConfigManager:
    """Load, validate, and provide access to simulation configuration."""
    
    def __init__(self, config_path: str):
        """Initialize configuration manager.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self._validate_config()
        self._compute_derived_parameters()
        self._create_output_directories()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load YAML configuration file.
        
        Returns:
            Configuration dictionary
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If YAML is malformed
        """
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _validate_config(self):
        """Validate configuration against schema.
        
        Raises:
            ValueError: If configuration is invalid
        """
        # Load schema
        schema_path = Path(__file__).parent / 'config_schema.yaml'
        
        if not schema_path.exists():
            print(f"Warning: Schema file not found at {schema_path}. Skipping validation.")
            return
            
        with open(schema_path, 'r') as f:
            schema = yaml.safe_load(f)
        
        try:
            jsonschema.validate(instance=self.config, schema=schema)
            print("✓ Configuration validated successfully")
        except jsonschema.exceptions.ValidationError as e:
            raise ValueError(f"Configuration validation failed: {e.message}")
    
    def _compute_derived_parameters(self):
        """Compute derived parameters (nu, tau, cs2, etc.)."""
        physics = self.config['physics']
        solver = self.config.get('solver', {})
        
        # Compute kinematic viscosity from Reynolds number
        U = physics['characteristic_velocity']
        L = physics['characteristic_length']
        Re = physics['reynolds_number']
        nu = U * L / Re
        
        # Speed of sound squared (D2Q9)
        cs2 = 1.0 / 3.0
        
        # Compute relaxation time from viscosity
        if 'tau' not in solver or solver.get('tau') is None:
            tau = nu / cs2 + 0.5
            self.config['solver']['tau'] = tau
        else:
            tau = solver['tau']
            nu_computed = cs2 * (tau - 0.5)
            if abs(nu - nu_computed) > 1e-6:
                print(f"Warning: Provided tau={tau} gives nu={nu_computed:.6f}, "
                      f"but Re={Re} requires nu={nu:.6f}")
        
        # Store derived parameters
        self.config['derived'] = {
            'nu': nu,
            'tau': self.config['solver']['tau'],
            'cs2': cs2,
            'cs': cs2**0.5
        }
        
        print(f"✓ Computed derived parameters:")
        print(f"  - Kinematic viscosity (nu): {nu:.6f}")
        print(f"  - Relaxation time (tau): {self.config['solver']['tau']:.6f}")
        print(f"  - Speed of sound (cs): {cs2**0.5:.6f}")
    
    def _create_output_directories(self):
        """Create output directory structure."""
        output_dir = Path(self.config['simulation']['output_dir'])
        
        subdirs = ['fields', 'forces', 'vtk', 'videos', 'figures', 'data']
        for subdir in subdirs:
            (output_dir / subdir).mkdir(parents=True, exist_ok=True)
        
        print(f"✓ Created output directories in {output_dir}")
    
    def get(self, key_path: str, default: Optional[Any] = None) -> Any:
        """Get configuration value by dot-separated path.
        
        Args:
            key_path: Dot-separated path (e.g., 'grid.Nx')
            default: Default value if key not found
            
        Returns:
            Configuration value
            
        Example:
            >>> config.get('grid.Nx')
            800
            >>> config.get('nonexistent.key', 42)
            42
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def save_metadata(self):
        """Save configuration + derived parameters for reproducibility."""
        output_dir = Path(self.config['simulation']['output_dir'])
        metadata_path = output_dir / 'config_used.yaml'
        
        with open(metadata_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        print(f"✓ Configuration saved to {metadata_path}")
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"ConfigManager(name='{self.get('simulation.name')}', "
                f"grid={self.get('grid.Nx')}x{self.get('grid.Ny')}, "
                f"Re={self.get('physics.reynolds_number')})")
