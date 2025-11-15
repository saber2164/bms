#!/usr/bin/env python3
"""
Metadata Loader for BMS Dataset

Provides utilities to load and extract per-file parameters from metadata.csv
for improved EKF initialization and parameter selection.

Usage:
    metadata = MetadataLoader('cleaned_dataset/metadata.csv')
    params = metadata.get_ekf_params('00001.csv')
    # Returns: C_nom, R0 (from Re), R_D (from Rct), etc.
"""

import os
import pandas as pd
from typing import Dict, Optional, Tuple


class MetadataLoader:
    """Load and provide per-file metadata for EKF parameter initialization."""
    
    def __init__(self, metadata_path: str):
        """
        Initialize metadata loader.
        
        Parameters
        ----------
        metadata_path : str
            Path to cleaned_dataset/metadata.csv
        """
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        self.metadata_df = pd.read_csv(metadata_path)
        self._index_by_filename()
    
    def _index_by_filename(self):
        """Create filename -> row index mapping for fast lookup."""
        self.filename_index = {}
        for idx, row in self.metadata_df.iterrows():
            filename = row.get('filename', '')
            if filename:
                self.filename_index[filename] = idx
    
    def get_capacity(self, filename: str) -> Optional[float]:
        """
        Get measured capacity for a discharge file.
        
        Parameters
        ----------
        filename : str
            CSV filename (e.g., '00001.csv')
        
        Returns
        -------
        float or None
            Measured capacity in Ah, or None if not available
        """
        if filename not in self.filename_index:
            return None
        
        idx = self.filename_index[filename]
        cap = self.metadata_df.iloc[idx].get('Capacity')
        
        # Handle NaN or missing values
        if pd.isna(cap):
            return None
        
        return float(cap)
    
    def get_series_resistance(self, filename: str) -> Optional[float]:
        """
        Get series resistance (Re) for impedance measurements.
        
        Parameters
        ----------
        filename : str
            CSV filename (e.g., '00002.csv' for impedance)
        
        Returns
        -------
        float or None
            Series resistance in Ohms, or None if not available
        """
        if filename not in self.filename_index:
            return None
        
        idx = self.filename_index[filename]
        re = self.metadata_df.iloc[idx].get('Re')
        
        # Handle NaN or missing values
        if pd.isna(re):
            return None
        
        return float(re)
    
    def get_charge_transfer_resistance(self, filename: str) -> Optional[float]:
        """
        Get charge transfer resistance (Rct) for impedance measurements.
        
        Parameters
        ----------
        filename : str
            CSV filename (e.g., '00002.csv' for impedance)
        
        Returns
        -------
        float or None
            Charge transfer resistance in Ohms, or None if not available
        """
        if filename not in self.filename_index:
            return None
        
        idx = self.filename_index[filename]
        rct = self.metadata_df.iloc[idx].get('Rct')
        
        # Handle NaN or missing values
        if pd.isna(rct):
            return None
        
        return float(rct)
    
    def get_ekf_params(self, filename: str, defaults: Optional[Dict] = None) -> Dict:
        """
        Get EKF parameters for a file, using metadata where available.
        
        Parameters
        ----------
        filename : str
            CSV filename (e.g., '00001.csv')
        defaults : dict, optional
            Default parameters to use as fallback
        
        Returns
        -------
        dict
            EKF parameters with structure:
            {
                'dt': 1.0,
                'C_nom': <from Capacity or default>,
                'R0': <from Re or default>,
                'R_D': <from Rct or default>,
                'C_D': <default>,
                'eta': <default>,
                'ocv_coeffs': None
            }
        """
        if defaults is None:
            defaults = {
                'dt': 1.0,
                'C_nom': 2.3,           # Default nominal capacity (Ah)
                'R0': 0.05,             # Default series resistance (Ohm)
                'R_D': 0.01,            # Default polarization resistance (Ohm)
                'C_D': 500.0,           # Default polarization capacitance (Farad)
                'eta': 0.99,            # Default Coulombic efficiency
            }
        
        params = defaults.copy()
        
        # Override with metadata values if available
        cap = self.get_capacity(filename)
        if cap is not None and cap > 0:
            params['C_nom'] = cap
        
        re = self.get_series_resistance(filename)
        if re is not None and re > 0:
            params['R0'] = re
        
        rct = self.get_charge_transfer_resistance(filename)
        if rct is not None and rct > 0:
            # Use Rct as part of the polarization resistance model
            # Rct represents charge transfer resistance (charge/discharge interface)
            params['R_D'] = rct
        
        return params
    
    def get_battery_id(self, filename: str) -> Optional[str]:
        """Get battery ID for a file."""
        if filename not in self.filename_index:
            return None
        idx = self.filename_index[filename]
        return self.metadata_df.iloc[idx].get('battery_id')
    
    def get_test_type(self, filename: str) -> Optional[str]:
        """Get test type (discharge, charge, impedance) for a file."""
        if filename not in self.filename_index:
            return None
        idx = self.filename_index[filename]
        return self.metadata_df.iloc[idx].get('type')
    
    def get_file_info(self, filename: str) -> Dict:
        """Get all metadata for a file."""
        if filename not in self.filename_index:
            return {}
        idx = self.filename_index[filename]
        row = self.metadata_df.iloc[idx]
        return row.to_dict()
    
    def get_capacity_for_battery(self, battery_id: str) -> Optional[float]:
        """
        Get nominal (maximum) capacity observed for a battery.
        
        Parameters
        ----------
        battery_id : str
            Battery identifier (e.g., 'B0047')
        
        Returns
        -------
        float or None
            Maximum capacity observed for this battery (Ah)
        """
        battery_rows = self.metadata_df[self.metadata_df['battery_id'] == battery_id]
        if battery_rows.empty:
            return None
        
        caps = pd.to_numeric(battery_rows['Capacity'], errors='coerce')
        caps = caps.dropna()
        
        if len(caps) == 0:
            return None
        
        return float(caps.max())


def get_ekf_params_for_file(filename: str, metadata_path: str = None,
                             defaults: Optional[Dict] = None) -> Dict:
    """
    Convenience function to get EKF parameters for a single file.
    
    Parameters
    ----------
    filename : str
        CSV filename (e.g., '00001.csv')
    metadata_path : str, optional
        Path to metadata.csv. If None, uses default: cleaned_dataset/metadata.csv
    defaults : dict, optional
        Default parameters
    
    Returns
    -------
    dict
        EKF parameters
    """
    if metadata_path is None:
        metadata_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'cleaned_dataset',
            'metadata.csv'
        )
    
    loader = MetadataLoader(metadata_path)
    return loader.get_ekf_params(filename, defaults=defaults)


if __name__ == "__main__":
    # Quick test
    loader = MetadataLoader('cleaned_dataset/metadata.csv')
    
    # Test on a discharge file
    params = loader.get_ekf_params('00001.csv')
    print("EKF params for 00001.csv (discharge):")
    print(f"  C_nom: {params['C_nom']:.4f} Ah")
    print(f"  R0:    {params['R0']:.6f} Ohm")
    print(f"  R_D:   {params['R_D']:.6f} Ohm")
    
    # Test on impedance file
    params = loader.get_ekf_params('00002.csv')
    print("\nEKF params for 00002.csv (impedance):")
    print(f"  C_nom: {params['C_nom']:.4f} Ah")
    print(f"  R0:    {params['R0']:.6f} Ohm (from Re)")
    print(f"  R_D:   {params['R_D']:.6f} Ohm (from Rct)")
    
    # Show raw metadata
    info = loader.get_file_info('00001.csv')
    print("\nRaw metadata for 00001.csv:")
    print(f"  Battery: {info.get('battery_id')}")
    print(f"  Type: {info.get('type')}")
    print(f"  Capacity: {info.get('Capacity')}")
    print(f"  Re: {info.get('Re')}")
    print(f"  Rct: {info.get('Rct')}")
