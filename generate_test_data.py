"""
NASA Exoplanet Detection - Test Data Generator
==============================================

This module generates realistic mock CSV data for testing the exoplanet detection system.
Creates synthetic exoplanet candidates with realistic astronomical parameters.

Usage:
    python generate_test_data.py
"""

import pandas as pd
import numpy as np
from datetime import datetime
import random

class ExoplanetTestDataGenerator:
    """
    Generates realistic test data for exoplanet detection testing
    """
    
    def __init__(self, random_seed=42):
        np.random.seed(random_seed)
        random.seed(random_seed)
        
    def generate_realistic_exoplanet_data(self, num_samples=100):
        """
        Generate realistic exoplanet candidate data
        """
        print(f"Generating {num_samples} realistic exoplanet candidates...")
        
        data = []
        
        for i in range(num_samples):
            # Determine if this will be a real exoplanet or false positive
            is_exoplanet = random.choice([True, False])
            
            # Generate realistic parameters based on research
            if is_exoplanet:
                # Real exoplanet parameters (more constrained)
                orbital_period = np.random.lognormal(2.5, 1.0)  # 1-100 days
                orbital_period = np.clip(orbital_period, 0.5, 500)
                
                transit_duration = orbital_period * np.random.uniform(0.05, 0.15)  # 5-15% of period
                transit_duration = np.clip(transit_duration, 0.5, 24)
                
                planet_radius = np.random.lognormal(0.5, 0.8)  # 0.5-10 Earth radii
                planet_radius = np.clip(planet_radius, 0.3, 15)
                
                transit_depth = (planet_radius / 10) ** 2 * np.random.uniform(100, 1000)  # Realistic depth
                transit_depth = np.clip(transit_depth, 50, 5000)
                
                stellar_temp = np.random.normal(5500, 1000)  # 3500-7500 K
                stellar_temp = np.clip(stellar_temp, 3000, 8000)
                
                stellar_radius = np.random.lognormal(0, 0.3)  # 0.5-2 solar radii
                stellar_radius = np.clip(stellar_radius, 0.3, 3)
                
                stellar_logg = np.random.normal(4.4, 0.3)  # 3.5-5.0
                stellar_logg = np.clip(stellar_logg, 3.0, 5.5)
                
                magnitude = np.random.normal(12, 2)  # 8-16 magnitude
                magnitude = np.clip(magnitude, 6, 18)
                
                impact_param = np.random.uniform(0, 0.8)  # Most transits have low impact
                
                signal_snr = np.random.lognormal(2.5, 0.8)  # 5-50 SNR
                signal_snr = np.clip(signal_snr, 3, 100)
                
                insolation = np.random.lognormal(0, 1.5)  # 0.1-100 Earth flux
                insolation = np.clip(insolation, 0.01, 1000)
                
                equilibrium_temp = stellar_temp * (insolation / 1367) ** 0.25
                equilibrium_temp = np.clip(equilibrium_temp, 100, 3000)
                
                score = np.random.uniform(0.7, 1.0)  # High confidence for real planets
                
            else:
                # False positive parameters (more scattered)
                orbital_period = np.random.lognormal(2.0, 1.5)  # Wider range
                orbital_period = np.clip(orbital_period, 0.1, 1000)
                
                transit_duration = orbital_period * np.random.uniform(0.01, 0.3)  # Very wide range
                transit_duration = np.clip(transit_duration, 0.1, 48)
                
                planet_radius = np.random.lognormal(0, 1.2)  # Very wide range
                planet_radius = np.clip(planet_radius, 0.1, 50)
                
                transit_depth = np.random.lognormal(6, 1.5)  # Random depths
                transit_depth = np.clip(transit_depth, 10, 10000)
                
                stellar_temp = np.random.normal(5000, 1500)  # Wider range
                stellar_temp = np.clip(stellar_temp, 2000, 10000)
                
                stellar_radius = np.random.lognormal(0, 0.6)  # Wider range
                stellar_radius = np.clip(stellar_radius, 0.1, 10)
                
                stellar_logg = np.random.normal(4.0, 0.8)  # Wider range
                stellar_logg = np.clip(stellar_logg, 2.0, 6.0)
                
                magnitude = np.random.normal(13, 3)  # Fainter stars
                magnitude = np.clip(magnitude, 5, 20)
                
                impact_param = np.random.uniform(0, 1.0)  # Random impact
                
                signal_snr = np.random.lognormal(1.5, 1.2)  # Lower SNR
                signal_snr = np.clip(signal_snr, 1, 50)
                
                insolation = np.random.lognormal(0, 2.0)  # Very wide range
                insolation = np.clip(insolation, 0.001, 10000)
                
                equilibrium_temp = stellar_temp * (insolation / 1367) ** 0.25
                equilibrium_temp = np.clip(equilibrium_temp, 50, 5000)
                
                score = np.random.uniform(0.0, 0.6)  # Low confidence for false positives
            
            # Generate coordinates
            ra = np.random.uniform(0, 360)
            dec = np.random.uniform(-90, 90)
            
            # Create data point
            data_point = {
                'koi_period': round(orbital_period, 6),
                'koi_duration': round(transit_duration, 4),
                'koi_depth': round(transit_depth, 2),
                'koi_prad': round(planet_radius, 3),
                'koi_teq': round(equilibrium_temp, 1),
                'koi_steff': round(stellar_temp, 1),
                'koi_srad': round(stellar_radius, 3),
                'koi_slogg': round(stellar_logg, 3),
                'koi_kepmag': round(magnitude, 2),
                'koi_impact': round(impact_param, 3),
                'ra': round(ra, 6),
                'dec': round(dec, 6),
                'koi_model_snr': round(signal_snr, 2),
                'koi_insol': round(insolation, 3),
                'koi_score': round(score, 4),
                'target': 1 if is_exoplanet else 0,
                'candidate_id': f"TEST_{i+1:04d}",
                'discovery_method': 'Transit',
                'mission': random.choice(['Kepler', 'K2', 'TESS']),
                'notes': 'Synthetic test data'
            }
            
            data.append(data_point)
        
        return pd.DataFrame(data)
    
    def generate_simple_test_data(self, num_samples=20):
        """
        Generate simple test data for quick testing
        """
        print(f"Generating {num_samples} simple test candidates...")
        
        data = []
        
        for i in range(num_samples):
            # Simple parameters
            orbital_period = np.random.uniform(1, 50)
            transit_duration = orbital_period * np.random.uniform(0.05, 0.15)
            planet_radius = np.random.uniform(0.5, 5)
            transit_depth = planet_radius ** 2 * np.random.uniform(100, 1000)
            stellar_temp = np.random.uniform(4000, 7000)
            stellar_radius = np.random.uniform(0.5, 2)
            magnitude = np.random.uniform(10, 15)
            impact_param = np.random.uniform(0, 0.8)
            signal_snr = np.random.uniform(5, 30)
            insolation = np.random.uniform(0.1, 10)
            equilibrium_temp = stellar_temp * (insolation / 1367) ** 0.25
            
            data_point = {
                'period': orbital_period,
                'duration': transit_duration,
                'depth': transit_depth,
                'radius': planet_radius,
                'temperature': equilibrium_temp,
                'stellar_temp': stellar_temp,
                'stellar_radius': stellar_radius,
                'magnitude': magnitude,
                'impact': impact_param,
                'signal_snr': signal_snr,
                'insolation': insolation,
                'candidate_id': f"SIMPLE_{i+1:03d}"
            }
            
            data.append(data_point)
        
        return pd.DataFrame(data)
    
    def save_test_data(self, df, filename=None):
        """
        Save generated data to CSV file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"test_exoplanet_data_{timestamp}.csv"
        
        df.to_csv(filename, index=False)
        print(f"Test data saved to: {filename}")
        return filename
    
    def create_multiple_test_files(self):
        """
        Create multiple test files for different scenarios
        """
        print("Creating multiple test data files...")
        
        # 1. Realistic exoplanet data (100 samples)
        realistic_df = self.generate_realistic_exoplanet_data(100)
        self.save_test_data(realistic_df, "realistic_exoplanet_test_data.csv")
        
        # 2. Simple test data (20 samples)
        simple_df = self.generate_simple_test_data(20)
        self.save_test_data(simple_df, "simple_exoplanet_test_data.csv")
        
        # 3. Large dataset (500 samples)
        large_df = self.generate_realistic_exoplanet_data(500)
        self.save_test_data(large_df, "large_exoplanet_test_data.csv")
        
        # 4. Edge cases (50 samples with extreme values)
        edge_df = self.generate_edge_case_data(50)
        self.save_test_data(edge_df, "edge_case_exoplanet_test_data.csv")
        
        print("All test files created successfully!")
    
    def generate_edge_case_data(self, num_samples=50):
        """
        Generate edge case data with extreme values
        """
        print(f"Generating {num_samples} edge case candidates...")
        
        data = []
        
        for i in range(num_samples):
            # Extreme cases
            case_type = random.choice(['hot_jupiter', 'cold_neptune', 'tiny_planet', 'giant_star', 'dwarf_star'])
            
            # Initialize default values
            stellar_radius = np.random.uniform(0.5, 2)  # Default stellar radius
            
            if case_type == 'hot_jupiter':
                orbital_period = np.random.uniform(0.5, 2)  # Very short period
                planet_radius = np.random.uniform(8, 15)  # Large planet
                stellar_temp = np.random.uniform(6000, 8000)  # Hot star
                insolation = np.random.uniform(100, 1000)  # High insolation
                
            elif case_type == 'cold_neptune':
                orbital_period = np.random.uniform(100, 500)  # Long period
                planet_radius = np.random.uniform(3, 6)  # Medium planet
                stellar_temp = np.random.uniform(3000, 5000)  # Cool star
                insolation = np.random.uniform(0.01, 0.1)  # Low insolation
                
            elif case_type == 'tiny_planet':
                orbital_period = np.random.uniform(1, 10)  # Short period
                planet_radius = np.random.uniform(0.1, 0.5)  # Very small planet
                stellar_temp = np.random.uniform(4000, 6000)  # Normal star
                insolation = np.random.uniform(1, 10)  # Normal insolation
                
            elif case_type == 'giant_star':
                orbital_period = np.random.uniform(10, 100)  # Medium period
                planet_radius = np.random.uniform(1, 3)  # Normal planet
                stellar_temp = np.random.uniform(7000, 10000)  # Very hot star
                stellar_radius = np.random.uniform(2, 5)  # Large star
                insolation = np.random.uniform(10, 100)  # High insolation
                
            else:  # dwarf_star
                orbital_period = np.random.uniform(1, 20)  # Short period
                planet_radius = np.random.uniform(0.5, 2)  # Small planet
                stellar_temp = np.random.uniform(2000, 4000)  # Cool star
                stellar_radius = np.random.uniform(0.1, 0.5)  # Small star
                insolation = np.random.uniform(0.1, 1)  # Low insolation
            
            # Calculate derived parameters
            transit_duration = orbital_period * np.random.uniform(0.05, 0.15)
            transit_depth = (planet_radius / 10) ** 2 * np.random.uniform(100, 1000)
            equilibrium_temp = stellar_temp * (insolation / 1367) ** 0.25
            stellar_logg = np.random.uniform(3.5, 5.0)
            magnitude = np.random.uniform(8, 16)
            impact_param = np.random.uniform(0, 0.9)
            signal_snr = np.random.uniform(3, 50)
            
            data_point = {
                'koi_period': round(orbital_period, 6),
                'koi_duration': round(transit_duration, 4),
                'koi_depth': round(transit_depth, 2),
                'koi_prad': round(planet_radius, 3),
                'koi_teq': round(equilibrium_temp, 1),
                'koi_steff': round(stellar_temp, 1),
                'koi_srad': round(stellar_radius, 3),
                'koi_slogg': round(stellar_logg, 3),
                'koi_kepmag': round(magnitude, 2),
                'koi_impact': round(impact_param, 3),
                'ra': round(np.random.uniform(0, 360), 6),
                'dec': round(np.random.uniform(-90, 90), 6),
                'koi_model_snr': round(signal_snr, 2),
                'koi_insol': round(insolation, 3),
                'koi_score': round(np.random.uniform(0.3, 1.0), 4),
                'target': random.choice([0, 1]),
                'candidate_id': f"EDGE_{case_type.upper()}_{i+1:03d}",
                'case_type': case_type,
                'discovery_method': 'Transit',
                'mission': random.choice(['Kepler', 'K2', 'TESS']),
                'notes': f'Edge case: {case_type}'
            }
            
            data.append(data_point)
        
        return pd.DataFrame(data)

def main():
    """
    Main function to generate test data
    """
    print("="*60)
    print("NASA Exoplanet Detection - Test Data Generator")
    print("="*60)
    
    generator = ExoplanetTestDataGenerator()
    
    # Create multiple test files
    generator.create_multiple_test_files()
    
    print("\nGenerated test files:")
    print("1. realistic_exoplanet_test_data.csv - 100 realistic candidates")
    print("2. simple_exoplanet_test_data.csv - 20 simple candidates") 
    print("3. large_exoplanet_test_data.csv - 500 candidates")
    print("4. edge_case_exoplanet_test_data.csv - 50 edge cases")
    
    print("\nYou can now use these files to test the web application!")
    print("Go to 'Make Predictions' -> 'Upload CSV File' in the web app.")

if __name__ == "__main__":
    main()
