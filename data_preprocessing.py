"""
NASA Exoplanet Detection - Data Preprocessing Module
====================================================

This module handles data loading, cleaning, and preprocessing for the 
exoplanet detection ML pipeline based on research findings.

Based on the research papers:
1. "Exoplanet detection using machine learning" (2022) - LightGBM with TSFRESH features
2. "Assessment of Ensemble-Based Machine Learning Algorithms for Exoplanet Identification" (2024) - Ensemble methods
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

class ExoplanetDataProcessor:
    """
    Comprehensive data processor for NASA exoplanet datasets
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.imputer = SimpleImputer(strategy='median')
        self.feature_columns = None
        self.target_column = None
        self.is_fitted = False
        
    def load_koi_dataset(self, file_path="Kepler Objects of Interest (KOI).csv"):
        """Load and preprocess KOI dataset - main dataset for training"""
        print("Loading KOI dataset...")
        df = pd.read_csv(file_path, comment='#')
        
        # Use disposition using Kepler data as target (binary classification)
        df = df[df['koi_pdisposition'].isin(['CANDIDATE', 'FALSE POSITIVE'])]
        
        # Create binary target: 1 for CANDIDATE, 0 for FALSE POSITIVE
        df['target'] = (df['koi_pdisposition'] == 'CANDIDATE').astype(int)
        
        print(f"KOI dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"Target distribution: {df['target'].value_counts().to_dict()}")
        
        return df
    
    def load_toi_dataset(self, file_path="TESS Objects of Interest (TOI).csv"):
        """Load TOI dataset for additional training/testing"""
        print("Loading TOI dataset...")
        df = pd.read_csv(file_path, comment='#')
        
        # Map dispositions to binary target
        # CP (Confirmed Planet) and PC (Planet Candidate) -> 1
        # FP (False Positive) -> 0
        valid_dispositions = ['CP', 'PC', 'FP']
        df = df[df['tfopwg_disp'].isin(valid_dispositions)]
        
        df['target'] = df['tfopwg_disp'].apply(
            lambda x: 1 if x in ['CP', 'PC'] else 0
        )
        
        print(f"TOI dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"Target distribution: {df['target'].value_counts().to_dict()}")
        
        return df
    
    def load_k2_dataset(self, file_path="K2 Planets and Candidates.csv"):
        """Load K2 dataset for additional validation"""
        print("Loading K2 dataset...")
        df = pd.read_csv(file_path, comment='#')
        
        # Map dispositions to binary target
        valid_dispositions = ['CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE']
        df = df[df['disposition'].isin(valid_dispositions)]
        
        df['target'] = df['disposition'].apply(
            lambda x: 1 if x in ['CONFIRMED', 'CANDIDATE'] else 0
        )
        
        print(f"K2 dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"Target distribution: {df['target'].value_counts().to_dict()}")
        
        return df
    
    def select_features(self, df, dataset_type='koi'):
        """
        Select relevant features based on research findings and domain knowledge
        """
        if dataset_type == 'koi':
            # Key features from KOI dataset based on research
            feature_cols = [
                # Orbital characteristics
                'koi_period', 'koi_duration', 'koi_depth', 'koi_impact',
                # Planetary characteristics  
                'koi_prad', 'koi_teq', 'koi_insol',
                # Stellar characteristics
                'koi_steff', 'koi_slogg', 'koi_srad', 'koi_kepmag',
                # Signal characteristics
                'koi_model_snr', 'koi_score',
                # Position
                'ra', 'dec'
            ]
        elif dataset_type == 'toi':
            # Key features from TOI dataset
            feature_cols = [
                # Orbital characteristics
                'pl_orbper', 'pl_trandurh', 'pl_trandep',
                # Planetary characteristics
                'pl_rade', 'pl_insol', 'pl_eqt',
                # Stellar characteristics  
                'st_teff', 'st_logg', 'st_rad', 'st_tmag', 'st_dist',
                # Position
                'ra', 'dec'
            ]
        else:  # k2
            # Key features from K2 dataset
            feature_cols = [
                # Orbital characteristics
                'pl_orbper', 'pl_rade', 'pl_insol', 'pl_eqt',
                # Stellar characteristics
                'st_teff', 'st_rad', 'st_mass', 'st_logg',
                # Position
                'ra', 'dec'
            ]
        
        # Filter to only existing columns
        available_cols = [col for col in feature_cols if col in df.columns]
        print(f"Selected {len(available_cols)} features for {dataset_type} dataset")
        
        return df[available_cols + ['target']]
    
    def preprocess_features(self, df, fit_transform=True):
        """
        Clean and preprocess features
        """
        print("Preprocessing features...")
        
        # Separate features and target
        X = df.drop('target', axis=1)
        y = df['target']
        
        # Handle missing values
        if fit_transform:
            X_imputed = pd.DataFrame(
                self.imputer.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
        else:
            X_imputed = pd.DataFrame(
                self.imputer.transform(X),
                columns=X.columns,
                index=X.index
            )
        
        # Create additional features based on domain knowledge
        X_engineered = self._engineer_features(X_imputed)
        
        # Scale features
        if fit_transform:
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X_engineered),
                columns=X_engineered.columns,
                index=X_engineered.index
            )
            self.feature_columns = X_scaled.columns.tolist()
            self.is_fitted = True
        else:
            if not self.is_fitted:
                raise ValueError("Processor must be fitted on training data before making predictions. Call preprocess_features with fit_transform=True first.")
            X_scaled = pd.DataFrame(
                self.scaler.transform(X_engineered),
                columns=X_engineered.columns,
                index=X_engineered.index
            )
        
        print(f"Preprocessed features: {X_scaled.shape[1]} columns")
        return X_scaled, y
    
    def _engineer_features(self, X):
        """
        Engineer additional features based on domain knowledge
        """
        X_eng = X.copy()
        
        # Planetary characteristics ratios and derived features
        if 'koi_period' in X_eng.columns and 'koi_duration' in X_eng.columns:
            X_eng['period_duration_ratio'] = X_eng['koi_period'] / (X_eng['koi_duration'] + 1e-8)
        
        if 'pl_orbper' in X_eng.columns and 'pl_trandurh' in X_eng.columns:
            X_eng['period_duration_ratio'] = X_eng['pl_orbper'] / (X_eng['pl_trandurh'] + 1e-8)
        
        # Stellar characteristics derived features
        if 'koi_steff' in X_eng.columns and 'koi_srad' in X_eng.columns:
            X_eng['stellar_luminosity'] = (X_eng['koi_steff'] / 5778)**4 * (X_eng['koi_srad'])**2
        
        if 'st_teff' in X_eng.columns and 'st_rad' in X_eng.columns:
            X_eng['stellar_luminosity'] = (X_eng['st_teff'] / 5778)**4 * (X_eng['st_rad'])**2
        
        # Planetary equilibrium temperature validation
        if 'koi_teq' in X_eng.columns and 'koi_steff' in X_eng.columns and 'koi_insol' in X_eng.columns:
            X_eng['teq_theoretical'] = X_eng['koi_steff'] * (X_eng['koi_insol'] / 1367)**0.25
            X_eng['teq_residual'] = X_eng['koi_teq'] - X_eng['teq_theoretical']
        
        # Signal strength indicators
        if 'koi_depth' in X_eng.columns and 'koi_model_snr' in X_eng.columns:
            X_eng['signal_strength'] = X_eng['koi_depth'] * X_eng['koi_model_snr']
        
        # Remove any infinite or very large values
        X_eng = X_eng.replace([np.inf, -np.inf], np.nan)
        X_eng = X_eng.fillna(X_eng.median())
        
        return X_eng
    
    def prepare_training_data(self, test_size=0.2, random_state=42):
        """
        Prepare training and testing datasets
        """
        print("Preparing training data...")
        
        # Load and preprocess main KOI dataset
        koi_df = self.load_koi_dataset()
        koi_features = self.select_features(koi_df, 'koi')
        X_koi, y_koi = self.preprocess_features(koi_features, fit_transform=True)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X_koi, y_koi, test_size=test_size, random_state=random_state, 
            stratify=y_koi
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        return X_train, X_test, y_train, y_test
    
    def prepare_external_validation(self):
        """
        Prepare TOI and K2 datasets for external validation
        """
        print("Preparing external validation datasets...")
        
        validation_sets = {}
        
        try:
            # TOI dataset
            toi_df = self.load_toi_dataset()
            toi_features = self.select_features(toi_df, 'toi')
            
            # Align features with training set
            toi_aligned = self._align_features(toi_features)
            X_toi, y_toi = self.preprocess_features(toi_aligned, fit_transform=False)
            validation_sets['TOI'] = (X_toi, y_toi)
            
        except Exception as e:
            print(f"Warning: Could not prepare TOI validation set: {e}")
        
        try:
            # K2 dataset
            k2_df = self.load_k2_dataset()
            k2_features = self.select_features(k2_df, 'k2')
            
            # Align features with training set
            k2_aligned = self._align_features(k2_features)
            X_k2, y_k2 = self.preprocess_features(k2_aligned, fit_transform=False)
            validation_sets['K2'] = (X_k2, y_k2)
            
        except Exception as e:
            print(f"Warning: Could not prepare K2 validation set: {e}")
        
        return validation_sets
    
    def _align_features(self, df):
        """
        Align features with the training set features
        """
        if self.feature_columns is None:
            raise ValueError("Must fit on training data first")
        
        # Get base features (without engineered ones)
        base_features = [col for col in df.columns if col != 'target']
        
        # Create a dataframe with all required base features
        aligned_df = pd.DataFrame(index=df.index)
        
        # Copy existing features
        for col in base_features:
            if col in df.columns:
                aligned_df[col] = df[col]
        
        # Add missing features with median values
        training_base_features = [col for col in self.feature_columns 
                                if not any(derived in col for derived in 
                                         ['ratio', 'luminosity', 'theoretical', 'residual', 'strength'])]
        
        for col in training_base_features:
            if col not in aligned_df.columns:
                aligned_df[col] = 0  # Will be handled by imputer
        
        aligned_df['target'] = df['target']
        
        return aligned_df

if __name__ == "__main__":
    # Test the preprocessing pipeline
    processor = ExoplanetDataProcessor()
    X_train, X_test, y_train, y_test = processor.prepare_training_data()
    
    print(f"\nTraining features shape: {X_train.shape}")
    print(f"Feature columns: {X_train.columns.tolist()}")
    
    validation_sets = processor.prepare_external_validation()
    for dataset_name, (X_val, y_val) in validation_sets.items():
        print(f"{dataset_name} validation set: {X_val.shape}")
