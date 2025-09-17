"""
NASA Exoplanet Detection - Machine Learning Models Module
=========================================================

This module implements ensemble-based machine learning algorithms for exoplanet identification
based on the research findings from two key papers:

1. "Exoplanet detection using machine learning" (2022) - LightGBM approach
2. "Assessment of Ensemble-Based Machine Learning Algorithms for Exoplanet Identification" (2024) - Ensemble methods

The module implements:
- Stacking Classifier (best performer according to research)
- Random Forest
- LightGBM with feature engineering
- AdaBoost
- Extra Trees
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import (
    RandomForestClassifier, 
    AdaBoostClassifier, 
    ExtraTreesClassifier,
    StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import lightgbm as lgb
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

class ExoplanetMLPipeline:
    """
    Comprehensive ML pipeline for exoplanet detection using ensemble methods
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.model_scores = {}
        self.best_model = None
        self.feature_importance = {}
        # Canonical model name mapping to ensure consistent keys across save/load/UI
        self._canonical_name_by_lower = {
            'lightgbm': 'LightGBM',
            'randomforest': 'RandomForest',
            'extratrees': 'ExtraTrees',
            'adaboost': 'AdaBoost',
            'xgboost': 'XGBoost',
            'stacking': 'Stacking',
        }

    def _normalize_model_name(self, name: str) -> str:
        """Return canonical model name regardless of input capitalization/spacing."""
        if name is None:
            return None
        key = str(name).replace(" ", "").lower()
        return self._canonical_name_by_lower.get(key, name)
        
    def create_base_models(self):
        """
        Create base models based on research findings
        """
        print("Creating base models...")
        
        # LightGBM - showed excellent results in first paper (AUC=0.948)
        lgb_model = lgb.LGBMClassifier(
            objective='binary',
            n_estimators=500,
            learning_rate=0.05,
            max_depth=8,
            num_leaves=31,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
            random_state=self.random_state,
            verbose=-1
        )
        
        # Random Forest - consistently good performer
        rf_model = RandomForestClassifier(
            n_estimators=500,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            bootstrap=True,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # Extra Trees - high randomization, good for ensemble
        et_model = ExtraTreesClassifier(
            n_estimators=500,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            bootstrap=True,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # AdaBoost - adaptive boosting
        ada_model = AdaBoostClassifier(
            n_estimators=200,
            learning_rate=0.1,
            random_state=self.random_state
        )
        
        # XGBoost - gradient boosting
        xgb_model = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=500,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            eval_metric='logloss',
            verbosity=0
        )
        
        self.base_models = {
            'LightGBM': lgb_model,
            'RandomForest': rf_model,
            'ExtraTrees': et_model,
            'AdaBoost': ada_model,
            'XGBoost': xgb_model
        }
        
        return self.base_models
    
    def create_stacking_ensemble(self, base_models=None):
        """
        Create stacking ensemble - best performer according to 2024 research
        """
        if base_models is None:
            base_models = self.create_base_models()
        
        print("Creating stacking ensemble...")
        
        # Use a subset of best performing base models for stacking
        stacking_estimators = [
            ('lightgbm', base_models['LightGBM']),
            ('rf', base_models['RandomForest']),
            ('et', base_models['ExtraTrees']),
            ('xgb', base_models['XGBoost'])
        ]
        
        # Meta-learner: Logistic Regression
        meta_learner = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000
        )
        
        stacking_model = StackingClassifier(
            estimators=stacking_estimators,
            final_estimator=meta_learner,
            cv=5,
            stack_method='predict_proba',
            n_jobs=-1
        )
        
        self.models['Stacking'] = stacking_model
        return stacking_model
    
    def train_models(self, X_train, y_train):
        """
        Train all models
        """
        print("Training models...")
        
        # Create base models
        base_models = self.create_base_models()
        
        # Train individual models
        for name, model in base_models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            self.models[name] = model
        
        # Create and train stacking ensemble
        stacking_model = self.create_stacking_ensemble(base_models)
        print("Training Stacking ensemble...")
        stacking_model.fit(X_train, y_train)
        
        print(f"Trained {len(self.models)} models successfully!")
        
    def evaluate_models(self, X_test, y_test, detailed=True):
        """
        Evaluate all trained models
        """
        print("Evaluating models...")
        
        results = {}
        
        for name, model in self.models.items():
            print(f"\nEvaluating {name}...")
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc': auc
            }
            
            if detailed:
                print(f"{name} Results:")
                print(f"  Accuracy:  {accuracy:.4f}")
                print(f"  Precision: {precision:.4f}")
                print(f"  Recall:    {recall:.4f}")
                print(f"  F1-Score:  {f1:.4f}")
                print(f"  AUC:       {auc:.4f}")
        
        self.model_scores = results
        
        # Find best model based on F1-score (balanced metric)
        best_model_name = max(results.keys(), key=lambda x: results[x]['f1_score'])
        self.best_model = self.models[best_model_name]
        print(f"\nBest model: {best_model_name} (F1: {results[best_model_name]['f1_score']:.4f})")
        
        return results
    
    def cross_validate_models(self, X_train, y_train, cv=5):
        """
        Perform cross-validation for model selection
        """
        print(f"Performing {cv}-fold cross-validation...")
        
        cv_results = {}
        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        for name, model in self.models.items():
            print(f"Cross-validating {name}...")
            
            # Cross-validation scores
            cv_scores = cross_val_score(
                model, X_train, y_train, 
                cv=cv_splitter, 
                scoring='f1',
                n_jobs=-1
            )
            
            cv_results[name] = {
                'mean_f1': cv_scores.mean(),
                'std_f1': cv_scores.std(),
                'scores': cv_scores
            }
            
            print(f"  {name}: F1 = {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return cv_results
    
    def get_feature_importance(self, feature_names, top_n=20):
        """
        Extract feature importance from trained models
        """
        print("Extracting feature importance...")
        
        for name, model in self.models.items():
            if name == 'Stacking':
                # For stacking, we'll use the meta-learner's coefficients
                if hasattr(model.final_estimator_, 'coef_'):
                    importance = np.abs(model.final_estimator_.coef_[0])
                    self.feature_importance[name] = dict(zip(
                        [f"meta_feature_{i}" for i in range(len(importance))], 
                        importance
                    ))
            elif hasattr(model, 'feature_importances_'):
                # Tree-based models
                importance = model.feature_importances_
                self.feature_importance[name] = dict(zip(feature_names, importance))
            elif hasattr(model, 'coef_'):
                # Linear models
                importance = np.abs(model.coef_[0])
                self.feature_importance[name] = dict(zip(feature_names, importance))
        
        # Display top features for best model
        if self.best_model and hasattr(self.best_model, 'feature_importances_'):
            best_model_name = [name for name, model in self.models.items() 
                             if model == self.best_model][0]
            
            if best_model_name in self.feature_importance:
                importance_dict = self.feature_importance[best_model_name]
                top_features = sorted(importance_dict.items(), 
                                    key=lambda x: x[1], reverse=True)[:top_n]
                
                print(f"\nTop {top_n} features for {best_model_name}:")
                for i, (feature, importance) in enumerate(top_features, 1):
                    print(f"  {i:2d}. {feature}: {importance:.4f}")
        
        return self.feature_importance
    
    def predict_new_data(self, X_new, model_name=None):
        """
        Make predictions on new data
        """
        # Normalize requested model name to canonical key
        normalized_name = self._normalize_model_name(model_name) if model_name else None

        if normalized_name is None:
            if self.best_model is None:
                raise ValueError("No trained model available. Train models first.")
            model = self.best_model
            model_name = [name for name, m in self.models.items() if m == self.best_model][0]
        else:
            if normalized_name not in self.models:
                raise ValueError(f"Model {model_name} not found.")
            model = self.models[normalized_name]
            model_name = normalized_name
        
        predictions = model.predict(X_new)
        probabilities = model.predict_proba(X_new)
        
        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'model_used': model_name
        }
    
    def save_models(self, filepath_prefix="exoplanet_models"):
        """
        Save trained models to disk
        """
        print(f"Saving models to disk...")
        
        for name, model in self.models.items():
            filename = f"{filepath_prefix}_{name.lower()}.joblib"
            joblib.dump(model, filename)
            print(f"  Saved {name} to {filename}")
        
        # Save model scores and feature importance
        joblib.dump(self.model_scores, f"{filepath_prefix}_scores.joblib")
        joblib.dump(self.feature_importance, f"{filepath_prefix}_feature_importance.joblib")
        
        print("All models saved successfully!")
    
    def load_models(self, filepath_prefix="exoplanet_models"):
        """
        Load trained models from disk
        """
        print(f"Loading models from disk...")
        
        model_names = ['lightgbm', 'randomforest', 'extratrees', 'adaboost', 'xgboost', 'stacking']
        
        for name in model_names:
            try:
                filename = f"{filepath_prefix}_{name}.joblib"
                model = joblib.load(filename)
                canonical = self._normalize_model_name(name)
                self.models[canonical] = model
                print(f"  Loaded {canonical}")
            except FileNotFoundError:
                print(f"  Warning: {filename} not found")
        
        # Load scores and feature importance
        try:
            self.model_scores = joblib.load(f"{filepath_prefix}_scores.joblib")
            self.feature_importance = joblib.load(f"{filepath_prefix}_feature_importance.joblib")
        except FileNotFoundError:
            print("  Warning: Model scores or feature importance not found")
        
        # Set best model
        if self.model_scores:
            # Use canonical name to retrieve the model from disk-loaded dict
            raw_best_name = max(self.model_scores.keys(), key=lambda x: self.model_scores[x]['f1_score'])
            best_model_name = self._normalize_model_name(raw_best_name)
            # Fallback if scores keys differ in capitalization
            if best_model_name not in self.models and raw_best_name in self.models:
                best_model_name = raw_best_name
            self.best_model = self.models.get(best_model_name)
        
        print("Models loaded successfully!")

if __name__ == "__main__":
    # Test the ML pipeline
    from data_preprocessing import ExoplanetDataProcessor
    
    # Prepare data
    processor = ExoplanetDataProcessor()
    X_train, X_test, y_train, y_test = processor.prepare_training_data()
    
    # Initialize and train models
    ml_pipeline = ExoplanetMLPipeline()
    ml_pipeline.train_models(X_train, y_train)
    
    # Evaluate models
    results = ml_pipeline.evaluate_models(X_test, y_test)
    
    # Cross-validation
    cv_results = ml_pipeline.cross_validate_models(X_train, y_train)
    
    # Feature importance
    feature_importance = ml_pipeline.get_feature_importance(X_train.columns)
    
    # Save models
    ml_pipeline.save_models()
