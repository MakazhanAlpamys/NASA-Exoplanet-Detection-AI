"""
NASA Exoplanet Detection - Model Training Script
================================================

This script trains the ensemble ML models on NASA's exoplanet datasets
and saves them for use in the web application.

Usage:
    python train_models.py

The script will:
1. Load and preprocess the NASA datasets
2. Train multiple ensemble models
3. Evaluate model performance
4. Save trained models for deployment
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from data_preprocessing import ExoplanetDataProcessor
from models import ExoplanetMLPipeline

def create_performance_report(ml_pipeline, X_train, X_test, y_train, y_test, output_dir="reports"):
    """
    Create a comprehensive performance report with visualizations
    """
    print("Creating performance report...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get model scores
    scores = ml_pipeline.model_scores
    
    # Create performance comparison plot
    plt.figure(figsize=(15, 10))
    
    # Subplot 1: Accuracy comparison
    plt.subplot(2, 3, 1)
    models = list(scores.keys())
    accuracies = [scores[model]['accuracy'] for model in models]
    plt.bar(models, accuracies, color='skyblue')
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    # Subplot 2: F1-Score comparison
    plt.subplot(2, 3, 2)
    f1_scores = [scores[model]['f1_score'] for model in models]
    plt.bar(models, f1_scores, color='lightgreen')
    plt.title('Model F1-Score Comparison')
    plt.ylabel('F1-Score')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    # Subplot 3: AUC comparison
    plt.subplot(2, 3, 3)
    auc_scores = [scores[model]['auc'] for model in models]
    plt.bar(models, auc_scores, color='coral')
    plt.title('Model AUC Comparison')
    plt.ylabel('AUC')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    # Subplot 4: Precision vs Recall
    plt.subplot(2, 3, 4)
    precisions = [scores[model]['precision'] for model in models]
    recalls = [scores[model]['recall'] for model in models]
    plt.scatter(recalls, precisions, s=100)
    for i, model in enumerate(models):
        plt.annotate(model, (recalls[i], precisions[i]), 
                    xytext=(5, 5), textcoords='offset points')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision vs Recall')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    # Subplot 5: Performance heatmap
    plt.subplot(2, 3, 5)
    metrics_df = pd.DataFrame(scores).T
    sns.heatmap(metrics_df, annot=True, cmap='viridis', fmt='.3f')
    plt.title('Performance Metrics Heatmap')
    
    # Subplot 6: Feature importance (best model)
    plt.subplot(2, 3, 6)
    if ml_pipeline.feature_importance:
        best_model_name = max(scores.keys(), key=lambda x: scores[x]['f1_score'])
        if best_model_name in ml_pipeline.feature_importance:
            importance = ml_pipeline.feature_importance[best_model_name]
            top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
            features, importances = zip(*top_features)
            plt.barh(range(len(features)), importances)
            plt.yticks(range(len(features)), features)
            plt.xlabel('Importance')
            plt.title(f'Top 10 Features - {best_model_name}')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/model_performance_report.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create detailed text report
    report_text = f"""
NASA Exoplanet Detection - Model Performance Report
==================================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Dataset Information:
- Training samples: {len(X_train)}
- Test samples: {len(X_test)}
- Features: {X_train.shape[1]}
- Class distribution (training): {pd.Series(y_train).value_counts().to_dict()}

Model Performance Summary:
"""
    
    for model_name, model_scores in scores.items():
        report_text += f"""
{model_name}:
  - Accuracy:  {model_scores['accuracy']:.4f}
  - Precision: {model_scores['precision']:.4f}
  - Recall:    {model_scores['recall']:.4f}
  - F1-Score:  {model_scores['f1_score']:.4f}
  - AUC:       {model_scores['auc']:.4f}
"""
    
    # Best model info
    best_model_name = max(scores.keys(), key=lambda x: scores[x]['f1_score'])
    report_text += f"""
Best Performing Model: {best_model_name}
- Best F1-Score: {scores[best_model_name]['f1_score']:.4f}

Research Comparison:
- Target from "Exoplanet detection using machine learning" (2022): AUC = 0.948
- Our best AUC: {max(scores[m]['auc'] for m in scores):.4f}
- Target from "Assessment of Ensemble-Based ML" (2024): >80% accuracy
- Our best accuracy: {max(scores[m]['accuracy'] for m in scores):.1%}

Feature Engineering Impact:
- Used domain knowledge for feature creation
- Engineered features include period/duration ratios, stellar luminosity
- Preprocessing includes missing value imputation and standardization

Cross-Validation Results:
(See cross-validation output above for detailed CV scores)
"""
    
    # Save text report
    with open(f"{output_dir}/performance_report.txt", 'w') as f:
        f.write(report_text)
    
    print(f"Performance report saved to {output_dir}/")
    return report_text

def validate_external_datasets(ml_pipeline, processor):
    """
    Validate models on external datasets (TOI and K2)
    """
    print("\nValidating on external datasets...")
    
    try:
        validation_sets = processor.prepare_external_validation()
        
        external_results = {}
        
        for dataset_name, (X_val, y_val) in validation_sets.items():
            print(f"\nValidating on {dataset_name} dataset ({len(X_val)} samples)...")
            
            if len(X_val) == 0:
                print(f"Warning: No valid samples in {dataset_name} dataset")
                continue
            
            # Make predictions with best model
            result = ml_pipeline.predict_new_data(X_val)
            predictions = result['predictions']
            probabilities = result['probabilities'][:, 1]
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            
            accuracy = accuracy_score(y_val, predictions)
            precision = precision_score(y_val, predictions, zero_division=0)
            recall = recall_score(y_val, predictions, zero_division=0)
            f1 = f1_score(y_val, predictions, zero_division=0)
            
            try:
                auc = roc_auc_score(y_val, probabilities)
            except:
                auc = 0.0
            
            external_results[dataset_name] = {
                'accuracy': accuracy,
                'precision': precision, 
                'recall': recall,
                'f1_score': f1,
                'auc': auc,
                'samples': len(X_val)
            }
            
            print(f"{dataset_name} Results:")
            print(f"  Accuracy:  {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  F1-Score:  {f1:.4f}")
            print(f"  AUC:       {auc:.4f}")
            
    except Exception as e:
        print(f"Warning: External validation failed: {e}")
        external_results = {}
    
    return external_results

def main():
    """
    Main training pipeline
    """
    print("="*60)
    print("NASA Exoplanet Detection - Model Training Pipeline")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize processor and ML pipeline
    print("\n1. Initializing data processor and ML pipeline...")
    processor = ExoplanetDataProcessor()
    ml_pipeline = ExoplanetMLPipeline(random_state=42)
    
    # Prepare training data
    print("\n2. Preparing training data...")
    X_train, X_test, y_train, y_test = processor.prepare_training_data(
        test_size=0.2, 
        random_state=42
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Features: {list(X_train.columns)}")
    
    # Train models
    print("\n3. Training ensemble models...")
    print("This may take several minutes...")
    ml_pipeline.train_models(X_train, y_train)
    
    # Evaluate models
    print("\n4. Evaluating model performance...")
    results = ml_pipeline.evaluate_models(X_test, y_test, detailed=True)
    
    # Cross-validation
    print("\n5. Performing cross-validation...")
    cv_results = ml_pipeline.cross_validate_models(X_train, y_train, cv=5)
    
    # Feature importance analysis
    print("\n6. Analyzing feature importance...")
    feature_importance = ml_pipeline.get_feature_importance(X_train.columns, top_n=15)
    
    # External validation
    print("\n7. External dataset validation...")
    external_results = validate_external_datasets(ml_pipeline, processor)
    
    # Save models
    print("\n8. Saving trained models...")
    ml_pipeline.save_models("exoplanet_models")
    
    # Create performance report
    print("\n9. Creating performance report...")
    report = create_performance_report(ml_pipeline, X_train, X_test, y_train, y_test)
    
    # Final summary
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    best_model = max(results.keys(), key=lambda x: results[x]['f1_score'])
    print(f"Best Model: {best_model}")
    print(f"F1-Score: {results[best_model]['f1_score']:.4f}")
    print(f"AUC: {results[best_model]['auc']:.4f}")
    print(f"Accuracy: {results[best_model]['accuracy']:.4f}")
    
    if external_results:
        print(f"\nExternal Validation:")
        for dataset, scores in external_results.items():
            print(f"  {dataset}: F1={scores['f1_score']:.3f}, AUC={scores['auc']:.3f}")
    
    print(f"\nFiles created:")
    print(f"  - Model files: exoplanet_models_*.joblib")
    print(f"  - Performance report: reports/performance_report.txt")
    print(f"  - Performance plots: reports/model_performance_report.png")
    
    print(f"\nTo run the web application:")
    print(f"  streamlit run web_app.py")
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
