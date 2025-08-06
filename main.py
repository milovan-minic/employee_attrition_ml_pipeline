"""
Main script for Job Change Prediction
This script orchestrates the complete data science pipeline:
1. Data loading and preprocessing
2. Exploratory data analysis
3. Feature engineering
4. Model training and evaluation
5. Prediction generation
6. Production-ready model management
"""

import os
import sys
import warnings
import pandas as pd
import numpy as np
from datetime import datetime

# Add src directory to path
sys.path.append('src')

# Import custom modules
from data_processing import DataProcessor
from feature_engineering import FeatureEngineer
from modeling import ModelTrainer
from evaluation import ModelEvaluator

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def main():
    """
    Main execution function that runs the complete pipeline
    """
    print("=" * 60)
    print("JOB CHANGE PREDICTION - PRODUCTION PIPELINE")
    print("=" * 60)
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Initialize pipeline components
    data_processor = DataProcessor()
    feature_engineer = FeatureEngineer()
    model_trainer = ModelTrainer()
    evaluator = ModelEvaluator()
    
    try:
        # Step 1: Load and process data
        print("\n1. Loading and processing data...")
        train_data, test_data = data_processor.load_data()
        
        # Step 2: Exploratory data analysis
        print("\n2. Performing exploratory data analysis...")
        data_processor.analyze_data(train_data)
        
        # Step 3: Feature engineering
        print("\n3. Engineering features...")
        X_train, y_train, X_test = feature_engineer.engineer_features(train_data, test_data)
        
        # Step 4: Train models with hyperparameter optimization
        print("\n4. Training models with hyperparameter optimization...")
        best_model, model_performance = model_trainer.train_models(
            X_train, y_train, enable_hyperparameter_tuning=True
        )
        
        # Step 5: Save the best model with metadata
        print("\n5. Saving production model...")
        model_metadata = {
            'dataset_size': len(train_data),
            'feature_count': len(X_train.columns),
            'class_distribution': y_train.value_counts().to_dict(),
            'training_timestamp': datetime.now().isoformat(),
            'pipeline_version': '1.0.0'
        }
        
        model_version = model_trainer.save_model(
            best_model, "production_model", metadata=model_metadata
        )
        
        # Step 6: Evaluate models
        print("\n6. Evaluating models...")
        evaluator.evaluate_models(X_train, y_train, best_model, model_performance)
        
        # Step 7: Generate predictions
        print("\n7. Generating predictions...")
        predictions = model_trainer.predict(best_model, X_test)
        
        # Step 8: Create submission file
        print("\n8. Creating submission file...")
        submission_df = pd.DataFrame({
            'enrollee_id': test_data['enrollee_id'],
            'target': predictions
        })
        submission_df.to_csv('results/submission.csv', index=False)
        
        # Step 9: Model management and versioning
        print("\n9. Model management and versioning...")
        model_trainer.list_model_versions()
        
        # Step 10: Performance monitoring setup
        print("\n10. Setting up performance monitoring...")
        # Split data for validation (in production, this would be separate)
        from sklearn.model_selection import train_test_split
        X_train_monitor, X_val_monitor, y_train_monitor, y_val_monitor = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        # Simulate automated retraining pipeline
        print("\n11. Testing automated retraining pipeline...")
        performance_history = model_trainer.auto_retrain_pipeline(
            X_train_monitor, y_train_monitor, X_val_monitor, y_val_monitor
        )
        
        # Step 12: Generate production report
        print("\n12. Generating production report...")
        generate_production_report(model_trainer, model_performance, model_version)
        
        print(f"\n✅ Production pipeline completed successfully!")
        print(f"📊 Submission file saved to: results/submission.csv")
        print(f"📈 Model saved to: models/{model_version}.pkl")
        print(f"📋 Performance history saved to: models/performance_history.json")
        
    except Exception as e:
        print(f"\n❌ Error in pipeline: {str(e)}")
        raise

def generate_production_report(model_trainer, model_performance, model_version):
    """
    Generate comprehensive production report
    """
    report = {
        'pipeline_execution': {
            'timestamp': datetime.now().isoformat(),
            'model_version': model_version,
            'best_model': model_trainer.best_model_name,
            'total_models_trained': len(model_performance)
        },
        'model_performance': {
            name: {
                'cv_auc_mean': metrics['cv_mean'],
                'cv_auc_std': metrics['cv_std'],
                'best_params': metrics.get('best_params', {})
            }
            for name, metrics in model_performance.items()
        },
        'hyperparameter_optimization': {
            name: {
                'best_params': results['best_params'],
                'best_score': results['best_score']
            }
            for name, results in model_trainer.hyperparameter_results.items()
        },
        'production_ready_features': [
            'Hyperparameter optimization',
            'Model versioning',
            'Automated retraining',
            'Performance monitoring',
            'Incremental training',
            'Fine-tuning capabilities',
            'Ensemble methods',
            'Metadata tracking'
        ]
    }
    
    # Save production report
    import json
    with open('results/production_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("✅ Production report saved to: results/production_report.json")
    
    # Print summary
    print(f"\n📋 PRODUCTION SUMMARY:")
    print(f"   Best Model: {model_trainer.best_model_name}")
    print(f"   Model Version: {model_version}")
    print(f"   CV AUC: {model_performance[model_trainer.best_model_name]['cv_mean']:.4f}")
    print(f"   Models Trained: {len(model_performance)}")
    print(f"   Hyperparameter Optimization: ✅ Enabled")
    print(f"   Model Versioning: ✅ Enabled")
    print(f"   Automated Retraining: ✅ Enabled")

if __name__ == "__main__":
    main() 