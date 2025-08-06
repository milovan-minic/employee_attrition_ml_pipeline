"""
Production Deployment Script
Demonstrates all production-ready features of the ML pipeline
"""

import os
import sys
import warnings
import pandas as pd
import numpy as np
from datetime import datetime
import json

# Add src directory to path
sys.path.append('src')

# Import custom modules
from data_processing import DataProcessor
from feature_engineering import FeatureEngineer
from modeling import ModelTrainer
from evaluation import ModelEvaluator
from model_monitoring import ModelMonitor, ProductionMetrics

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def deploy_production_pipeline():
    """
    Deploy the complete production pipeline with all features
    """
    print("🚀 DEPLOYING PRODUCTION ML PIPELINE")
    print("=" * 60)
    
    # Create necessary directories
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Initialize components
    data_processor = DataProcessor()
    feature_engineer = FeatureEngineer()
    model_trainer = ModelTrainer()
    evaluator = ModelEvaluator()
    
    try:
        # Step 1: Load and process data
        print("\n1️⃣ Loading and processing data...")
        train_data, test_data = data_processor.load_data()
        
        # Step 2: Feature engineering
        print("\n2️⃣ Engineering features...")
        X_train, y_train, X_test = feature_engineer.engineer_features(train_data, test_data)
        
        # Step 3: Train models with hyperparameter optimization
        print("\n3️⃣ Training models with hyperparameter optimization...")
        best_model, model_performance = model_trainer.train_models(
            X_train, y_train, enable_hyperparameter_tuning=True
        )
        
        # Step 4: Save production model with metadata
        print("\n4️⃣ Saving production model...")
        model_metadata = {
            'dataset_size': len(train_data),
            'feature_count': len(X_train.columns),
            'class_distribution': y_train.value_counts().to_dict(),
            'training_timestamp': datetime.now().isoformat(),
            'pipeline_version': '1.0.0',
            'hyperparameter_optimization': 'enabled',
            'model_versioning': 'enabled'
        }
        
        model_version = model_trainer.save_model(
            best_model, "production_model", metadata=model_metadata
        )
        
        # Step 5: Setup model monitoring
        print("\n5️⃣ Setting up model monitoring...")
        monitor = ModelMonitor()
        monitor.setup_automated_monitoring(
            model_path=f'models/{model_version}.pkl',
            baseline_data_path='data/train.csv',
            monitoring_schedule='daily'
        )
        
        # Step 6: Initialize production metrics
        print("\n6️⃣ Initializing production metrics...")
        production_metrics = ProductionMetrics()
        
        # Step 7: Generate predictions and track metrics
        print("\n7️⃣ Generating predictions with metrics tracking...")
        import time
        
        start_time = time.time()
        predictions = model_trainer.predict(best_model, X_test)
        prediction_time = time.time() - start_time
        
        # Update production metrics
        production_metrics.update_metrics(prediction_time=prediction_time)
        production_metrics.save_metrics()
        
        # Step 8: Create submission file
        print("\n8️⃣ Creating submission file...")
        submission_df = pd.DataFrame({
            'enrollee_id': test_data['enrollee_id'],
            'target': predictions
        })
        submission_df.to_csv('results/submission.csv', index=False)
        
        # Step 9: Demonstrate incremental training
        print("\n9️⃣ Demonstrating incremental training...")
        # Split data to simulate new data arrival
        from sklearn.model_selection import train_test_split
        X_train_new, X_new, y_train_new, y_new = train_test_split(
            X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
        )
        
        # Fine-tune model with new data
        fine_tuned_model = model_trainer.fine_tune_model(
            best_model, X_new, y_new, learning_rate_factor=0.1
        )
        
        # Save fine-tuned model
        fine_tuned_version = model_trainer.save_model(
            fine_tuned_model, "fine_tuned_model", 
            metadata={'fine_tuning_timestamp': datetime.now().isoformat()}
        )
        
        # Step 10: Demonstrate model comparison
        print("\n🔟 Comparing model versions...")
        model_trainer.list_model_versions()
        
        # Step 11: Generate comprehensive production report
        print("\n1️⃣1️⃣ Generating production report...")
        generate_comprehensive_report(
            model_trainer, model_performance, model_version, 
            fine_tuned_version, production_metrics
        )
        
        print(f"\n✅ PRODUCTION PIPELINE DEPLOYED SUCCESSFULLY!")
        print(f"📊 Submission file: results/submission.csv")
        print(f"📈 Production model: models/{model_version}.pkl")
        print(f"🔄 Fine-tuned model: models/{fine_tuned_version}.pkl")
        print(f"📋 Monitoring config: models/monitoring_config.json")
        print(f"📊 Production metrics: results/production_metrics.json")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Deployment failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def generate_comprehensive_report(model_trainer, model_performance, model_version, 
                                fine_tuned_version, production_metrics):
    """
    Generate comprehensive production deployment report
    """
    report = {
        'deployment_info': {
            'timestamp': datetime.now().isoformat(),
            'pipeline_version': '1.0.0',
            'deployment_status': 'successful'
        },
        'model_versions': {
            'production_model': model_version,
            'fine_tuned_model': fine_tuned_version
        },
        'performance_summary': {
            'best_model': model_trainer.best_model_name,
            'cv_auc': model_performance[model_trainer.best_model_name]['cv_mean'],
            'total_models_trained': len(model_performance)
        },
        'production_features': {
            'hyperparameter_optimization': '✅ Enabled',
            'model_versioning': '✅ Enabled',
            'automated_retraining': '✅ Enabled',
            'performance_monitoring': '✅ Enabled',
            'incremental_training': '✅ Enabled',
            'fine_tuning': '✅ Enabled',
            'ensemble_methods': '✅ Enabled',
            'metadata_tracking': '✅ Enabled',
            'drift_detection': '✅ Enabled',
            'production_metrics': '✅ Enabled'
        },
        'hyperparameter_optimization_results': {
            name: {
                'best_params': results['best_params'],
                'best_score': results['best_score']
            }
            for name, results in model_trainer.hyperparameter_results.items()
        },
        'production_metrics': production_metrics.get_metrics_summary()
    }
    
    # Save comprehensive report
    with open('results/production_deployment_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("✅ Comprehensive production report saved")
    
    # Print summary
    print(f"\n📋 PRODUCTION DEPLOYMENT SUMMARY:")
    print(f"   Best Model: {model_trainer.best_model_name}")
    print(f"   CV AUC: {model_performance[model_trainer.best_model_name]['cv_mean']:.4f}")
    print(f"   Models Trained: {len(model_performance)}")
    print(f"   Production Features: {len(report['production_features'])}")
    print(f"   Model Versions: {len(model_trainer.model_versions)}")

def demonstrate_production_features():
    """
    Demonstrate all production-ready features
    """
    print("\n🎯 DEMONSTRATING PRODUCTION FEATURES")
    print("=" * 50)
    
    features = [
        "✅ Hyperparameter Optimization",
        "✅ Model Versioning & Metadata",
        "✅ Automated Retraining Pipeline",
        "✅ Performance Monitoring",
        "✅ Incremental Training",
        "✅ Fine-tuning Capabilities",
        "✅ Ensemble Methods",
        "✅ Drift Detection",
        "✅ Production Metrics Tracking",
        "✅ Comprehensive Reporting"
    ]
    
    for feature in features:
        print(f"   {feature}")
    
    print(f"\n🚀 All production features are now available!")
    print(f"   Run 'python deploy_production.py' to deploy the complete pipeline")

if __name__ == "__main__":
    success = deploy_production_pipeline()
    if success:
        demonstrate_production_features()
    else:
        print("\n❌ Production deployment failed. Please check the errors above.") 