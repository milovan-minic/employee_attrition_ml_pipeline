#!/usr/bin/env python3
"""
JOB CHANGE PREDICTION - FAST PIPELINE
A faster version for quick testing and demonstration
"""

import os
import sys
import warnings
import pandas as pd
import numpy as np
from datetime import datetime
import time

# Add src to path
sys.path.append('src')

from data_processing import DataProcessor
from feature_engineering import FeatureEngineer
from modeling import ModelTrainer
from evaluation import ModelEvaluator

warnings.filterwarnings('ignore')

def run_fast_pipeline():
    """Run the fast version of the pipeline"""
    print("🚀 JOB CHANGE PREDICTION - FAST PIPELINE")
    print("=" * 50)
    
    start_time = time.time()
    
    try:
        # Step 1: Data Processing
        print("\n📊 Step 1: Data Processing")
        data_processor = DataProcessor()
        train_data, test_data = data_processor.load_data()
        
        # Step 2: Feature Engineering
        print("\n🔧 Step 2: Feature Engineering")
        feature_engineer = FeatureEngineer()
        X_train, y_train, X_test = feature_engineer.engineer_features(train_data, test_data)
        feature_names = feature_engineer.feature_names
        
        print(f"   Training features: {X_train.shape}")
        print(f"   Test features: {X_test.shape}")
        
        # Step 3: Model Training (FAST - no hyperparameter tuning)
        print("\n🤖 Step 3: Model Training (Fast Mode)")
        model_trainer = ModelTrainer()
        best_model, model_performance = model_trainer.train_models(
            X_train, y_train, enable_hyperparameter_tuning=False  # FAST!
        )
        
        # Step 4: Generate Predictions
        print("\n📈 Step 4: Generating Predictions")
        predictions = model_trainer.predict(best_model, X_test)
        
        # Step 5: Create Submission
        print("\n💾 Step 5: Creating Submission File")
        submission_df = pd.DataFrame({
            'enrollee_id': test_data['enrollee_id'],
            'target': predictions
        })
        
        # Ensure output directory exists
        os.makedirs('data/out', exist_ok=True)
        submission_path = 'data/out/submission_fast.csv'
        submission_df.to_csv(submission_path, index=False)
        print(f"   ✅ Submission saved to: {submission_path}")
        
        # Step 6: Model Evaluation
        print("\n📊 Step 6: Model Evaluation")
        evaluator = ModelEvaluator()
        evaluator.evaluate_models(X_train, y_train, best_model, model_performance)
        
        # Step 7: Save Best Model
        print("\n💾 Step 7: Saving Best Model")
        model_trainer.save_model(best_model, 'best_model_fast', model_performance=model_performance)
        
        # Step 8: Generate Results
        print("\n📋 Step 8: Generating Results")
        # Results are already generated in Step 6 (Model Evaluation)
        
        elapsed_time = time.time() - start_time
        print(f"\n✅ Fast pipeline completed in {elapsed_time:.2f} seconds!")
        print(f"🏆 Best model: {model_trainer.best_model_name}")
        print(f"📊 Best CV AUC: {max(model_performance.values(), key=lambda x: x['cv_mean'])['cv_mean']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error in fast pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_fast_pipeline()
    if success:
        print("\n🎉 Fast pipeline completed successfully!")
        print("💡 For full production features, run: python main.py")
    else:
        print("\n❌ Fast pipeline failed. Check the errors above.") 