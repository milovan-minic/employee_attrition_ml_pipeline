"""
Test script to verify the complete pipeline works correctly
"""

import os
import sys
import warnings

# Add src directory to path
sys.path.append('src')

# Import custom modules
from data_processing import DataProcessor
from feature_engineering import FeatureEngineer
from modeling import ModelTrainer
from evaluation import ModelEvaluator

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def test_pipeline():
    """Test the complete pipeline with sample data"""
    print("🧪 TESTING COMPLETE PIPELINE")
    print("=" * 50)
    
    try:
        # Step 1: Load and process data
        print("\n1. Testing data loading...")
        data_processor = DataProcessor()
        train_data, test_data = data_processor.load_data()
        
        print(f"✅ Training data: {train_data.shape}")
        print(f"✅ Test data: {test_data.shape}")
        
        # Step 2: Exploratory data analysis
        print("\n2. Testing data analysis...")
        data_processor.analyze_data(train_data)
        
        # Step 3: Feature engineering
        print("\n3. Testing feature engineering...")
        feature_engineer = FeatureEngineer()
        X_train, y_train, X_test = feature_engineer.engineer_features(train_data, test_data)
        
        print(f"✅ Training features: {X_train.shape}")
        print(f"✅ Test features: {X_test.shape}")
        print(f"✅ Target shape: {y_train.shape}")
        
        # Step 4: Train models
        print("\n4. Testing model training...")
        model_trainer = ModelTrainer()
        best_model, model_performance = model_trainer.train_models(X_train, y_train)
        
        print(f"✅ Best model: {model_trainer.best_model_name}")
        print(f"✅ Best CV AUC: {max([metrics['cv_mean'] for metrics in model_performance.values()]):.4f}")
        
        # Step 5: Evaluate models
        print("\n5. Testing model evaluation...")
        evaluator = ModelEvaluator()
        evaluator.evaluate_models(X_train, y_train, best_model, model_performance)
        
        # Step 6: Generate predictions
        print("\n6. Testing predictions...")
        predictions = model_trainer.predict(best_model, X_test)
        
        print(f"✅ Predictions generated: {len(predictions)}")
        print(f"✅ Prediction range: {predictions.min():.4f} - {predictions.max():.4f}")
        
        # Step 7: Create submission file
        print("\n7. Testing submission file creation...")
        import pandas as pd
        submission_df = pd.DataFrame({
            'enrollee_id': test_data['enrollee_id'],
            'target': predictions
        })
        
        # Create results directory if it doesn't exist
        os.makedirs('results', exist_ok=True)
        submission_df.to_csv('results/submission.csv', index=False)
        
        print(f"✅ Submission file created: results/submission.csv")
        
        # Step 8: Verify outputs
        print("\n8. Verifying outputs...")
        
        # Check if results directory has files
        results_files = os.listdir('results')
        print(f"✅ Results files created: {len(results_files)} files")
        
        # Check submission file
        if os.path.exists('results/submission.csv'):
            submission_check = pd.read_csv('results/submission.csv')
            print(f"✅ Submission file: {submission_check.shape}")
            print(f"✅ Submission columns: {list(submission_check.columns)}")
        
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Pipeline is working correctly")
        print("✅ Ready for production use")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_pipeline()
    if success:
        print("\n🚀 Pipeline is ready for the technical interview!")
    else:
        print("\n🔧 Please fix the issues before the interview.") 