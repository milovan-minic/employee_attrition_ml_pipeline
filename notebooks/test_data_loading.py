"""
Test script to verify data loading from notebooks directory
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append('..')

# Import custom modules
from src.data_processing import DataProcessor

def test_data_loading():
    """Test data loading from notebooks directory"""
    print("🧪 Testing data loading from notebooks directory...")
    
    try:
        # Initialize data processor
        data_processor = DataProcessor()
        
        # Load data
        train_data, test_data = data_processor.load_data()
        
        print(f"✅ Data loaded successfully!")
        print(f"   Training data: {train_data.shape}")
        print(f"   Test data: {test_data.shape}")
        print(f"   Training columns: {list(train_data.columns)}")
        print(f"   Test columns: {list(test_data.columns)}")
        
        # Check target distribution
        if 'target' in train_data.columns:
            target_dist = train_data['target'].value_counts()
            print(f"   Target distribution: {target_dist.to_dict()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_data_loading()
    if success:
        print("\n✅ Data loading test passed!")
    else:
        print("\n❌ Data loading test failed!") 