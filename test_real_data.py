"""
Test script to verify real data loading
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append('src')

from data_processing import DataProcessor

def test_real_data():
    """Test loading real data files"""
    print("🧪 Testing real data loading...")
    
    # Check if data files exist
    data_dir = Path('data')
    train_file = data_dir / 'train.csv'
    test_file = data_dir / 'test.csv'
    
    print(f"Data directory exists: {data_dir.exists()}")
    print(f"Train file exists: {train_file.exists()}")
    print(f"Test file exists: {test_file.exists()}")
    
    if train_file.exists() and test_file.exists():
        print("✅ Data files found!")
        
        # Load data
        data_processor = DataProcessor()
        train_data, test_data = data_processor.load_data()
        
        print(f"✅ Real data loaded successfully!")
        print(f"   Training data: {train_data.shape}")
        print(f"   Test data: {test_data.shape}")
        print(f"   Training columns: {list(train_data.columns)}")
        
        # Check if it's real data (should have ~11k samples)
        if len(train_data) > 10000:
            print("✅ Confirmed: Real data loaded!")
            print(f"   Training samples: {len(train_data):,}")
            print(f"   Test samples: {len(test_data):,}")
            
            # Check target distribution
            if 'target' in train_data.columns:
                target_dist = train_data['target'].value_counts()
                print(f"   Target distribution: {target_dist.to_dict()}")
                print(f"   Imbalance ratio: {target_dist[1]/target_dist[0]:.3f}")
        else:
            print("❌ Sample data loaded instead of real data")
            
        return True
    else:
        print("❌ Data files not found!")
        return False

if __name__ == "__main__":
    success = test_real_data()
    if success:
        print("\n✅ Real data test passed!")
    else:
        print("\n❌ Real data test failed!") 