"""
Data Processing Module
----------------------
Handles data loading, cleaning, and exploratory data analysis (EDA) for the job change prediction pipeline.

Key Responsibilities:
- Loads train/test data from the data/in/ directory, ensuring original data is preserved.
- Provides robust sample data generation for demonstration or testing if real data is missing.
- Performs EDA: target distribution, missing value analysis, feature analysis, and correlation analysis.
- Visualizes key data characteristics and saves plots to the results/ directory.

Why this approach?
- Modular design: Keeps data loading/EDA separate from feature engineering and modeling for maintainability.
- Sample data fallback: Ensures the pipeline is always runnable, even if real data is missing (useful for CI/testing).
- Visual EDA: Plots are saved for reporting and presentation, not just printed.
- Handles missing values and data types robustly, anticipating real-world data issues.

Alternatives considered:
- Could have combined EDA and feature engineering, but separation improves clarity and reusability.
- Could have used more advanced EDA libraries (e.g., pandas-profiling), but custom code gives more control and is lighter-weight.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import os

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class DataProcessor:
    """
    Class for handling data loading, cleaning, and analysis.
    - Loads data from disk, or generates sample data if missing.
    - Performs EDA and saves visualizations.
    - Designed for modularity and reusability.
    """
    
    def __init__(self):
        """Initialize the data processor"""
        self.train_data = None
        self.test_data = None
        
    def load_data(self):
        """
        Load training and test datasets from data/in/ directory.
        Returns:
            tuple: (train_data, test_data)
        Why this approach?
        - Always loads from data/in/ to preserve original data.
        - If files are missing, generates sample data for demonstration/testing.
        """
        print("Loading datasets...")
        
        # Always read from data/in/
        in_dir = Path('data/in')
        out_dir = Path('data/out')
        out_dir.mkdir(parents=True, exist_ok=True)
        train_path = in_dir / 'train.csv'
        test_path = in_dir / 'test.csv'
        
        if not train_path.exists() or not test_path.exists():
            print("⚠️  Data files not found in data/in/. Creating sample data for demonstration...")
            return self._create_sample_data()
        
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        
        print(f"✅ Training data loaded: {train_data.shape}")
        print(f"✅ Test data loaded: {test_data.shape}")
        
        return train_data, test_data

    def _create_sample_data(self):
        """
        Create sample data for demonstration purposes and save to data/out/.
        Why?
        - Ensures the pipeline can run even if real data is missing (useful for CI, demos, or onboarding).
        """
        np.random.seed(42)
        n_samples = 1000
        data = {
            'enrollee_id': range(1, n_samples + 1),
            'city': np.random.choice(['city_1', 'city_2', 'city_3', 'city_4', 'city_5'], n_samples),
            'relevent_experience': np.random.choice(['Has relevent experience', 'No relevent experience'], n_samples),
            'education_level': np.random.choice(['Graduate', 'Masters', 'High School', 'Phd', 'Primary School'], n_samples),
            'major_discipline': np.random.choice(['STEM', 'Business Degree', 'Arts', 'Humanities', 'No Major', 'Other'], n_samples),
            'experience': np.random.choice(['<1', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '>20'], n_samples),
            'company_size': np.random.choice(['<10', '10/49', '50-99', '100-500', '500-999', '1000-4999', '5000-9999', '10000+'], n_samples),
            'last_new_job': np.random.choice(['1', '2', '3', '4', '>4', 'never'], n_samples),
            'training_hours': np.random.randint(1, 400, n_samples),
            'target': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
        }
        train_data = pd.DataFrame(data)
        test_data = train_data.copy().drop('target', axis=1)
        out_dir = Path('data/out')
        out_dir.mkdir(parents=True, exist_ok=True)
        train_data.to_csv(out_dir / 'train.csv', index=False)
        test_data.to_csv(out_dir / 'test.csv', index=False)
        return train_data, test_data
    
    def analyze_data(self, train_data):
        """
        Perform comprehensive exploratory data analysis (EDA).
        - Prints and visualizes target distribution, missing values, feature stats, and correlations.
        - Saves plots to results/ for reporting/presentation.
        Why?
        - EDA is essential for understanding data quality and guiding feature engineering/modeling.
        """
        print("\n📊 EXPLORATORY DATA ANALYSIS")
        print("=" * 50)
        
        # Basic info
        print(f"Dataset shape: {train_data.shape}")
        print(f"Memory usage: {train_data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Target distribution
        self._analyze_target_distribution(train_data)
        
        # Missing values analysis
        self._analyze_missing_values(train_data)
        
        # Feature analysis
        self._analyze_features(train_data)
        
        # Correlation analysis
        self._analyze_correlations(train_data)
        
        print("\n✅ Data analysis completed!")
    
    def _analyze_target_distribution(self, data):
        """
        Analyze target variable distribution and visualize it.
        Why?
        - Imbalanced data is a key challenge in this problem; visualizing it guides modeling choices.
        """
        print("\n🎯 TARGET VARIABLE ANALYSIS")
        
        target_counts = data['target'].value_counts()
        target_ratio = target_counts[1] / target_counts[0]
        
        print(f"Class 0 (Not looking): {target_counts[0]} ({target_counts[0]/len(data)*100:.1f}%)")
        print(f"Class 1 (Looking): {target_counts[1]} ({target_counts[1]/len(data)*100:.1f}%)")
        print(f"Imbalance ratio: {target_ratio:.3f}")
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        
        plt.subplot(1, 2, 1)
        data['target'].value_counts().plot(kind='bar', color=['skyblue', 'lightcoral'])
        plt.title('Target Variable Distribution')
        plt.xlabel('Target')
        plt.ylabel('Count')
        
        plt.subplot(1, 2, 2)
        data['target'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['skyblue', 'lightcoral'])
        plt.title('Target Variable Proportion')
        
        plt.tight_layout()
        plt.savefig('results/target_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _analyze_missing_values(self, data):
        """
        Analyze and visualize missing values in the dataset.
        Why?
        - Missing data is common in real-world datasets; understanding patterns informs imputation strategy.
        """
        print("\n🔍 MISSING VALUES ANALYSIS")
        
        missing_data = data.isnull().sum()
        missing_percent = (missing_data / len(data)) * 100
        
        missing_df = pd.DataFrame({
            'Missing_Count': missing_data,
            'Missing_Percent': missing_percent
        }).sort_values('Missing_Percent', ascending=False)
        
        # Check if there are any missing values
        missing_features = missing_df[missing_df['Missing_Count'] > 0]
        
        if len(missing_features) > 0:
            print(missing_features)
            
            # Visualize missing values
            plt.figure(figsize=(12, 6))
            missing_features['Missing_Percent'].plot(kind='bar')
            plt.title('Missing Values by Feature')
            plt.xlabel('Features')
            plt.ylabel('Missing Percentage')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('results/missing_values.png', dpi=300, bbox_inches='tight')
            plt.close()
        else:
            print("✅ No missing values found in the dataset")
            
            # Create a simple plot showing no missing values
            plt.figure(figsize=(8, 4))
            plt.text(0.5, 0.5, 'No Missing Values\nFound in Dataset', 
                    ha='center', va='center', fontsize=14, transform=plt.gca().transAxes)
            plt.title('Missing Values Analysis')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig('results/missing_values.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _analyze_features(self, data):
        """
        Analyze individual features (categorical and numerical).
        Why?
        - Understanding feature distributions and their relationship to the target helps with feature engineering and model selection.
        """
        print("\n📈 FEATURE ANALYSIS")
        
        # Categorical features
        categorical_features = ['city', 'relevent_experience', 'education_level', 
                              'major_discipline', 'experience', 'company_size', 'last_new_job']
        
        # Numerical features
        numerical_features = ['training_hours']
        
        # Analyze categorical features
        for feature in categorical_features:
            if feature in data.columns:
                print(f"\n{feature}:")
                value_counts = data[feature].value_counts()
                print(f"  Unique values: {data[feature].nunique()}")
                print(f"  Most common: {value_counts.index[0]} ({value_counts.iloc[0]} times)")
                
                # Analyze relationship with target
                if 'target' in data.columns:
                    target_ratio = data.groupby(feature)['target'].mean().sort_values(ascending=False)
                    print(f"  Target ratio by {feature}:")
                    for val, ratio in target_ratio.head(3).items():
                        print(f"    {val}: {ratio:.3f}")
        
        # Analyze numerical features
        for feature in numerical_features:
            if feature in data.columns:
                print(f"\n{feature}:")
                print(f"  Mean: {data[feature].mean():.2f}")
                print(f"  Median: {data[feature].median():.2f}")
                print(f"  Std: {data[feature].std():.2f}")
                print(f"  Min: {data[feature].min()}")
                print(f"  Max: {data[feature].max()}")
    
    def _analyze_correlations(self, data):
        """
        Analyze correlations between numerical features and the target.
        Why?
        - Correlation analysis can reveal redundant features or strong predictors.
        """
        print("\n🔗 CORRELATION ANALYSIS")
        
        # Create correlation matrix for numerical features
        numerical_data = data.select_dtypes(include=[np.number])
        
        if len(numerical_data.columns) > 1:
            correlation_matrix = numerical_data.corr()
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
            plt.title('Correlation Matrix')
            plt.tight_layout()
            plt.savefig('results/correlation_matrix.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print("Correlation with target:")
            if 'target' in correlation_matrix.columns:
                target_corr = correlation_matrix['target'].sort_values(ascending=False)
                print(target_corr) 