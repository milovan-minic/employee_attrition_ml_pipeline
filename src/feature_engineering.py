"""
Feature Engineering Module
-------------------------
Handles data preprocessing, missing value imputation, and feature transformation for the job change prediction pipeline.

Key Responsibilities:
- Cleans and imputes missing values using robust, interpretable strategies.
- Creates new features (numeric encodings, ratios, binary flags, etc.) to enhance model performance.
- Encodes categorical variables with label encoding, handling unseen categories in test data.
- Scales numerical features for model compatibility.
- Ensures train and test feature alignment for robust inference.

Why this approach?
- Modular feature engineering: Keeps preprocessing logic separate from EDA and modeling for clarity and reusability.
- Explicit, interpretable feature creation: Business logic is transparent and easy to explain in interviews.
- Robust handling of missing/unseen categories: Prevents pipeline failures on real-world data.
- Label encoding (vs. one-hot): Chosen for high-cardinality features (e.g., city) to avoid dimensionality explosion.

Alternatives considered:
- Could have used one-hot encoding, but label encoding is more scalable for high-cardinality categorical features.
- Could have used automated feature engineering libraries, but custom logic is more transparent.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import warnings

class FeatureEngineer:
    """
    Class for feature engineering and data preprocessing.
    - Handles missing values, feature creation, encoding, and scaling.
    - Designed for modularity, transparency, and robustness.
    """
    
    def __init__(self):
        """Initialize the feature engineer"""
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='most_frequent')
        self.feature_names = None
        
    def engineer_features(self, train_data, test_data):
        """
        Perform comprehensive feature engineering.
        Returns:
            tuple: (X_train, y_train, X_test)
        Why this approach?
        - Separates target and ID columns for clean modeling.
        - Handles missing values and feature creation before encoding/scaling.
        - Ensures train/test alignment for robust inference.
        """
        print("Starting feature engineering...")
        
        # Separate target variable
        y_train = train_data['target'].copy()
        
        # Remove target and ID columns from features
        train_features = train_data.drop(['target', 'enrollee_id'], axis=1, errors='ignore')
        test_features = test_data.drop(['enrollee_id'], axis=1, errors='ignore')
        
        # Handle missing values
        train_features = self._handle_missing_values(train_features)
        test_features = self._handle_missing_values(test_features)
        
        # Create new features
        train_features = self._create_new_features(train_features)
        test_features = self._create_new_features(test_features)
        
        # Encode categorical variables
        train_features = self._encode_categorical_features(train_features)
        test_features = self._encode_categorical_features(test_features)
        
        # Scale numerical features
        train_features = self._scale_numerical_features(train_features)
        test_features = self._scale_numerical_features(test_features)
        
        # Ensure same columns in both datasets
        train_features, test_features = self._align_columns(train_features, test_features)
        
        # Store feature names for later use
        self.feature_names = train_features.columns.tolist()
        
        print(f"✅ Feature engineering completed!")
        print(f"   Final features: {len(self.feature_names)}")
        print(f"   Training shape: {train_features.shape}")
        print(f"   Test shape: {test_features.shape}")
        
        return train_features, y_train, test_features
    
    def _handle_missing_values(self, data):
        """
        Handle missing values in the dataset.
        Why?
        - Categorical: Fill with 'Unknown' to preserve information and avoid dropping rows.
        - Numerical: Fill with median for robustness to outliers.
        """
        print("  Handling missing values...")
        
        # Create a copy to avoid modifying original data
        data_clean = data.copy()
        
        # Handle specific missing value patterns
        for column in data_clean.columns:
            if data_clean[column].dtype == 'object':
                # For categorical variables, fill with 'Unknown'
                data_clean[column] = data_clean[column].fillna('Unknown')
            else:
                # For numerical variables, fill with median
                data_clean[column] = data_clean[column].fillna(data_clean[column].median())
        
        return data_clean
    
    def _create_new_features(self, data):
        """
        Create new features from existing ones (numeric encodings, ratios, binary flags).
        Why?
        - Feature engineering can boost model performance and interpretability.
        - Numeric encodings allow models to leverage ordinal/continuous relationships.
        """
        print("  Creating new features...")
        
        data_new = data.copy()
        
        # Convert experience to numerical
        if 'experience' in data_new.columns:
            data_new['experience_numeric'] = self._convert_experience_to_numeric(data_new['experience'])
        
        # Convert company_size to numerical
        if 'company_size' in data_new.columns:
            data_new['company_size_numeric'] = self._convert_company_size_to_numeric(data_new['company_size'])
        
        # Convert last_new_job to numerical
        if 'last_new_job' in data_new.columns:
            data_new['last_new_job_numeric'] = self._convert_lastnewjob_to_numeric(data_new['last_new_job'])
        
        # Create interaction features
        if 'training_hours' in data_new.columns and 'experience_numeric' in data_new.columns:
            data_new['training_experience_ratio'] = data_new['training_hours'] / (data_new['experience_numeric'] + 1)
        
        # Create training intensity feature
        if 'training_hours' in data_new.columns:
            data_new['training_intensity'] = data_new['training_hours'] / 100  # Normalize by 100
        
        # Create binary features
        if 'relevent_experience' in data_new.columns:
            data_new['has_relevant_experience'] = (data_new['relevent_experience'] == 'Has relevent experience').astype(int)
        
        # Create education level encoding
        if 'education_level' in data_new.columns:
            education_order = ['Primary School', 'High School', 'Graduate', 'Masters', 'Phd']
            data_new['education_level_encoded'] = data_new['education_level'].map(
                {level: idx for idx, level in enumerate(education_order)}
            ).fillna(-1)
        
        return data_new
    
    def _convert_experience_to_numeric(self, experience_series):
        """
        Convert experience categories to numerical values.
        Why?
        - Numeric encoding allows models to learn from ordinal relationships in experience.
        - Median imputation is robust to outliers and missing values.
        """
        experience_mapping = {
            '<1': 0.5,
            '1': 1, '2': 2, '3': 3, '4': 4, '5': 5,
            '6': 6, '7': 7, '8': 8, '9': 9, '10': 10,
            '11': 11, '12': 12, '13': 13, '14': 14, '15': 15,
            '16': 16, '17': 17, '18': 18, '19': 19, '20': 20,
            '>20': 25
        }
        # Convert to numeric, handling missing values
        numeric_experience = experience_series.map(experience_mapping)
        # Fill missing values with median of non-null values
        median_value = numeric_experience.dropna().median()
        return numeric_experience.fillna(median_value)
    
    def _convert_company_size_to_numeric(self, company_size_series):
        """
        Convert company size categories to numerical values.
        Why?
        - Numeric encoding allows models to learn from company size as a continuous variable.
        - Median imputation is robust to missing values.
        """
        size_mapping = {
            '<10': 5,
            '10/49': 30,
            '50-99': 75,
            '100-500': 300,
            '500-999': 750,
            '1000-4999': 3000,
            '5000-9999': 7500,
            '10000+': 15000
        }
        # Convert to numeric, handling missing values
        numeric_size = company_size_series.map(size_mapping)
        # Fill missing values with median of non-null values
        median_value = numeric_size.dropna().median()
        return numeric_size.fillna(median_value)
    
    def _convert_lastnewjob_to_numeric(self, lastnewjob_series):
        """
        Convert last_new_job categories to numerical values.
        Why?
        - Numeric encoding allows models to learn from job change frequency.
        - Median imputation is robust to missing values.
        """
        job_mapping = {
            '1': 1, '2': 2, '3': 3, '4': 4, '>4': 6, 'never': 0
        }
        # Convert to numeric, handling missing values
        numeric_job = lastnewjob_series.map(job_mapping)
        # Fill missing values with median of non-null values
        median_value = numeric_job.dropna().median()
        return numeric_job.fillna(median_value)
    
    def _encode_categorical_features(self, data):
        """
        Encode categorical features using label encoding.
        Why?
        - Label encoding is scalable for high-cardinality features (e.g., city).
        - Handles unseen categories in test data by mapping to -1.
        """
        print("  Encoding categorical features...")
        
        data_encoded = data.copy()
        
        # Identify categorical columns
        categorical_columns = data_encoded.select_dtypes(include=['object']).columns
        
        for column in categorical_columns:
            if column not in self.label_encoders:
                # Create new encoder for this column
                le = LabelEncoder()
                # Fit on training data and transform both train and test
                data_encoded[column] = le.fit_transform(data_encoded[column].astype(str))
                self.label_encoders[column] = le
            else:
                # Use existing encoder for test data
                le = self.label_encoders[column]
                # Handle unseen categories in test data
                data_encoded[column] = data_encoded[column].astype(str).map(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
        
        return data_encoded
    
    def _scale_numerical_features(self, data):
        """
        Scale numerical features using StandardScaler.
        Why?
        - Scaling ensures features are on comparable scales for many ML algorithms.
        - StandardScaler is robust and widely used.
        """
        print("  Scaling numerical features...")
        
        data_scaled = data.copy()
        
        # Identify numerical columns (excluding already encoded categorical)
        numerical_columns = data_scaled.select_dtypes(include=[np.number]).columns
        
        # Remove columns that are already encoded categorical variables
        encoded_categorical = [col for col in numerical_columns if any(cat_col in col for cat_col in self.label_encoders.keys())]
        numerical_columns = [col for col in numerical_columns if col not in encoded_categorical]
        
        if len(numerical_columns) > 0:
            # Scale numerical features
            data_scaled[numerical_columns] = self.scaler.fit_transform(data_scaled[numerical_columns])
        
        return data_scaled
    
    def _align_columns(self, train_data, test_data):
        """
        Ensure train and test data have the same columns.
        Why?
        - Prevents inference errors due to column mismatches.
        - Adds missing columns as zeros for robustness.
        """
        print("  Aligning columns between train and test...")
        
        # Get all unique columns
        all_columns = list(set(train_data.columns) | set(test_data.columns))
        
        # Add missing columns to train data
        for col in all_columns:
            if col not in train_data.columns:
                train_data[col] = 0
        
        # Add missing columns to test data
        for col in all_columns:
            if col not in test_data.columns:
                test_data[col] = 0
        
        # Ensure same order
        train_data = train_data[all_columns]
        test_data = test_data[all_columns]
        
        return train_data, test_data
    
    def get_feature_importance_dataframe(self, feature_importance):
        """
        Create a DataFrame with feature importance for reporting.
        Why?
        - Facilitates easy analysis and visualization of feature importances.
        """
        if self.feature_names is None:
            return None
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        return importance_df 