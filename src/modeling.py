"""
Modeling Module
--------------
Handles model training, hyperparameter tuning, prediction generation, model versioning, incremental training, and automated retraining for the job change prediction pipeline.

Key Responsibilities:
- Trains multiple ML models (ensemble, boosting, linear, etc.) with optional hyperparameter optimization.
- Handles class imbalance using SMOTE for robust model performance.
- Selects the best model based on cross-validation AUC.
- Saves models with versioning and metadata for reproducibility and production management.
- Supports incremental training, fine-tuning, and automated retraining pipelines.
- Provides model comparison, feature importance extraction, and ensemble modeling.

Why this approach?
- Modular, extensible design: Easily add new models or retraining strategies.
- SMOTE for imbalance: Outperforms naive resampling for minority class prediction.
- RandomizedSearchCV: Efficient hyperparameter search for production-scale data.
- Model versioning: Ensures traceability and reproducibility in production.
- Incremental/fine-tuning: Enables continuous learning and adaptation to new data.

Alternatives considered:
- Could have used only a single model, but ensemble/model comparison improves robustness.
- Could have used grid search, but randomized search is faster for large parameter spaces.
- Could have used only joblib for model copying, but pickle is more general for in-memory cloning.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import warnings
from datetime import datetime
import json
import os
import pickle

class ModelTrainer:
    """
    Class for training and evaluating multiple models with hyperparameter optimization, versioning, and production features.
    - Trains, tunes, and selects the best model.
    - Handles class imbalance, feature importance, and model persistence.
    - Supports incremental training, fine-tuning, and automated retraining.
    - Designed for extensibility and production-readiness.
    """
    
    def __init__(self):
        """Initialize the model trainer"""
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_importance = None
        self.model_versions = {}
        self.hyperparameter_results = {}
        
    def train_models(self, X_train, y_train, enable_hyperparameter_tuning=True):
        """
        Train multiple models with optional hyperparameter tuning.
        Returns:
            tuple: (best_model, model_performance)
        Why this approach?
        - Trains a diverse set of models for robust selection.
        - Uses SMOTE to address class imbalance.
        - Hyperparameter tuning is optional for speed vs. accuracy tradeoff.
        """
        print("Training multiple models...")
        
        # Define base models (SVM commented out for speed - uncomment if needed)
        base_models = {
            'Random Forest': RandomForestClassifier(random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss'),
            'LightGBM': LGBMClassifier(random_state=42, verbose=-1),
            'CatBoost': CatBoostClassifier(random_state=42, verbose=False),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            # 'SVM': SVC(probability=True, random_state=42)  # Uncomment for SVM (slow!)
        }
        
        # Define hyperparameter grids for optimization
        hyperparameter_grids = {
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'Gradient Boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            },
            'XGBoost': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            },
            'LightGBM': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'num_leaves': [31, 50, 100]
            },
            'CatBoost': {
                'iterations': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'depth': [3, 5, 7],
                'l2_leaf_reg': [1, 3, 5]
            },
            'Logistic Regression': {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            'SVM': {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            }
        }
        
        # Handle class imbalance with SMOTE
        print("  Handling class imbalance with SMOTE...")
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        # Train and evaluate models
        model_performance = {}
        
        for name, model in base_models.items():
            print(f"  Training {name}...")
            
            # Hyperparameter tuning if enabled
            if enable_hyperparameter_tuning and name in hyperparameter_grids:
                print(f"    Performing hyperparameter optimization...")
                best_model = self._optimize_hyperparameters(
                    model, hyperparameter_grids[name], X_train_balanced, y_train_balanced, name
                )
            else:
                best_model = model
            
            # Use cross-validation for evaluation
            cv_scores = cross_val_score(
                best_model, X_train_balanced, y_train_balanced, 
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                scoring='roc_auc'
            )
            
            # Train on full balanced dataset
            best_model.fit(X_train_balanced, y_train_balanced)
            
            # Store model and performance
            self.models[name] = best_model
            model_performance[name] = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'cv_scores': cv_scores,
                'best_params': getattr(best_model, 'best_params_', None)
            }
            
            print(f"    CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Select best model
        best_score = max(model_performance.items(), key=lambda x: x[1]['cv_mean'])
        self.best_model_name = best_score[0]
        self.best_model = self.models[self.best_model_name]
        
        print(f"\n🏆 Best model: {self.best_model_name}")
        print(f"   CV AUC: {best_score[1]['cv_mean']:.4f}")
        
        # Get feature importance if available
        self._extract_feature_importance()
        
        return self.best_model, model_performance
    
    def _optimize_hyperparameters(self, model, param_grid, X_train, y_train, model_name):
        """
        Perform hyperparameter optimization using RandomizedSearchCV.
        Why?
        - Randomized search is faster than grid search for large parameter spaces.
        - n_iter and cv_folds are tuned per model for efficiency.
        """
        # Optimize iterations based on model type
        if model_name == 'SVM':
            n_iter = 5  # Very few iterations for slow SVM
            cv_folds = 2  # Fewer CV folds for SVM
        elif model_name in ['Random Forest', 'Gradient Boosting', 'XGBoost', 'LightGBM', 'CatBoost']:
            n_iter = 10  # Moderate iterations for tree-based models
            cv_folds = 3
        else:
            n_iter = 8  # Default for other models
            cv_folds = 3
        
        # Use RandomizedSearchCV for faster optimization
        random_search = RandomizedSearchCV(
            model, param_grid, n_iter=n_iter, cv=cv_folds, scoring='roc_auc',
            random_state=42, n_jobs=-1, verbose=0
        )
        
        print(f"      Running {n_iter} iterations with {cv_folds}-fold CV...")
        random_search.fit(X_train, y_train)
        
        # Store optimization results
        self.hyperparameter_results[model_name] = {
            'best_params': random_search.best_params_,
            'best_score': random_search.best_score_,
            'cv_results': random_search.cv_results_
        }
        
        print(f"      Best params: {random_search.best_params_}")
        print(f"      Best CV score: {random_search.best_score_:.4f}")
        
        return random_search.best_estimator_
    
    def _extract_feature_importance(self):
        """
        Extract feature importance from the best model.
        Why?
        - Feature importance helps interpret model decisions and guides business actions.
        - Supports both tree-based and linear models.
        """
        if self.best_model is None:
            return
        
        # Try to get feature importance
        if hasattr(self.best_model, 'feature_importances_'):
            self.feature_importance = self.best_model.feature_importances_
        elif hasattr(self.best_model, 'coef_'):
            # For linear models, use absolute coefficients
            self.feature_importance = np.abs(self.best_model.coef_[0])
        else:
            self.feature_importance = None
    
    def predict(self, model, X_test):
        """
        Generate predictions using the trained model.
        Returns:
            np.ndarray: Probability predictions (preferred for imbalanced data).
        Why?
        - Probability outputs allow for threshold tuning and business-driven decision making.
        """
        print(f"Generating predictions with {self.best_model_name}...")
        
        # Get probability predictions
        if hasattr(model, 'predict_proba'):
            predictions = model.predict_proba(X_test)[:, 1]
        else:
            predictions = model.predict(X_test)
        
        print(f"✅ Predictions generated: {len(predictions)} samples")
        return predictions
    
    def save_model(self, model, filename, metadata=None, model_performance=None):
        """
        Save the trained model with metadata and versioning.
        Why?
        - Versioning and metadata tracking are essential for production ML (traceability, rollback, audit).
        - Saves both model and metadata for reproducibility.
        """
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Generate version timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        version = f"{filename}_v{timestamp}"
        
        # Save model
        model_path = f'models/{version}.pkl'
        joblib.dump(model, model_path)
        
        # Get CV AUC from model_performance if available
        cv_auc = None
        if model_performance and self.best_model_name in model_performance:
            cv_auc = model_performance[self.best_model_name]['cv_mean']
        
        # Save metadata
        metadata_path = f'models/{version}_metadata.json'
        model_metadata = {
            'model_name': self.best_model_name,
            'timestamp': timestamp,
            'version': version,
            'cv_auc': cv_auc,
            'feature_count': model.n_features_in_ if hasattr(model, 'n_features_in_') else None,
            'hyperparameters': getattr(model, 'best_params_', model.get_params()),
            'additional_metadata': metadata or {}
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(model_metadata, f, indent=2)
        
        # Store version info
        self.model_versions[version] = {
            'model_path': model_path,
            'metadata_path': metadata_path,
            'metadata': model_metadata
        }
        
        print(f"✅ Model saved to: {model_path}")
        print(f"✅ Metadata saved to: {metadata_path}")
        
        return version
    
    def load_model(self, version):
        """
        Load a trained model by version.
        Why?
        - Enables model rollback, comparison, and reproducibility in production.
        """
        if version not in self.model_versions:
            raise ValueError(f"Model version {version} not found")
        
        model_path = self.model_versions[version]['model_path']
        model = joblib.load(model_path)
        
        print(f"✅ Model loaded from: {model_path}")
        return model
    
    def list_model_versions(self):
        """
        List all available model versions.
        Why?
        - Provides visibility into model history and supports model management.
        """
        if not self.model_versions:
            print("No models have been saved yet.")
            return
        
        print("Available model versions:")
        for version, info in self.model_versions.items():
            metadata = info['metadata']
            print(f"  {version}:")
            cv_auc = metadata.get('cv_auc')
            if cv_auc is not None:
                print(f"    AUC: {cv_auc:.4f}")
            else:
                print("    AUC: N/A")
            print(f"    Timestamp: {metadata['timestamp']}")
            print(f"    Model: {metadata['model_name']}")
    
    def compare_models(self, versions):
        """
        Compare multiple model versions.
        Why?
        - Facilitates model selection and performance tracking over time.
        """
        if len(versions) < 2:
            print("Need at least 2 versions to compare")
            return
        
        comparison_data = []
        for version in versions:
            if version in self.model_versions:
                metadata = self.model_versions[version]['metadata']
                comparison_data.append({
                    'Version': version,
                    'Model': metadata['model_name'],
                    'AUC': metadata.get('cv_auc', 'N/A'),
                    'Timestamp': metadata['timestamp']
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        print("Model Comparison:")
        print(comparison_df.to_string(index=False))
        
        return comparison_df
    
    def create_model_comparison_plot(self, model_performance):
        """
        Create a comparison plot of model performances.
        Why?
        - Visual comparison aids in model selection and reporting.
        """
        plt.figure(figsize=(12, 6))
        
        # Extract model names and scores
        model_names = list(model_performance.keys())
        cv_means = [model_performance[name]['cv_mean'] for name in model_names]
        cv_stds = [model_performance[name]['cv_std'] for name in model_names]
        
        # Create bar plot
        x_pos = np.arange(len(model_names))
        bars = plt.bar(x_pos, cv_means, yerr=cv_stds, capsize=5, 
                      color=['skyblue' if name != self.best_model_name else 'gold' for name in model_names])
        
        # Highlight best model
        if self.best_model_name in model_names:
            best_idx = model_names.index(self.best_model_name)
            bars[best_idx].set_color('gold')
            bars[best_idx].set_edgecolor('orange')
            bars[best_idx].set_linewidth(2)
        
        plt.xlabel('Models')
        plt.ylabel('Cross-Validation AUC Score')
        plt.title('Model Performance Comparison')
        plt.xticks(x_pos, model_names, rotation=45, ha='right')
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for i, (bar, mean, std) in enumerate(zip(bars, cv_means, cv_stds)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{mean:.3f}\n±{std:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('results/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_feature_importance_plot(self, feature_names):
        """
        Create a feature importance plot for the best model.
        Why?
        - Visualizes which features drive model predictions, supporting business interpretation.
        """
        if self.feature_importance is None:
            print("⚠️  Feature importance not available for this model")
            return
        
        # Create feature importance DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.feature_importance
        }).sort_values('importance', ascending=False)
        
        # Plot top 15 features
        top_features = importance_df.head(15)
        
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'Top 15 Feature Importance - {self.best_model_name}')
        plt.gca().invert_yaxis()
        
        # Add value labels
        for i, (bar, importance) in enumerate(zip(bars, top_features['importance'])):
            plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{importance:.4f}', ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('results/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save feature importance to CSV
        importance_df.to_csv('results/feature_importance.csv', index=False)
        print("✅ Feature importance saved to: results/feature_importance.csv") 

    def fine_tune_model(self, model, X_new, y_new, learning_rate_factor=0.1):
        """
        Fine-tune an existing model with new data using a lower learning rate.
        Why?
        - Fine-tuning allows the model to adapt to new data without full retraining.
        - Lower learning rate prevents catastrophic forgetting.
        """
        print(f"Fine-tuning {self.best_model_name} with new data...")
        
        # Create a copy of the model for fine-tuning
        fine_tuned_model = pickle.loads(pickle.dumps(model))
        
        # Adjust learning rate for fine-tuning
        if hasattr(fine_tuned_model, 'learning_rate'):
            original_lr = fine_tuned_model.learning_rate
            fine_tuned_model.learning_rate = original_lr * learning_rate_factor
            print(f"  Adjusted learning rate: {original_lr} -> {fine_tuned_model.learning_rate}")
        
        # For tree-based models, we can continue training
        if hasattr(fine_tuned_model, 'fit'):
            try:
                # Continue training with new data
                fine_tuned_model.fit(X_new, y_new)
                print("  ✅ Model fine-tuned successfully")
                return fine_tuned_model
            except Exception as e:
                print(f"  ⚠️  Fine-tuning failed: {e}")
                return model
        else:
            print("  ⚠️  Model type doesn't support fine-tuning")
            return model
    
    def incremental_train(self, model, X_new, y_new, method='warm_start'):
        """
        Incrementally train a model with new data.
        Why?
        - Supports continuous learning and adaptation in production.
        - Offers both warm_start and ensemble options for flexibility.
        """
        print(f"Incremental training of {self.best_model_name}...")
        
        if method == 'warm_start':
            # Use warm_start for tree-based models
            if hasattr(model, 'warm_start'):
                model.warm_start = True
                model.fit(X_new, y_new)
                print("  ✅ Incremental training completed with warm_start")
                return model
            else:
                print("  ⚠️  Model doesn't support warm_start")
                return model
        elif method == 'ensemble':
            # Create ensemble with existing and new model
            new_model = pickle.loads(pickle.dumps(model))
            new_model.fit(X_new, y_new)
            
            # Create ensemble (simple average)
            ensemble_model = EnsembleModel([model, new_model])
            print("  ✅ Ensemble model created")
            return ensemble_model
        else:
            print("  ⚠️  Unknown incremental training method")
            return model
    
    def retrain_with_validation(self, model, X_train, y_train, X_val, y_val, retrain_threshold=0.02):
        """
        Retrain model if validation performance drops below threshold.
        Why?
        - Ensures model quality is maintained in production.
        - Only retrains if significant improvement is possible.
        """
        print("Checking if retraining is needed...")
        
        # Evaluate current model on validation set
        current_score = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
        
        # Train new model
        new_model = pickle.loads(pickle.dumps(model))
        new_model.fit(X_train, y_train)
        
        # Evaluate new model
        new_score = roc_auc_score(y_val, new_model.predict_proba(X_val)[:, 1])
        
        improvement = new_score - current_score
        
        print(f"  Current validation AUC: {current_score:.4f}")
        print(f"  New model validation AUC: {new_score:.4f}")
        print(f"  Improvement: {improvement:.4f}")
        
        if improvement > retrain_threshold:
            print("  ✅ Retraining recommended - significant improvement detected")
            return new_model, improvement
        else:
            print("  ⚠️  No retraining needed - improvement below threshold")
            return model, improvement
    
    def auto_retrain_pipeline(self, X_train, y_train, X_val, y_val, retrain_schedule='weekly', performance_threshold=0.02):
        """
        Automated retraining pipeline with performance monitoring.
        Why?
        - Simulates production retraining cycles with performance tracking and versioning.
        - Saves performance history for audit and reporting.
        """
        print("Setting up automated retraining pipeline...")
        
        # Save initial model
        initial_version = self.save_model(self.best_model, "initial_model")
        
        # Monitor performance over time
        performance_history = []
        
        # Simulate periodic retraining (in real scenario, this would be scheduled)
        for epoch in range(3):  # Simulate 3 retraining cycles
            print(f"\n--- Retraining Cycle {epoch + 1} ---")
            
            # Check if retraining is needed
            updated_model, improvement = self.retrain_with_validation(
                self.best_model, X_train, y_train, X_val, y_val, performance_threshold
            )
            
            if improvement > performance_threshold:
                # Save new model version
                new_version = self.save_model(updated_model, f"retrained_model_epoch_{epoch}")
                self.best_model = updated_model
                
                performance_history.append({
                    'epoch': epoch,
                    'version': new_version,
                    'improvement': improvement,
                    'timestamp': datetime.now().isoformat()
                })
                
                print(f"  ✅ Model updated to version: {new_version}")
            else:
                print(f"  ⚠️  No update needed for epoch {epoch}")
        
        # Save performance history
        with open('models/performance_history.json', 'w') as f:
            json.dump(performance_history, f, indent=2)
        
        print(f"\n✅ Automated retraining completed. Performance history saved.")
        return performance_history


class EnsembleModel:
    """
    Simple ensemble model for combining multiple models (average probabilities).
    Why?
    - Ensemble methods often improve predictive performance and robustness.
    - Simple averaging is interpretable and easy to implement.
    """
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights if weights else [1/len(models)] * len(models)
    
    def predict_proba(self, X):
        predictions = []
        for model in self.models:
            pred = model.predict_proba(X)[:, 1]
            predictions.append(pred)
        
        # Weighted average
        ensemble_pred = np.zeros(len(X))
        for pred, weight in zip(predictions, self.weights):
            ensemble_pred += pred * weight
        
        return np.column_stack([1 - ensemble_pred, ensemble_pred])
    
    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int) 