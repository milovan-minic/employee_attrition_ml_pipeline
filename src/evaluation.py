"""
Evaluation Module
----------------
Handles model evaluation, performance metrics, and result visualization for the job change prediction pipeline.

Key Responsibilities:
- Evaluates models using cross-validation, ROC AUC, precision-recall, and confusion matrix metrics.
- Generates and saves comprehensive plots (ROC, PR, confusion matrix, feature importance, model comparison).
- Produces detailed performance reports for business and technical stakeholders.

Why this approach?
- Modular evaluation: Keeps evaluation logic separate from modeling for clarity and reusability.
- Multiple metrics: Imbalanced data requires more than just accuracy (AUC, PR, F1, etc.).
- Visualizations: Plots are essential for presentations and business communication.
- Saves all results to disk for reproducibility and reporting.

Alternatives considered:
- Could have used automated reporting libraries, but custom code gives more control and is lighter-weight.
- Could have used only accuracy, but AUC/PR/F1 are more informative for imbalanced problems.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                           roc_curve, precision_recall_curve, average_precision_score)
import warnings

class ModelEvaluator:
    """
    Class for comprehensive model evaluation and visualization.
    - Evaluates models using multiple metrics and cross-validation.
    - Generates plots and reports for business/technical communication.
    - Designed for modularity and extensibility.
    """
    
    def __init__(self):
        """Initialize the model evaluator"""
        pass
    
    def evaluate_models(self, X_train, y_train, best_model, model_performance):
        """
        Perform comprehensive model evaluation.
        Why this approach?
        - Uses multiple metrics and plots to fully characterize model performance.
        - Saves results for reproducibility and presentation.
        """
        print("\n📊 MODEL EVALUATION")
        print("=" * 50)
        
        # Create evaluation plots
        self._create_evaluation_plots(X_train, y_train, best_model, model_performance)
        
        # Generate detailed performance report
        self._generate_performance_report(X_train, y_train, best_model, model_performance)
        
        print("✅ Model evaluation completed!")
    
    def _create_evaluation_plots(self, X_train, y_train, best_model, model_performance):
        """
        Create comprehensive evaluation plots (ROC, PR, confusion matrix, model comparison, feature importance).
        Why?
        - Visualizations are essential for understanding and communicating model performance.
        """
        
        # Split data for evaluation
        X_train_eval, X_test_eval, y_train_eval, y_test_eval = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        # Train model on evaluation split
        best_model.fit(X_train_eval, y_train_eval)
        y_pred_proba = best_model.predict_proba(X_test_eval)[:, 1]
        y_pred = best_model.predict(X_test_eval)
        
        # 1. ROC Curve
        self._plot_roc_curve(y_test_eval, y_pred_proba)
        
        # 2. Precision-Recall Curve
        self._plot_precision_recall_curve(y_test_eval, y_pred_proba)
        
        # 3. Confusion Matrix
        self._plot_confusion_matrix(y_test_eval, y_pred)
        
        # 4. Model Comparison
        self._plot_model_comparison(model_performance)
        
        # 5. Feature Importance (if available)
        if hasattr(best_model, 'feature_importances_'):
            self._plot_feature_importance(best_model, X_train.columns)
    
    def _plot_roc_curve(self, y_true, y_pred_proba):
        """
        Plot ROC curve and save to disk.
        Why?
        - ROC AUC is a robust metric for imbalanced classification.
        - Plots are useful for presentations and business communication.
        """
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc_score = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('results/roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 ROC AUC Score: {auc_score:.4f}")
    
    def _plot_precision_recall_curve(self, y_true, y_pred_proba):
        """
        Plot Precision-Recall curve and save to disk.
        Why?
        - PR curves are more informative than ROC for highly imbalanced data.
        """
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        avg_precision = average_precision_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2, 
                label=f'Precision-Recall curve (AP = {avg_precision:.3f})')
        
        # Plot random baseline
        baseline = len(y_true[y_true == 1]) / len(y_true)
        plt.axhline(y=baseline, color='red', linestyle='--', 
                   label=f'Random baseline (AP = {baseline:.3f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="upper right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('results/precision_recall_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Average Precision Score: {avg_precision:.4f}")
    
    def _plot_confusion_matrix(self, y_true, y_pred):
        """
        Plot confusion matrix and print metrics.
        Why?
        - Confusion matrix provides detailed error analysis (TP, FP, TN, FN).
        - Prints accuracy, precision, recall, and F1 for business/technical reporting.
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Not Looking', 'Looking'],
                   yticklabels=['Not Looking', 'Looking'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('results/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Calculate metrics
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"📋 Confusion Matrix Metrics:")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
        print(f"   F1-Score: {f1_score:.4f}")
    
    def _plot_model_comparison(self, model_performance):
        """
        Plot model comparison bar chart.
        Why?
        - Visualizes cross-validation performance for all models.
        - Aids in model selection and justification.
        """
        model_names = list(model_performance.keys())
        cv_means = [model_performance[name]['cv_mean'] for name in model_names]
        cv_stds = [model_performance[name]['cv_std'] for name in model_names]
        
        plt.figure(figsize=(12, 6))
        x_pos = np.arange(len(model_names))
        bars = plt.bar(x_pos, cv_means, yerr=cv_stds, capsize=5, alpha=0.7)
        
        plt.xlabel('Models')
        plt.ylabel('Cross-Validation AUC Score')
        plt.title('Model Performance Comparison')
        plt.xticks(x_pos, model_names, rotation=45, ha='right')
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (bar, mean, std) in enumerate(zip(bars, cv_means, cv_stds)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{mean:.3f}\n±{std:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('results/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_feature_importance(self, model, feature_names):
        """
        Plot feature importance for tree-based models.
        Why?
        - Helps interpret which features drive predictions.
        - Useful for business recommendations.
        """
        if not hasattr(model, 'feature_importances_'):
            return
        
        importance = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        # Plot top 15 features
        top_features = feature_importance_df.head(15)
        
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title('Top 15 Feature Importance')
        plt.gca().invert_yaxis()
        
        # Add value labels
        for i, (bar, importance) in enumerate(zip(bars, top_features['importance'])):
            plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{importance:.4f}', ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('results/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save to CSV
        feature_importance_df.to_csv('results/feature_importance.csv', index=False)
        print("📊 Feature importance analysis completed")
    
    def _generate_performance_report(self, X_train, y_train, best_model, model_performance):
        """
        Generate detailed performance report (CV AUC, best model, dataset size, class distribution, model comparison).
        Why?
        - Summarizes key results for business and technical audiences.
        - Saves to CSV for reproducibility and reporting.
        """
        
        # Cross-validation predictions
        cv_predictions = cross_val_predict(best_model, X_train, y_train, cv=5, method='predict_proba')[:, 1]
        
        # Calculate metrics
        cv_auc = roc_auc_score(y_train, cv_predictions)
        
        # Create performance summary
        performance_summary = {
            'Metric': ['CV AUC Score', 'Best Model', 'Dataset Size', 'Class Distribution'],
            'Value': [
                f"{cv_auc:.4f}",
                type(best_model).__name__,
                f"{len(X_train)} samples",
                f"Class 0: {sum(y_train == 0)} ({sum(y_train == 0)/len(y_train)*100:.1f}%), "
                f"Class 1: {sum(y_train == 1)} ({sum(y_train == 1)/len(y_train)*100:.1f}%)"
            ]
        }
        
        performance_df = pd.DataFrame(performance_summary)
        performance_df.to_csv('results/performance_summary.csv', index=False)
        
        print("\n📋 PERFORMANCE SUMMARY")
        print("=" * 30)
        for _, row in performance_df.iterrows():
            print(f"{row['Metric']}: {row['Value']}")
        
        # Model comparison table
        comparison_data = []
        for name, metrics in model_performance.items():
            comparison_data.append({
                'Model': name,
                'CV AUC Mean': f"{metrics['cv_mean']:.4f}",
                'CV AUC Std': f"{metrics['cv_std']:.4f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_csv('results/model_comparison_table.csv', index=False)
        
        print(f"\n🏆 Best performing model: {comparison_df.loc[comparison_df['CV AUC Mean'].astype(float).idxmax(), 'Model']}") 