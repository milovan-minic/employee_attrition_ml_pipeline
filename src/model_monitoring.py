"""
Model Monitoring Module
----------------------
Handles model performance monitoring, drift detection, and production metrics for the job change prediction pipeline.

Key Responsibilities:
- Monitors model performance (AUC, accuracy, precision, recall, F1) over time.
- Detects data drift using statistical tests (KS test) and feature distribution monitoring.
- Generates monitoring reports and alerts for performance degradation or data drift.
- Tracks production metrics (latency, error rate, data quality issues, uptime).
- Supports automated monitoring configuration and reporting.

Why this approach?
- Modular monitoring: Keeps monitoring logic separate from modeling and evaluation for clarity and maintainability.
- Statistical drift detection: KS test is robust and interpretable for numerical features.
- Production metrics: Essential for real-world ML deployments (latency, errors, retraining triggers).
- Automated reporting: Saves all monitoring results and alerts for audit and compliance.

Alternatives considered:
- Could have used third-party monitoring tools, but custom code is lightweight and fully integrated.
- Could have used only performance metrics, but drift detection is critical for data-driven retraining.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
import joblib
import json
import os
from datetime import datetime, timedelta
import warnings

class ModelMonitor:
    """
    Class for monitoring model performance and detecting drift.
    - Loads models and baseline data, calculates metrics, and detects drift.
    - Generates monitoring reports and alerts for production use.
    - Designed for modularity and extensibility.
    """
    
    def __init__(self, model_path=None, baseline_data=None):
        """Initialize the model monitor"""
        self.model = None
        self.baseline_data = baseline_data
        self.performance_history = []
        self.drift_metrics = {}
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """
        Load a trained model from disk.
        Why?
        - Enables monitoring of any saved model version.
        - Supports production model management and rollback.
        """
        self.model = joblib.load(model_path)
        print(f"✅ Model loaded from: {model_path}")
    
    def calculate_performance_metrics(self, X, y_true, y_pred_proba=None):
        """
        Calculate comprehensive performance metrics (AUC, accuracy, precision, recall, F1).
        Why?
        - Tracks model quality in production and triggers alerts if performance drops.
        - Supports business/technical reporting.
        """
        if y_pred_proba is None and self.model:
            y_pred_proba = self.model.predict_proba(X)[:, 1]
        
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        metrics = {
            'auc': roc_auc_score(y_true, y_pred_proba),
            'accuracy': (y_true == y_pred).mean(),
            'precision': None,
            'recall': None,
            'f1_score': None,
            'timestamp': datetime.now().isoformat()
        }
        
        # Calculate additional metrics
        from sklearn.metrics import precision_score, recall_score, f1_score
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
        
        return metrics
    
    def detect_data_drift(self, current_data, baseline_data=None):
        """
        Detect data drift between current and baseline data using KS test for numerical features.
        Why?
        - Data drift is a leading cause of model degradation in production.
        - KS test is robust and interpretable for drift detection.
        """
        if baseline_data is None:
            baseline_data = self.baseline_data
        
        if baseline_data is None:
            print("⚠️  No baseline data available for drift detection")
            return {}
        
        drift_metrics = {}
        
        # Statistical drift detection
        for column in current_data.columns:
            if column in baseline_data.columns:
                # KS test for numerical columns
                if current_data[column].dtype in ['int64', 'float64']:
                    from scipy.stats import ks_2samp
                    try:
                        stat, p_value = ks_2samp(
                            baseline_data[column].dropna(),
                            current_data[column].dropna()
                        )
                        drift_metrics[column] = {
                            'ks_statistic': stat,
                            'p_value': p_value,
                            'drift_detected': p_value < 0.05
                        }
                    except:
                        drift_metrics[column] = {'error': 'Could not compute KS test'}
        
        self.drift_metrics = drift_metrics
        return drift_metrics
    
    def monitor_feature_distribution(self, current_data, baseline_data=None):
        """
        Monitor feature distribution changes (mean/std) between current and baseline data.
        Why?
        - Feature distribution monitoring can catch subtle data shifts before they impact performance.
        """
        if baseline_data is None:
            baseline_data = self.baseline_data
        
        if baseline_data is None:
            return {}
        
        distribution_changes = {}
        
        for column in current_data.columns:
            if column in baseline_data.columns:
                if current_data[column].dtype in ['int64', 'float64']:
                    # Compare means and standard deviations
                    baseline_mean = baseline_data[column].mean()
                    current_mean = current_data[column].mean()
                    baseline_std = baseline_data[column].std()
                    current_std = current_data[column].std()
                    
                    distribution_changes[column] = {
                        'mean_change': (current_mean - baseline_mean) / baseline_mean if baseline_mean != 0 else 0,
                        'std_change': (current_std - baseline_std) / baseline_std if baseline_std != 0 else 0,
                        'baseline_mean': baseline_mean,
                        'current_mean': current_mean,
                        'baseline_std': baseline_std,
                        'current_std': current_std
                    }
        
        return distribution_changes
    
    def generate_monitoring_report(self, X_test, y_test, drift_data=None):
        """
        Generate comprehensive monitoring report with performance metrics, drift metrics, and alerts.
        Why?
        - Automated reporting is essential for audit, compliance, and production monitoring.
        - Alerts enable proactive model management.
        """
        print("📊 Generating Model Monitoring Report...")
        
        # Performance metrics
        performance_metrics = self.calculate_performance_metrics(X_test, y_test)
        
        # Drift detection
        if drift_data is not None:
            drift_metrics = self.detect_data_drift(drift_data)
        else:
            drift_metrics = {}
        
        # Create monitoring report
        report = {
            'timestamp': datetime.now().isoformat(),
            'performance_metrics': performance_metrics,
            'drift_metrics': drift_metrics,
            'alerts': []
        }
        
        # Generate alerts
        alerts = []
        
        # Performance alerts
        if performance_metrics['auc'] < 0.7:
            alerts.append({
                'type': 'performance_degradation',
                'severity': 'high',
                'message': f"Model AUC below threshold: {performance_metrics['auc']:.4f}"
            })
        
        # Drift alerts
        for feature, drift_info in drift_metrics.items():
            if isinstance(drift_info, dict) and drift_info.get('drift_detected', False):
                alerts.append({
                    'type': 'data_drift',
                    'severity': 'medium',
                    'message': f"Data drift detected in feature: {feature}"
                })
        
        report['alerts'] = alerts
        
        # Save monitoring report
        os.makedirs('results', exist_ok=True)
        with open('results/monitoring_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print(f"\n📋 MONITORING SUMMARY:")
        print(f"   AUC Score: {performance_metrics['auc']:.4f}")
        print(f"   Accuracy: {performance_metrics['accuracy']:.4f}")
        print(f"   Precision: {performance_metrics['precision']:.4f}")
        print(f"   Recall: {performance_metrics['recall']:.4f}")
        print(f"   F1 Score: {performance_metrics['f1_score']:.4f}")
        print(f"   Alerts Generated: {len(alerts)}")
        
        return report
    
    def plot_performance_trends(self, performance_history):
        """
        Plot performance trends (AUC, accuracy, F1, precision/recall) over time.
        Why?
        - Visualizes model health and supports retraining decisions.
        - Saves plots for reporting and compliance.
        """
        if not performance_history:
            print("No performance history available")
            return
        
        df = pd.DataFrame(performance_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # AUC trend
        axes[0, 0].plot(df['timestamp'], df['auc'], marker='o')
        axes[0, 0].set_title('AUC Score Trend')
        axes[0, 0].set_ylabel('AUC')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Accuracy trend
        axes[0, 1].plot(df['timestamp'], df['accuracy'], marker='o', color='orange')
        axes[0, 1].set_title('Accuracy Trend')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # F1 Score trend
        axes[1, 0].plot(df['timestamp'], df['f1_score'], marker='o', color='green')
        axes[1, 0].set_title('F1 Score Trend')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Precision vs Recall
        axes[1, 1].scatter(df['precision'], df['recall'], alpha=0.7)
        axes[1, 1].set_title('Precision vs Recall')
        axes[1, 1].set_xlabel('Precision')
        axes[1, 1].set_ylabel('Recall')
        
        plt.tight_layout()
        plt.savefig('results/performance_trends.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Performance trends plot saved to: results/performance_trends.png")
    
    def setup_automated_monitoring(self, model_path, baseline_data_path, monitoring_schedule='daily'):
        """
        Setup automated monitoring pipeline (loads model, baseline, saves config).
        Why?
        - Automates monitoring setup for production deployments.
        - Saves configuration for reproducibility and audit.
        """
        print("🔧 Setting up automated monitoring...")
        
        # Load model and baseline data
        self.load_model(model_path)
        
        if baseline_data_path and os.path.exists(baseline_data_path):
            self.baseline_data = pd.read_csv(baseline_data_path)
            print(f"✅ Baseline data loaded from: {baseline_data_path}")
        
        # Create monitoring configuration
        config = {
            'model_path': model_path,
            'baseline_data_path': baseline_data_path,
            'monitoring_schedule': monitoring_schedule,
            'performance_thresholds': {
                'auc_min': 0.7,
                'accuracy_min': 0.8,
                'f1_min': 0.6
            },
            'drift_thresholds': {
                'ks_p_value': 0.05,
                'mean_change_threshold': 0.1,
                'std_change_threshold': 0.2
            }
        }
        
        # Save monitoring configuration
        with open('models/monitoring_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print("✅ Automated monitoring configured")
        print(f"   Monitoring schedule: {monitoring_schedule}")
        print(f"   Performance thresholds: {config['performance_thresholds']}")
        print(f"   Drift thresholds: {config['drift_thresholds']}")
        
        return config


class ProductionMetrics:
    """
    Class for tracking production-specific metrics (latency, errors, data quality, uptime).
    Why?
    - Production metrics are essential for real-world ML deployments and SLA compliance.
    - Enables tracking and reporting of model health in production.
    """
    
    def __init__(self):
        self.metrics = {
            'predictions_made': 0,
            'average_prediction_time': 0,
            'model_errors': 0,
            'data_quality_issues': 0,
            'last_retraining': None,
            'model_uptime': 0
        }
    
    def update_metrics(self, prediction_time=None, error_occurred=False, data_quality_issue=False):
        """
        Update production metrics (prediction count, latency, errors, data quality).
        Why?
        - Tracks key metrics for monitoring and alerting.
        - Supports business/technical reporting and SLA compliance.
        """
        self.metrics['predictions_made'] += 1
        
        if prediction_time:
            # Update average prediction time
            current_avg = self.metrics['average_prediction_time']
            total_predictions = self.metrics['predictions_made']
            self.metrics['average_prediction_time'] = (
                (current_avg * (total_predictions - 1) + prediction_time) / total_predictions
            )
        
        if error_occurred:
            self.metrics['model_errors'] += 1
        
        if data_quality_issue:
            self.metrics['data_quality_issues'] += 1
    
    def get_metrics_summary(self):
        """
        Get production metrics summary (totals, averages, error rates).
        Why?
        - Summarizes key metrics for reporting and dashboarding.
        """
        return {
            'total_predictions': self.metrics['predictions_made'],
            'average_prediction_time_ms': self.metrics['average_prediction_time'] * 1000,
            'error_rate': self.metrics['model_errors'] / max(self.metrics['predictions_made'], 1),
            'data_quality_issue_rate': self.metrics['data_quality_issues'] / max(self.metrics['predictions_made'], 1),
            'last_retraining': self.metrics['last_retraining'],
            'model_uptime_hours': self.metrics['model_uptime']
        }
    
    def save_metrics(self, filepath='results/production_metrics.json'):
        """
        Save production metrics to file for reporting and audit.
        Why?
        - Ensures all metrics are persisted for compliance and troubleshooting.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        metrics_data = {
            'timestamp': datetime.now().isoformat(),
            'metrics': self.metrics,
            'summary': self.get_metrics_summary()
        }
        
        with open(filepath, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        print(f"✅ Production metrics saved to: {filepath}") 