import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, precision_recall_curve, roc_auc_score, confusion_matrix, accuracy_score, precision_score, recall_score
from xgboost import XGBClassifier
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class FraudDetectionModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_cols = None
        self.X_test = None
        self.y_test = None
        self.y_proba = None
        self.y_pred = None
        self.optimal_threshold = None
        self.evaluation_results = {}
        self.suspicious_transactions = None
        self.merchant_fraud_rates = {}  # Store merchant fraud rates from training
        
    def preprocess_data(self, df, is_training=False):
        """Feature engineering and preprocessing
        
        Args:
            df: DataFrame with transaction data
            is_training: If True, compute merchant fraud rates from data. 
                        If False, use stored rates from training (for inference)
        """
        df = df.copy()
        
        # Convert timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        # Feature Engineering
        df['hour'] = df['timestamp'].dt.hour
        df['is_night'] = df['hour'].apply(lambda x: 1 if x < 6 or x > 20 else 0)
        df['is_weekend'] = df['timestamp'].dt.weekday.apply(lambda x: 1 if x >= 5 else 0)
        df['log_amount'] = np.log1p(df['amount'])
        
        # User spending patterns
        df['avg_user_spend'] = df.groupby('user_id')['amount'].transform('mean')
        df['user_spend_dev'] = df['amount'] - df['avg_user_spend']
        
        # Merchant fraud rate
        if is_training and 'is_fraud' in df.columns:
            # Training mode: compute fraud rates from data and store them
            fraud_rate_map = df.groupby('merchant_id')['is_fraud'].mean()
            self.merchant_fraud_rates = fraud_rate_map.to_dict()
            df['merchant_fraud_rate'] = df['merchant_id'].map(fraud_rate_map).fillna(0)
        else:
            # Inference mode: use stored rates from training
            if self.merchant_fraud_rates:
                df['merchant_fraud_rate'] = df['merchant_id'].map(self.merchant_fraud_rates).fillna(0.1)
            else:
                # Fallback if no training data available
                df['merchant_fraud_rate'] = 0.1
        
        # Transaction velocity
        df = df.sort_values(['user_id', 'timestamp'])
        df['user_txn_gap'] = df.groupby('user_id')['timestamp'].diff().dt.total_seconds().fillna(3600)
        df['velocity_1h'] = (df['user_txn_gap'] < 3600).astype(int)
        
        # Anomaly detection
        iso = IsolationForest(random_state=42, n_jobs=-1)
        df['anomaly_score'] = iso.fit_predict(df[['log_amount', 'user_spend_dev']])
        
        # One-hot encode transaction type
        df = pd.get_dummies(df, columns=['transaction_type'])
        
        return df
    
    def train(self, df, custom_params=None):
        """Train the fraud detection model"""
        # Preprocess data in training mode
        processed_df = self.preprocess_data(df, is_training=True)
        
        # Define features
        self.feature_cols = [
            'log_amount', 'user_spend_dev', 'hour', 'is_night', 'is_weekend',
            'merchant_fraud_rate', 'velocity_1h', 'anomaly_score'
        ] + [col for col in processed_df.columns if col.startswith('transaction_type_')]
        
        X = processed_df[self.feature_cols]
        y = processed_df['is_fraud']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, test_size=0.2, random_state=42
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Store test data for later use
        self.X_test = X_test_scaled
        self.y_test = y_test
        
        # Train XGBoost model
        scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
        
        # Default parameters
        default_params = {
            'n_estimators': 150,
            'max_depth': 5,
            'learning_rate': 0.05,
            'gamma': 0.2,
            'min_child_weight': 5,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'precision_range': (0.4, 0.6)
        }
        
        # Use custom parameters if provided
        if custom_params is not None:
            params = {**default_params, **custom_params}
        else:
            params = default_params
        
        # Extract precision range
        desired_precision_range = params.pop('precision_range', (0.4, 0.6))
        
        self.model = XGBClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            learning_rate=params['learning_rate'],
            gamma=params['gamma'],
            min_child_weight=params['min_child_weight'],
            subsample=params['subsample'],
            colsample_bytree=params['colsample_bytree'],
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
            n_jobs=-1
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions and find optimal threshold
        self.y_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Find optimal threshold
        precision, recall, thresholds = precision_recall_curve(y_test, self.y_proba)
        
        valid_indices = np.where((precision[:-1] >= desired_precision_range[0]) &
                                (precision[:-1] <= desired_precision_range[1]))[0]
        
        if len(valid_indices) == 0:
            self.optimal_threshold = 0.5
        else:
            f1_scores = 2 * (precision[valid_indices] * recall[valid_indices]) / \
                       (precision[valid_indices] + recall[valid_indices] + 1e-8)
            best_index = valid_indices[np.argmax(f1_scores)]
            self.optimal_threshold = thresholds[best_index]
        
        self.y_pred = (self.y_proba >= self.optimal_threshold).astype(int)
        
        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, self.y_pred)
        roc_auc = roc_auc_score(y_test, self.y_proba)
        precision = precision_score(y_test, self.y_pred)
        recall = recall_score(y_test, self.y_pred)
        class_report = classification_report(y_test, self.y_pred, digits=4)
        
        self.evaluation_results = {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'precision': precision,
            'recall': recall,
            'classification_report': class_report,
            'optimal_threshold': self.optimal_threshold
        }
        
        # Create suspicious transactions dataframe
        suspect_transactions = processed_df.iloc[y_test.index].copy()
        suspect_transactions['fraud_proba'] = self.y_proba
        suspect_transactions['fraud_pred'] = self.y_pred
        
        self.suspicious_transactions = suspect_transactions[
            suspect_transactions['fraud_pred'] == 1
        ].sort_values(by='fraud_proba', ascending=False)
        
        return self.evaluation_results
    
    def predict_single_transaction(self, transaction_data):
        """Predict fraud for a single transaction"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Create a dataframe with the transaction
        df = pd.DataFrame([transaction_data])
        
        # Add required columns for feature engineering
        if 'is_fraud' not in df.columns:
            df['is_fraud'] = 0
        
        # Create dummy data for preprocessing (merchant fraud rate calculation)
        # This is a simplification - in production, you'd have historical data
        df['merchant_fraud_rate'] = 0.1  # Default merchant fraud rate
        
        # Basic feature engineering for single prediction
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['is_night'] = df['hour'].apply(lambda x: 1 if x < 6 or x > 20 else 0)
        df['is_weekend'] = df['timestamp'].dt.weekday.apply(lambda x: 1 if x >= 5 else 0)
        df['log_amount'] = np.log1p(df['amount'])
        df['avg_user_spend'] = df['amount']  # Simplified for single prediction
        df['user_spend_dev'] = 0  # Simplified
        df['velocity_1h'] = 0  # Simplified
        df['anomaly_score'] = 1  # Normal transaction assumption
        
        # One-hot encode transaction type
        transaction_type_cols = ['transaction_type_atm', 'transaction_type_pos', 
                               'transaction_type_upi', 'transaction_type_wallet', 
                               'transaction_type_web']
        
        for col in transaction_type_cols:
            df[col] = 0
        
        # Set the appropriate transaction type
        if f"transaction_type_{transaction_data['transaction_type']}" in transaction_type_cols:
            df[f"transaction_type_{transaction_data['transaction_type']}"] = 1
        
        # Select features
        X = df[self.feature_cols]
        
        # Handle missing columns
        for col in self.feature_cols:
            if col not in X.columns:
                X[col] = 0
        
        # Scale features
        X_scaled = self.scaler.transform(X[self.feature_cols])
        
        # Make prediction
        fraud_proba = self.model.predict_proba(X_scaled)[0, 1]
        is_fraud = fraud_proba >= self.optimal_threshold
        
        return {
            'fraud_probability': fraud_proba,
            'is_fraud': is_fraud,
            'threshold_used': self.optimal_threshold
        }
    
    def get_evaluation_results(self):
        """Get model evaluation results"""
        return self.evaluation_results
    
    def get_suspicious_transactions(self):
        """Get suspicious transactions dataframe"""
        return self.suspicious_transactions
    
    def plot_confusion_matrix(self):
        """Create interactive confusion matrix plot"""
        cm = confusion_matrix(self.y_test, self.y_pred)
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Not Fraud', 'Fraud'],
            y=['Not Fraud', 'Fraud'],
            colorscale='Blues',
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 16},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Confusion Matrix",
            xaxis_title="Predicted",
            yaxis_title="Actual",
            height=400,
            width=500
        )
        
        return fig
    
    def plot_feature_importance(self):
        """Create interactive feature importance plot"""
        importances = self.model.feature_importances_
        feat_imp = pd.DataFrame({
            'feature': self.feature_cols, 
            'importance': importances
        }).sort_values('importance', ascending=True)
        
        fig = go.Figure(go.Bar(
            x=feat_imp['importance'],
            y=feat_imp['feature'],
            orientation='h'
        ))
        
        fig.update_layout(
            title="Feature Importance (XGBoost)",
            xaxis_title="Importance",
            yaxis_title="Features",
            height=600
        )
        
        return fig
    
    def plot_precision_recall_curve(self):
        """Create interactive precision-recall curve"""
        precision, recall, thresholds = precision_recall_curve(self.y_test, self.y_proba)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=thresholds,
            y=precision[:-1],
            mode='lines',
            name='Precision',
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            x=thresholds,
            y=recall[:-1],
            mode='lines',
            name='Recall',
            line=dict(color='red')
        ))
        
        # Add threshold lines
        fig.add_hline(y=0.4, line_dash="dash", line_color="red", 
                     annotation_text="Precision Lower Bound")
        fig.add_hline(y=0.6, line_dash="dash", line_color="green", 
                     annotation_text="Precision Upper Bound")
        fig.add_vline(x=self.optimal_threshold, line_dash="dash", line_color="purple",
                     annotation_text=f"Optimal Threshold: {self.optimal_threshold:.3f}")
        
        fig.update_layout(
            title="Precision-Recall Curve",
            xaxis_title="Threshold",
            yaxis_title="Score",
            height=500
        )
        
        return fig
