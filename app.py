import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from fraud_model import FraudDetectionModel
from utils import generate_sample_transaction
import io

# Page configuration
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'data' not in st.session_state:
    st.session_state.data = None
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

st.title("🚨 Fraud Detection Dashboard")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("Navigation")
    page = st.selectbox(
        "Select Page",
        ["Data Upload & Model Training", "Model Performance", "Fraud Prediction", "Suspicious Transactions", "Time-Series Analysis", "Batch Prediction", "Analytics & Reports", "Model Configuration"]
    )

# Data Upload & Model Training Page
if page == "Data Upload & Model Training":
    st.header("📊 Data Upload & Model Training")
    
    # Data upload options
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Upload Your Data")
        uploaded_file = st.file_uploader(
            "Choose a CSV file", 
            type="csv",
            help="Upload a CSV file with transaction data"
        )
    
    with col2:
        st.subheader("Or Use Sample Data")
        if st.button("Load Sample Dataset", type="primary"):
            try:
                # Try to load the provided sample data
                sample_data_path = "attached_assets/synthetic_transactions (1)_1760767859956.csv"
                st.session_state.data = pd.read_csv(sample_data_path)
                st.success(f"Sample dataset loaded! Shape: {st.session_state.data.shape}")
            except FileNotFoundError:
                st.error("Sample dataset not found. Please upload your own CSV file.")
    
    # Process uploaded file
    if uploaded_file is not None:
        try:
            st.session_state.data = pd.read_csv(uploaded_file)
            st.success(f"Data uploaded successfully! Shape: {st.session_state.data.shape}")
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
    
    # Display data info if available
    if st.session_state.data is not None:
        st.subheader("Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Transactions", len(st.session_state.data))
        with col2:
            fraud_count = st.session_state.data['is_fraud'].sum() if 'is_fraud' in st.session_state.data.columns else 0
            st.metric("Fraud Transactions", fraud_count)
        with col3:
            fraud_rate = (fraud_count / len(st.session_state.data) * 100) if len(st.session_state.data) > 0 else 0
            st.metric("Fraud Rate", f"{fraud_rate:.2f}%")
        with col4:
            total_amount = st.session_state.data['amount'].sum() if 'amount' in st.session_state.data.columns else 0
            st.metric("Total Amount", f"${total_amount:,.2f}")
        
        # Show sample data
        st.subheader("Sample Data")
        st.dataframe(st.session_state.data.head(), use_container_width=True)
        
        # Train model button
        st.subheader("Model Training")
        if st.button("Train Fraud Detection Model", type="primary"):
            with st.spinner("Training model... This may take a few minutes."):
                try:
                    st.session_state.model = FraudDetectionModel()
                    training_results = st.session_state.model.train(st.session_state.data)
                    st.session_state.model_trained = True
                    
                    st.success("Model trained successfully!")
                    
                    # Display training results
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Accuracy", f"{training_results['accuracy']:.4f}")
                    with col2:
                        st.metric("ROC-AUC", f"{training_results['roc_auc']:.4f}")
                    with col3:
                        st.metric("Optimal Threshold", f"{training_results['optimal_threshold']:.3f}")
                        
                except Exception as e:
                    st.error(f"Error training model: {str(e)}")
                    st.exception(e)

# Model Performance Page
elif page == "Model Performance":
    st.header("📈 Model Performance Metrics")
    
    if not st.session_state.model_trained:
        st.warning("Please train the model first in the 'Data Upload & Model Training' page.")
    else:
        # Performance metrics
        results = st.session_state.model.get_evaluation_results()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{results['accuracy']:.4f}")
        with col2:
            st.metric("ROC-AUC Score", f"{results['roc_auc']:.4f}")
        with col3:
            st.metric("Precision", f"{results['precision']:.4f}")
        with col4:
            st.metric("Recall", f"{results['recall']:.4f}")
        
        st.markdown("---")
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Confusion Matrix")
            cm_fig = st.session_state.model.plot_confusion_matrix()
            st.plotly_chart(cm_fig, use_container_width=True)
        
        with col2:
            st.subheader("Feature Importance")
            fi_fig = st.session_state.model.plot_feature_importance()
            st.plotly_chart(fi_fig, use_container_width=True)
        
        # Precision-Recall Curve
        st.subheader("Precision-Recall Curve")
        pr_fig = st.session_state.model.plot_precision_recall_curve()
        st.plotly_chart(pr_fig, use_container_width=True)
        
        # Classification Report
        st.subheader("Detailed Classification Report")
        st.text(results['classification_report'])

# Fraud Prediction Page
elif page == "Fraud Prediction":
    st.header("🔍 Single Transaction Fraud Prediction")
    
    if not st.session_state.model_trained:
        st.warning("Please train the model first in the 'Data Upload & Model Training' page.")
    else:
        st.subheader("Enter Transaction Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            transaction_id = st.text_input("Transaction ID", value=generate_sample_transaction()['transaction_id'])
            user_id = st.number_input("User ID", min_value=1, max_value=10000, value=1234)
            merchant_id = st.number_input("Merchant ID", min_value=1, max_value=10000, value=5678)
            amount = st.number_input("Transaction Amount", min_value=0.01, max_value=100000.0, value=100.0, step=0.01)
            transaction_type = st.selectbox("Transaction Type", ["pos", "atm", "web", "upi", "wallet"])
        
        with col2:
            device_type = st.selectbox("Device Type", ["mobile", "web", "pos", "atm"])
            location = st.selectbox("Location", ["Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad"])
            
            # Date and time input
            col_date, col_time = st.columns(2)
            with col_date:
                transaction_date = st.date_input("Transaction Date")
            with col_time:
                transaction_time = st.time_input("Transaction Time")
            
            # Combine date and time into timestamp
            from datetime import datetime
            timestamp = datetime.combine(transaction_date, transaction_time)
            
            account_number = st.text_input("Account Number", value=generate_sample_transaction()['account_number'])
        
        if st.button("Predict Fraud", type="primary"):
            # Create transaction data
            transaction_data = {
                'transaction_id': transaction_id,
                'account_number': account_number,
                'user_id': user_id,
                'merchant_id': merchant_id,
                'transaction_type': transaction_type,
                'amount': amount,
                'device_type': device_type,
                'location': location,
                'timestamp': timestamp,
                'is_fraud': 0  # placeholder
            }
            
            try:
                prediction_result = st.session_state.model.predict_single_transaction(transaction_data)
                
                st.markdown("---")
                st.subheader("Prediction Results")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    fraud_prob = prediction_result['fraud_probability']
                    st.metric("Fraud Probability", f"{fraud_prob:.4f}")
                
                with col2:
                    prediction = "FRAUD" if prediction_result['is_fraud'] else "LEGITIMATE"
                    color = "🔴" if prediction_result['is_fraud'] else "🟢"
                    st.metric("Prediction", f"{color} {prediction}")
                
                with col3:
                    confidence = max(fraud_prob, 1 - fraud_prob)
                    st.metric("Confidence", f"{confidence:.4f}")
                
                # Risk level indicator
                if fraud_prob > 0.7:
                    st.error("🚨 HIGH RISK TRANSACTION")
                elif fraud_prob > 0.5:
                    st.warning("⚠️ MEDIUM RISK TRANSACTION")
                else:
                    st.success("✅ LOW RISK TRANSACTION")
                    
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")

# Suspicious Transactions Page
elif page == "Suspicious Transactions":
    st.header("📋 Suspicious Transactions Analysis")
    
    if not st.session_state.model_trained:
        st.warning("Please train the model first in the 'Data Upload & Model Training' page.")
    else:
        suspicious_transactions = st.session_state.model.get_suspicious_transactions()
        
        if len(suspicious_transactions) > 0:
            # Summary statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Suspicious Transactions", len(suspicious_transactions))
            with col2:
                total_fraud_amount = suspicious_transactions['amount'].sum()
                st.metric("Total Suspected Fraud Amount", f"${total_fraud_amount:,.2f}")
            with col3:
                avg_fraud_prob = suspicious_transactions['fraud_proba'].mean()
                st.metric("Average Fraud Probability", f"{avg_fraud_prob:.4f}")
            with col4:
                max_fraud_prob = suspicious_transactions['fraud_proba'].max()
                st.metric("Highest Fraud Probability", f"{max_fraud_prob:.4f}")
            
            st.markdown("---")
            
            # Filters
            st.subheader("Filter Suspicious Transactions")
            col1, col2 = st.columns(2)
            
            with col1:
                min_prob = st.slider(
                    "Minimum Fraud Probability", 
                    min_value=0.0, 
                    max_value=1.0, 
                    value=0.5,
                    step=0.01
                )
            
            with col2:
                top_n = st.number_input(
                    "Show Top N Transactions", 
                    min_value=1, 
                    max_value=len(suspicious_transactions), 
                    value=min(20, len(suspicious_transactions))
                )
            
            # Filter data
            filtered_transactions = suspicious_transactions[
                suspicious_transactions['fraud_proba'] >= min_prob
            ].head(top_n)
            
            if len(filtered_transactions) > 0:
                st.subheader(f"Top {len(filtered_transactions)} Suspicious Transactions")
                
                # Display table
                display_columns = ['transaction_id', 'user_id', 'merchant_id', 'amount', 'timestamp', 'fraud_proba']
                st.dataframe(
                    filtered_transactions[display_columns].round(4),
                    use_container_width=True
                )
                
                # Visualization
                st.subheader("Fraud Probability Distribution")
                fig = px.histogram(
                    filtered_transactions, 
                    x='fraud_proba', 
                    bins=20,
                    title="Distribution of Fraud Probabilities",
                    labels={'fraud_proba': 'Fraud Probability', 'count': 'Number of Transactions'}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Amount vs Probability scatter plot
                st.subheader("Transaction Amount vs Fraud Probability")
                fig2 = px.scatter(
                    filtered_transactions,
                    x='amount',
                    y='fraud_proba',
                    title="Transaction Amount vs Fraud Probability",
                    labels={'amount': 'Transaction Amount ($)', 'fraud_proba': 'Fraud Probability'}
                )
                st.plotly_chart(fig2, use_container_width=True)
                
            else:
                st.info("No transactions meet the selected criteria.")
        else:
            st.info("No suspicious transactions found.")

# Time-Series Analysis Page
elif page == "Time-Series Analysis":
    st.header("📈 Historical Transaction Analysis")
    
    if not st.session_state.model_trained:
        st.warning("Please train the model first in the 'Data Upload & Model Training' page.")
    else:
        df = st.session_state.data.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Summary metrics
        st.subheader("Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            date_range = f"{df['timestamp'].min().date()} to {df['timestamp'].max().date()}"
            st.metric("Date Range", date_range)
        with col2:
            total_frauds = df['is_fraud'].sum()
            st.metric("Total Frauds", total_frauds)
        with col3:
            fraud_rate = (total_frauds / len(df) * 100)
            st.metric("Fraud Rate", f"{fraud_rate:.2f}%")
        with col4:
            fraud_amount = df[df['is_fraud'] == 1]['amount'].sum()
            st.metric("Fraud Amount", f"${fraud_amount:,.2f}")
        
        st.markdown("---")
        
        # Fraud trends over time
        st.subheader("Fraud Trends Over Time")
        
        # Time grouping options
        time_group = st.selectbox("Group by", ["Hour", "Day", "Week", "Month"])
        
        if time_group == "Hour":
            df['time_group'] = df['timestamp'].dt.hour
            x_label = "Hour of Day"
        elif time_group == "Day":
            df['time_group'] = df['timestamp'].dt.date
            x_label = "Date"
        elif time_group == "Week":
            df['time_group'] = df['timestamp'].dt.to_period('W').astype(str)
            x_label = "Week"
        else:  # Month
            df['time_group'] = df['timestamp'].dt.to_period('M').astype(str)
            x_label = "Month"
        
        # Calculate fraud metrics by time group
        fraud_by_time = df.groupby('time_group').agg({
            'is_fraud': ['sum', 'count', 'mean'],
            'amount': 'sum'
        }).reset_index()
        
        fraud_by_time.columns = ['time_group', 'fraud_count', 'total_transactions', 'fraud_rate', 'total_amount']
        fraud_by_time['fraud_rate'] = fraud_by_time['fraud_rate'] * 100
        
        # Line chart for fraud count and rate
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = px.line(
                fraud_by_time, 
                x='time_group', 
                y='fraud_count',
                title=f'Fraud Count by {time_group}',
                labels={'time_group': x_label, 'fraud_count': 'Number of Frauds'},
                markers=True
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            fig2 = px.line(
                fraud_by_time, 
                x='time_group', 
                y='fraud_rate',
                title=f'Fraud Rate by {time_group}',
                labels={'time_group': x_label, 'fraud_rate': 'Fraud Rate (%)'},
                markers=True
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        # Heatmap for hour and day of week patterns
        st.subheader("Fraud Patterns by Hour and Day of Week")
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.day_name()
        
        fraud_heatmap = df.groupby(['day_of_week', 'hour'])['is_fraud'].mean().reset_index()
        fraud_heatmap_pivot = fraud_heatmap.pivot(index='day_of_week', columns='hour', values='is_fraud')
        
        # Reorder days
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        fraud_heatmap_pivot = fraud_heatmap_pivot.reindex(day_order)
        
        fig3 = go.Figure(data=go.Heatmap(
            z=fraud_heatmap_pivot.values * 100,
            x=fraud_heatmap_pivot.columns,
            y=fraud_heatmap_pivot.index,
            colorscale='Reds',
            text=np.round(fraud_heatmap_pivot.values * 100, 2),
            texttemplate='%{text}%',
            textfont={"size": 10},
            colorbar=dict(title="Fraud Rate (%)")
        ))
        
        fig3.update_layout(
            title="Fraud Rate Heatmap (Day of Week vs Hour)",
            xaxis_title="Hour of Day",
            yaxis_title="Day of Week",
            height=400
        )
        
        st.plotly_chart(fig3, use_container_width=True)
        
        # Transaction amount distribution
        st.subheader("Transaction Amount Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig4 = px.histogram(
                df,
                x='amount',
                color='is_fraud',
                nbins=50,
                title='Transaction Amount Distribution',
                labels={'amount': 'Transaction Amount ($)', 'is_fraud': 'Fraud'},
                color_discrete_map={0: 'blue', 1: 'red'}
            )
            st.plotly_chart(fig4, use_container_width=True)
        
        with col2:
            fig5 = px.box(
                df,
                x='is_fraud',
                y='amount',
                title='Transaction Amount by Fraud Status',
                labels={'is_fraud': 'Fraud', 'amount': 'Transaction Amount ($)'},
                color='is_fraud',
                color_discrete_map={0: 'blue', 1: 'red'}
            )
            st.plotly_chart(fig5, use_container_width=True)

# Batch Prediction Page
elif page == "Batch Prediction":
    st.header("📤 Batch Transaction Fraud Detection")
    
    if not st.session_state.model_trained:
        st.warning("Please train the model first in the 'Data Upload & Model Training' page.")
    else:
        st.subheader("Upload Transactions for Batch Prediction")
        
        batch_file = st.file_uploader(
            "Upload CSV file with transactions",
            type="csv",
            help="Upload a CSV file with transaction data to get fraud predictions for all transactions"
        )
        
        if batch_file is not None:
            try:
                batch_df = pd.read_csv(batch_file)
                st.success(f"File uploaded successfully! Total transactions: {len(batch_df)}")
                
                # Show preview
                st.subheader("Data Preview")
                st.dataframe(batch_df.head(10), use_container_width=True)
                
                if st.button("Run Batch Prediction", type="primary"):
                    with st.spinner("Running predictions..."):
                        # Preprocess the data in inference mode (no is_fraud column required)
                        processed_df = st.session_state.model.preprocess_data(batch_df, is_training=False)
                        
                        # Get features
                        X_batch = processed_df[st.session_state.model.feature_cols]
                        X_batch_scaled = st.session_state.model.scaler.transform(X_batch)
                        
                        # Make predictions
                        fraud_probas = st.session_state.model.model.predict_proba(X_batch_scaled)[:, 1]
                        fraud_preds = (fraud_probas >= st.session_state.model.optimal_threshold).astype(int)
                        
                        # Add predictions to original dataframe
                        batch_df['fraud_probability'] = fraud_probas
                        batch_df['fraud_prediction'] = fraud_preds
                        batch_df['risk_level'] = pd.cut(
                            batch_df['fraud_probability'],
                            bins=[0, 0.3, 0.5, 0.7, 1.0],
                            labels=['Low', 'Medium', 'High', 'Critical']
                        )
                        
                        st.success("Predictions completed!")
                        
                        # Summary metrics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Total Transactions", len(batch_df))
                        with col2:
                            fraud_count = fraud_preds.sum()
                            st.metric("Predicted Frauds", fraud_count)
                        with col3:
                            fraud_rate = (fraud_count / len(batch_df) * 100)
                            st.metric("Fraud Rate", f"{fraud_rate:.2f}%")
                        with col4:
                            if fraud_count > 0:
                                fraud_amount = batch_df[batch_df['fraud_prediction'] == 1]['amount'].sum()
                                st.metric("Suspected Fraud Amount", f"${fraud_amount:,.2f}")
                        
                        st.markdown("---")
                        
                        # Show results
                        st.subheader("Prediction Results")
                        
                        # Filter options
                        filter_option = st.radio(
                            "Show",
                            ["All Transactions", "Fraudulent Only", "Legitimate Only"],
                            horizontal=True
                        )
                        
                        if filter_option == "Fraudulent Only":
                            display_df = batch_df[batch_df['fraud_prediction'] == 1]
                        elif filter_option == "Legitimate Only":
                            display_df = batch_df[batch_df['fraud_prediction'] == 0]
                        else:
                            display_df = batch_df
                        
                        st.dataframe(
                            display_df.sort_values('fraud_probability', ascending=False),
                            use_container_width=True
                        )
                        
                        # Download results
                        st.subheader("Download Results")
                        
                        csv = batch_df.to_csv(index=False)
                        st.download_button(
                            label="Download Predictions as CSV",
                            data=csv,
                            file_name="fraud_predictions.csv",
                            mime="text/csv"
                        )
                        
                        # Visualizations
                        st.subheader("Risk Distribution")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            risk_counts = batch_df['risk_level'].value_counts()
                            fig1 = px.pie(
                                values=risk_counts.values,
                                names=risk_counts.index,
                                title="Transactions by Risk Level",
                                color_discrete_sequence=['green', 'yellow', 'orange', 'red']
                            )
                            st.plotly_chart(fig1, use_container_width=True)
                        
                        with col2:
                            fig2 = px.histogram(
                                batch_df,
                                x='fraud_probability',
                                nbins=50,
                                title="Fraud Probability Distribution",
                                labels={'fraud_probability': 'Fraud Probability'}
                            )
                            st.plotly_chart(fig2, use_container_width=True)
                        
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                st.exception(e)

# Analytics & Reports Page
elif page == "Analytics & Reports":
    st.header("📊 Analytics & Downloadable Reports")
    
    if not st.session_state.model_trained:
        st.warning("Please train the model first in the 'Data Upload & Model Training' page.")
    else:
        st.subheader("User Segmentation Analysis")
        
        df = st.session_state.data.copy()
        
        # Merchant Analysis
        st.subheader("Analysis by Merchant")
        
        merchant_analysis = df.groupby('merchant_id').agg({
            'transaction_id': 'count',
            'amount': 'sum',
            'is_fraud': ['sum', 'mean']
        }).reset_index()
        
        merchant_analysis.columns = ['merchant_id', 'transaction_count', 'total_amount', 'fraud_count', 'fraud_rate']
        merchant_analysis['fraud_rate'] = merchant_analysis['fraud_rate'] * 100
        merchant_analysis = merchant_analysis.sort_values('fraud_count', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Merchants", len(merchant_analysis))
            top_risky_merchants = len(merchant_analysis[merchant_analysis['fraud_rate'] > 10])
            st.metric("High-Risk Merchants (>10% fraud rate)", top_risky_merchants)
        
        with col2:
            avg_merchant_fraud_rate = merchant_analysis['fraud_rate'].mean()
            st.metric("Average Merchant Fraud Rate", f"{avg_merchant_fraud_rate:.2f}%")
            max_fraud_merchant = merchant_analysis.iloc[0]['merchant_id']
            st.metric("Highest Fraud Count Merchant", f"ID: {max_fraud_merchant}")
        
        # Top risky merchants
        st.subheader("Top 10 Merchants by Fraud Count")
        st.dataframe(merchant_analysis.head(10), use_container_width=True)
        
        # Merchant visualization
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = px.scatter(
                merchant_analysis.head(50),
                x='transaction_count',
                y='fraud_rate',
                size='total_amount',
                hover_data=['merchant_id'],
                title='Merchant Risk Profile (Top 50)',
                labels={
                    'transaction_count': 'Number of Transactions',
                    'fraud_rate': 'Fraud Rate (%)',
                    'total_amount': 'Total Amount'
                }
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            fig2 = px.bar(
                merchant_analysis.head(10),
                x='merchant_id',
                y='fraud_count',
                title='Top 10 Merchants by Fraud Count',
                labels={'merchant_id': 'Merchant ID', 'fraud_count': 'Fraud Count'}
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        st.markdown("---")
        
        # Location Analysis
        st.subheader("Analysis by Location")
        
        location_analysis = df.groupby('location').agg({
            'transaction_id': 'count',
            'amount': 'sum',
            'is_fraud': ['sum', 'mean']
        }).reset_index()
        
        location_analysis.columns = ['location', 'transaction_count', 'total_amount', 'fraud_count', 'fraud_rate']
        location_analysis['fraud_rate'] = location_analysis['fraud_rate'] * 100
        location_analysis = location_analysis.sort_values('fraud_count', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig3 = px.bar(
                location_analysis,
                x='location',
                y='fraud_count',
                title='Fraud Count by Location',
                labels={'location': 'Location', 'fraud_count': 'Fraud Count'},
                color='fraud_rate',
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig3, use_container_width=True)
        
        with col2:
            fig4 = px.pie(
                location_analysis,
                values='transaction_count',
                names='location',
                title='Transaction Distribution by Location'
            )
            st.plotly_chart(fig4, use_container_width=True)
        
        st.dataframe(location_analysis, use_container_width=True)
        
        st.markdown("---")
        
        # Download comprehensive report
        st.subheader("Download Comprehensive Report")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Merchant report
            merchant_csv = merchant_analysis.to_csv(index=False)
            st.download_button(
                label="Download Merchant Analysis (CSV)",
                data=merchant_csv,
                file_name="merchant_analysis.csv",
                mime="text/csv"
            )
        
        with col2:
            # Location report
            location_csv = location_analysis.to_csv(index=False)
            st.download_button(
                label="Download Location Analysis (CSV)",
                data=location_csv,
                file_name="location_analysis.csv",
                mime="text/csv"
            )
        
        # Model performance report
        if st.session_state.model_trained:
            results = st.session_state.model.get_evaluation_results()
            
            report_text = f"""
FRAUD DETECTION MODEL PERFORMANCE REPORT
{'='*50}

Model Metrics:
- Accuracy: {results['accuracy']:.4f}
- ROC-AUC Score: {results['roc_auc']:.4f}
- Precision: {results['precision']:.4f}
- Recall: {results['recall']:.4f}
- Optimal Threshold: {results['optimal_threshold']:.3f}

Classification Report:
{results['classification_report']}

Dataset Summary:
- Total Transactions: {len(df)}
- Fraud Transactions: {df['is_fraud'].sum()}
- Fraud Rate: {(df['is_fraud'].sum() / len(df) * 100):.2f}%
- Total Transaction Amount: ${df['amount'].sum():,.2f}
- Total Fraud Amount: ${df[df['is_fraud'] == 1]['amount'].sum():,.2f}

Merchant Analysis:
- Total Merchants: {len(merchant_analysis)}
- High-Risk Merchants (>10% fraud): {len(merchant_analysis[merchant_analysis['fraud_rate'] > 10])}
- Average Merchant Fraud Rate: {merchant_analysis['fraud_rate'].mean():.2f}%

Location Analysis:
- Locations Analyzed: {len(location_analysis)}
- Highest Risk Location: {location_analysis.iloc[0]['location']} ({location_analysis.iloc[0]['fraud_rate']:.2f}% fraud rate)
"""
            
            st.download_button(
                label="Download Full Text Report",
                data=report_text,
                file_name="fraud_detection_report.txt",
                mime="text/plain"
            )

# Model Configuration Page
elif page == "Model Configuration":
    st.header("⚙️ Model Configuration & Retraining")
    
    st.subheader("Model Hyperparameters")
    
    st.info("Adjust the XGBoost model parameters and retrain with custom settings.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        n_estimators = st.slider("Number of Estimators", min_value=50, max_value=500, value=150, step=10)
        max_depth = st.slider("Max Depth", min_value=3, max_value=15, value=5, step=1)
        learning_rate = st.slider("Learning Rate", min_value=0.01, max_value=0.3, value=0.05, step=0.01)
        gamma = st.slider("Gamma", min_value=0.0, max_value=1.0, value=0.2, step=0.1)
    
    with col2:
        min_child_weight = st.slider("Min Child Weight", min_value=1, max_value=10, value=5, step=1)
        subsample = st.slider("Subsample", min_value=0.5, max_value=1.0, value=0.85, step=0.05)
        colsample_bytree = st.slider("Column Sample by Tree", min_value=0.5, max_value=1.0, value=0.85, step=0.05)
    
    st.markdown("---")
    
    st.subheader("Threshold Configuration")
    
    precision_lower = st.slider("Precision Lower Bound", min_value=0.0, max_value=1.0, value=0.4, step=0.05)
    precision_upper = st.slider("Precision Upper Bound", min_value=0.0, max_value=1.0, value=0.6, step=0.05)
    
    if precision_lower >= precision_upper:
        st.error("Precision lower bound must be less than upper bound!")
    
    st.markdown("---")
    
    if st.session_state.data is not None:
        if st.button("Retrain Model with Custom Parameters", type="primary"):
            with st.spinner("Retraining model with custom parameters... This may take a few minutes."):
                try:
                    # Create a custom model instance
                    custom_model = FraudDetectionModel()
                    
                    # Prepare custom parameters
                    custom_params = {
                        'n_estimators': n_estimators,
                        'max_depth': max_depth,
                        'learning_rate': learning_rate,
                        'gamma': gamma,
                        'min_child_weight': min_child_weight,
                        'subsample': subsample,
                        'colsample_bytree': colsample_bytree,
                        'precision_range': (precision_lower, precision_upper)
                    }
                    
                    # Train with custom parameters
                    training_results = custom_model.train(st.session_state.data, custom_params)
                    
                    # Update the session state model
                    st.session_state.model = custom_model
                    st.session_state.model_trained = True
                    
                    st.success("Model retrained successfully with custom parameters!")
                    
                    # Display training results
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Accuracy", f"{training_results['accuracy']:.4f}")
                    with col2:
                        st.metric("ROC-AUC", f"{training_results['roc_auc']:.4f}")
                    with col3:
                        st.metric("Optimal Threshold", f"{training_results['optimal_threshold']:.3f}")
                    
                    st.info("Note: The model has been retrained. You can now use it for predictions.")
                    
                except Exception as e:
                    st.error(f"Error retraining model: {str(e)}")
                    st.exception(e)
    else:
        st.warning("Please upload data first in the 'Data Upload & Model Training' page.")
    
    st.markdown("---")
    
    # Show current model info
    if st.session_state.model_trained:
        st.subheader("Current Model Information")
        
        results = st.session_state.model.get_evaluation_results()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{results['accuracy']:.4f}")
        with col2:
            st.metric("ROC-AUC", f"{results['roc_auc']:.4f}")
        with col3:
            st.metric("Precision", f"{results['precision']:.4f}")
        with col4:
            st.metric("Recall", f"{results['recall']:.4f}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Built with Streamlit • Fraud Detection Dashboard</p>
    </div>
    """, 
    unsafe_allow_html=True
)
