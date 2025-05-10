import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_prepare_data(file_path='e:\\Rohit\\samplecode.csv'):
    # Load the customer data
    df = pd.read_csv(file_path)
    
    # Convert 'TotalCharges' to numeric, handling any spaces
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Convert 'Churn' to binary (0/1)
    df['Churn'] = (df['Churn'] == 'Yes').astype(int)
    
    # Drop CustomerID as it's not relevant for prediction
    df = df.drop(['CustomerID'], axis=1, errors='ignore')
    
    # Convert categorical variables to numeric
    categorical_columns = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=categorical_columns)
    
    return df

def train_churn_model():
    # Load and prepare the data using the actual CSV file
    df = load_and_prepare_data()
    
    # Separate features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return model, feature_importance, y_test, y_pred, scaler

def visualize_results(feature_importance, y_test, y_pred, df):
    # Create a figure with subplots arranged in a grid
    plt.style.use('default')
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Feature Importance Plot (Top Left)
    ax1 = plt.subplot(2, 2, 1)
    total_importance = feature_importance['importance'].sum()
    feature_importance['percentage'] = (feature_importance['importance'] / total_importance) * 100
    sns.barplot(data=feature_importance.head(10),
                x='percentage', 
                y='feature',
                hue=None,
                palette='viridis', 
                ax=ax1)
    ax1.set_title('Top 10 Features Impact (%)', fontsize=12)
    for i, v in enumerate(feature_importance.head(10)['percentage']):
        ax1.text(v, i, f'{v:.1f}%', va='center', fontsize=10)
    
    # 2. Confusion Matrix (Top Right)
    ax2 = plt.subplot(2, 2, 2)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd',
                xticklabels=['Not Churned', 'Churned'],
                yticklabels=['Not Churned', 'Churned'], ax=ax2)
    ax2.set_title('Confusion Matrix', fontsize=12)
    
    # 3. Churn Distribution (Bottom Left)
    ax3 = plt.subplot(2, 2, 3)
    test_dist = pd.Series(y_test).value_counts(normalize=True) * 100
    ax3.pie(test_dist,
            labels=[f'Not Churned\n({test_dist[0]:.1f}%)', 
                   f'Churned\n({test_dist[1]:.1f}%)'],
            colors=['lightblue', 'coral'],
            autopct='%1.1f%%',
            explode=(0, 0.1))
    ax3.set_title('Churn Distribution', fontsize=12)
    
    # 4. Monthly Charges vs Churn (Bottom Right)
    ax4 = plt.subplot(2, 2, 4)
    sns.boxplot(x=y_test, y=df['MonthlyCharges'], ax=ax4,
                palette=['lightblue', 'coral'])
    ax4.set_title('Monthly Charges by Churn Status', fontsize=12)
    ax4.set_xlabel('Churn Status (0: Not Churned, 1: Churned)')
    ax4.set_ylabel('Monthly Charges ($)')
    
    plt.tight_layout(pad=3.0)
    plt.savefig('e:\\Rohit\\churn_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()

# Update the main execution block
if __name__ == "__main__":
    # Load data
    df = load_and_prepare_data()
    
    # Train the model and get results
    model, feature_importance, y_test, y_pred, scaler = train_churn_model()
    
    # Print model performance
    print("\nModel Performance Report:")
    print(classification_report(y_test, y_pred))
    
    # Create visualization dashboard
    visualize_results(feature_importance, y_test, y_pred, df)
    
    print("\nModel training completed. Check 'churn_dashboard.png' for visual insights.")