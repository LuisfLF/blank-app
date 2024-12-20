import streamlit as st


import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, classification_report
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from collections import Counter

# Configuración de la página con título e ícono
st.set_page_config(
    page_title="Detección de Inmunodeficiencias",
    page_icon="icon.svg",  # Asegúrate de que el archivo `icon.png` esté en tu directorio de trabajo
)

# Function to run the model and return results
def run_model_DX(data, target_num, features, scale=False):
    data['pre_target']= data['Category'].astype('category').cat.codes
    data['target'] = [1 if x == target_num else 0 for x in data['pre_target']]
    casos = data[data['target'] == 1].Category.tolist()
    positives = len(casos)
    negatives = len(data) - positives

    X = data[features]
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    model = xgb.XGBClassifier(
        colsample_bylevel=0.7,
        colsample_bynode=1,
        colsample_bytree=0.7,
        learning_rate=0.01,
        max_depth=3,
        n_estimators=500,
        subsample=1.0,
        seed=123,
        scale_pos_weight=negatives / positives if scale else 1
    )

    # Fit the model
    model.fit(X_train, y_train)
    pred_val = model.predict(X_test)

    # Evaluation metrics
    accuracy_test = accuracy_score(y_test, pred_val)
    report_precision_recall = classification_report(y_test, pred_val, output_dict=True)
    auc_test = roc_auc_score(y_test, pred_val)
    logloss_test = log_loss(y_test, pred_val)

    # SHAP analysis
    explainer = shap.TreeExplainer(model)
    shaps_values = pd.DataFrame(explainer.shap_values(X_test), columns=X_test.columns)

    return {
        'accuracy': accuracy_test,
        'precision_recall': report_precision_recall,
        'auc': auc_test,
        'logloss': logloss_test,
        'shap_values': shaps_values,
        'model': model
    }

# Function to plot top features (symptoms)
def plot_top_features(data, features):
    # Count the frequency of each feature (symptom)
    feature_counts = Counter()

    for feature in features:
        feature_counts[feature] += data[feature].sum()  # Assuming binary features (0/1)

    # Get the top 20 most common features
    top_20_features = feature_counts.most_common(20)

    # Create a DataFrame for easier plotting
    df_top_20 = pd.DataFrame(top_20_features, columns=['Feature', 'Count'])

    # Plotting the bar chart
    plt.figure(figsize=(12, 6))
    plt.barh(df_top_20['Feature'], df_top_20['Count'], color='skyblue')
    plt.xlabel('Count')
    plt.title('Top 20 Most Common Symptoms Across All Categories')
    plt.gca().invert_yaxis()  # Invert y-axis to show the most common feature at the top
    st.pyplot(plt)  # Display the plot in Streamlit

# Streamlit application
def main():
    st.title("Classification Model Evaluation with SHAP Analysis")
    BASE = "USIDNET2"
    # File uploader for CSV
    uploaded_file = pd.read_csv(BASE + ".csv")
    # Load data
   
    data = uploaded_file.copy()



    
    if uploaded_file is not None:
        
         # Agregar el título de la previsualización
        st.subheader("Database Preview")
        #data = pd.read_csv(uploaded_file)
        st.write(data.head())

        # Selecting features (symptoms)
        features = [x for x in data.columns if x not in ['target', 'id_px', 'Category']]
        
        # Plot the top features (symptoms)
        plot_top_features(data, features)

        # Create a mapping of target IDs to names for categories
        target_mapping = {i: f"{i} - {name}" for i, name in enumerate(data['Category'].unique())}
        
        # Select target category
        target_num = st.selectbox("Select Target Category", options=list(target_mapping.keys()), format_func=lambda x: target_mapping[x])
        
 
        
        if st.button("See Results"):
            results = run_model_DX(data, target_num, features, scale=False)
            
            # Display results
            st.subheader("Model Evaluation Metrics")
            st.write(f"Accuracy: {results['accuracy']:.4f}")
            st.write(f"AUC: {results['auc']:.4f}")
            st.write(f"Log Loss: {results['logloss']:.4f}")
            
            # Precision-Recall Report
            st.subheader("Precision-Recall Report")
            report_df = pd.DataFrame(results['precision_recall']).drop(columns=['macro avg', 'weighted avg', 'accuracy'])
            st.write(report_df)

            # SHAP Summary Plot
            st.subheader("SHAP Summary Plot")
            shap.summary_plot(results['shap_values'].values, results['shap_values'].columns, max_display=20)
            plt.title("SHAP Summary Plot")
            st.pyplot(plt)

            # Save model
            if not os.path.exists('data'):
                os.makedirs('data')
            filename = f'data/model_{target_num}.pkl'
            with open(filename, 'wb') as f:
                pickle.dump(results['model'], f)

if __name__ == "__main__":
    main()
