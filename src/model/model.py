import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import seaborn as sns
#from ydata_profiling import ProfileReport
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import xgboost as xgb
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    roc_auc_score, 
    precision_recall_curve, 
    average_precision_score,
    f1_score,
    log_loss
)
from joblib import Parallel, delayed
import joblib
import plotly.express as px
import os
from PyPDF2 import PdfMerger

def read_csv(input_path):
    """This function is read the input data path

    Args:
        input_path (csv): CSV file that contains the dataset
    Returns:
       df: Dataframe
    """
    df = pd.read_csv(input_path)
    print(df.head())
    return df

def prepare_data(df):
    """Preparing the dataset for training

    Args:
        df (dataframe): Dataframe
    Returns:
       X: Dataframe
       y: Dataframe
    """
    # Prepare the data for training
    df = df.drop(columns=['id'])
    X = df.drop('diagnosis', axis=1)
    print(X.head)
    y = df['diagnosis']
    print(y.head)
    return X,y

def cross_validation(X,y):  
    """Cross validation with XGB

    Args:
        X (dataframe): Dataframe
        y (dataframe): Dataframe
    Returns:
       fold_metrics: List of metrics
    """
    # Fold metrics to append into
    fold_metrics = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        print(f"Fold {fold + 1}")
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Scale the data
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Create the model
        model = xgb.XGBClassifier(
            objective = 'binary:logistic',
            eval_metric = 'logloss',
            random_state = 42,
            scale_pos_weight= 2
        )
        
        # Fit the model
        model.fit(X_train_scaled, y_train)
        
        # # Create a CalibratedClassifierCV
        # calibrated_model = CalibratedClassifierCV(model, method='sigmoid')
        # calibrated_model.fit(X_train_scaled, y_train, eval_set=[(X_test_scaled, y_test)], verbose=False)  # Fit the calibration

        # Prediction with the test dataset
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Print the classification report and confusion matrix
        print(classification_report(y_test, y_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        cm = confusion_matrix(y_test, y_pred)
        cm_norm = cm / cm.sum(axis=1)[:, np.newaxis]
        fig_confusion_matrix = px.imshow(cm_norm, labels=dict(x="Predicted", y="True"), 
                                     x=['Benign', 'Malignant'], y=['Benign', 'Malignant'], text_auto=True, color_continuous_scale=[[0.0, '#ee8ef5'], [1.0, 'rgb(50, 80, 168)']])
        fig_confusion_matrix.update_layout(title="Confusion Matrix", width=500, height=500, font=dict(
            family="Arial, Courier New, monospace",  # font family
            size=12,                                 # font size (pixels)
            color="darkblue"                         # font color
        ))
        fig_confusion_matrix.show()
        
        # Compute fold metrics
        fold_accuracy = accuracy_score(y_test, y_pred)
        fold_precision = precision_score(y_test, y_pred)
        fold_recall = recall_score(y_test, y_pred)
        fold_f1 = f1_score(y_test, y_pred)
        fold_roc_auc = roc_auc_score(y_test, y_proba)
        fold_log_loss = log_loss(y_test, y_proba)
        fold_pr_auc = average_precision_score(y_test, y_proba)


        fold_metrics.append({'precision': fold_precision, 'recall': fold_recall, 'f1': fold_f1, 'pr_auc': fold_pr_auc, 'roc_auc': fold_roc_auc, 'accuracy': fold_accuracy, 'log_loss': fold_log_loss})

        print(f"Fold {fold + 1} metrics: Precision={fold_precision}, Recall={fold_recall}, F1={fold_f1}, PR AUC={fold_pr_auc}, ROC AUC = {fold_roc_auc}  Accuracy={fold_accuracy}, Log Loss={fold_log_loss}")
    return fold_metrics

def aggregate_metrics(fold_metrics):
    """Aggreating the metrics from cross validation resulrs

    Args:
        fold_metrics (list): List of metrics
    Returns:
       model: Trained model
       calibrated_model: Calibrated model
       scaler: Scaler for the trained model
    """
    # Aggregate metrics
    mean_precision = np.mean([m['precision'] for m in fold_metrics])
    mean_recall = np.mean([m['recall'] for m in fold_metrics])
    mean_f1 = np.mean([m['f1'] for m in fold_metrics])
    mean_pr_auc = np.mean([m['pr_auc'] for m in fold_metrics])
    mean_accuracy = np.mean([m['accuracy'] for m in fold_metrics])
    mean_log_loss = np.mean([m['log_loss'] for m in fold_metrics])


    print(f"\nCross-Validation Results:\nPrecision: {mean_precision:.4f}, Recall: {mean_recall:.4f},  F1 Score: {mean_f1:.4f}, PR AUC: {mean_pr_auc:.4f}, Accuracy: {mean_accuracy:.4f}, Log Loss: {mean_log_loss:.4f}")

    # Create bar chart for accuracy, precision, recall, f1, and roc_auc
    fig_metrics = px.bar(x=['Accuracy', 'Precision', 'Recall', 'F1', 'PR-AUC'], y=[mean_accuracy, mean_precision, mean_recall,mean_f1, mean_pr_auc], 
                    labels={'x': 'Metric', 'y': 'Value'}, color=['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC'], 
                    color_discrete_map={'Accuracy': '#ee8ef5', 'Precision': 'rgb(176, 255, 221)', 'Recall': 'rgb(50, 80, 168)', 'F1': 'rgb(254, 255, 186)', 'PR-AUC': 'rgb(163, 231, 240)'})

    fig_metrics.update_layout(title="Model Metrics", width=1000, height=500)
    fig_metrics.update_layout(font=dict(
            family="Arial, Courier New, monospace",  # font family
            size=15,                                 # font size (pixels)
            color="darkblue"                         # font color
        ))
    fig_metrics.update_layout(plot_bgcolor='white')
    fig_metrics.update_xaxes(
                    mirror=True,
                    ticks='outside',
                    showline=True,
                    linecolor='rgb(242,242,242)',
                    gridcolor='rgb(242,242,242)'
                )
    fig_metrics.update_yaxes(
                    mirror=True,
                    ticks='outside',
                    showline=True,
                    linecolor='rgb(242,242,242)',
                    gridcolor='rgb(242,242,242)'
                )
    fig_metrics.show()

    filename = f"fig_metrics.pdf"
    fig_metrics.write_image(filename)
    
    return fig_metrics

def full_training(df):
    """Full training after cross validation

    Args:
        df (dataframe): Dataframe
    Returns:
       model: Trained model
       calibrated_model: Calibrated model
       scaler: Scaler for the trained model
    """
    # Scale the data
    X_train = df.drop(columns='diagnosis')

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    y_train = df['diagnosis'] 

    model = xgb.XGBClassifier(
            objective = 'binary:logistic',
            eval_metric = 'logloss',
            random_state = 42,
            scale_pos_weight= 2
        )
    model.fit(X_train_scaled, y_train)

    # Create a CalibratedClassifierCV
    calibrated_model = CalibratedClassifierCV(model, method='sigmoid')
    calibrated_model.fit(X_train_scaled, y_train, verbose=False)  # Fit the calibration
    
    return model, scaler, calibrated_model

def save_models(model, calibrated_model, scaler):
    # Save the XGBoost model (before calibration)
    model.save_model("xgboost_model.json")
    # Save the calibrated model
    joblib.dump(calibrated_model, 'calibrated_xgb_model.pkl')   
    # Save the scaler
    joblib.dump(scaler, 'scaler.pkl')