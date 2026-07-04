import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def main():
    print("Loading Iris dataset...")
    # Load scikit-learn's built-in iris dataset
    iris = load_iris()
    
    # Create a DataFrame for convenience
    df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                      columns=iris['feature_names'] + ['target'])
    
    # Map target integers to species names
    target_names = iris['target_names']
    df['species'] = df['target'].map(lambda x: target_names[int(x)])
    
    print("\nDataset Sample:")
    print(df.head())
    
    print("\nDataset Summary:")
    print(df.describe())
    
    # Features and target
    X = df[iris['feature_names']]
    y = df['target']
    
    # Train-test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Feature Scaling (important for Logistic Regression and SVM)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Dictionary of models to train
    models = {
        "Logistic Regression": LogisticRegression(max_iter=200, random_state=42),
        "Support Vector Machine": SVC(probability=True, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    best_model_name = None
    best_accuracy = 0
    best_model = None
    
    results = {}
    
    print("\nTraining and evaluating models:")
    for name, model in models.items():
        # Fit model
        if name in ["Logistic Regression", "Support Vector Machine"]:
            model.fit(X_train_scaled, y_train)
            predictions = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)  # Tree-based models don't need scaling
            predictions = model.predict(X_test)
            
        acc = accuracy_score(y_test, predictions)
        results[name] = acc
        print(f" - {name} Accuracy: {acc:.4f}")
        
        if acc > best_accuracy:
            best_accuracy = acc
            best_model_name = name
            best_model = model
            
    print(f"\nBest Model: {best_model_name} with Accuracy: {best_accuracy:.4f}")
    
    # Re-evaluate the best model in detail
    if best_model_name in ["Logistic Regression", "Support Vector Machine"]:
        y_pred = best_model.predict(X_test_scaled)
    else:
        y_pred = best_model.predict(X_test)
        
    print("\nClassification Report (Best Model):")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Save the best model and the scaler
    print("\nSaving best model and scaler...")
    joblib.dump(best_model, 'iris_model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    print("Saved 'iris_model.joblib' and 'scaler.joblib' successfully!")
    
    # Save target names as metadata for streamlit
    joblib.dump(target_names, 'target_names.joblib')
    print("Saved 'target_names.joblib'. Training pipeline complete!")

if __name__ == "__main__":
    main()
