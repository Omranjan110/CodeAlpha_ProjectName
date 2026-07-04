import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def main():
    print("=== Car Price Prediction Model Training Pipeline ===")
    
    # 1. Load the dataset
    file_path = os.path.join("car_price_data", "CarPrice_Assignment.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")
        
    df = pd.read_csv(file_path)
    print(f"Dataset shape: {df.shape}")
    
    # 2. Feature Engineering: Extract brand/company name from CarName
    # The first word in CarName is the brand (e.g. "alfa-romero giulia" -> "alfa-romero")
    df['brand'] = df['CarName'].apply(lambda x: x.split(' ')[0].strip().lower())
    
    # Standardize and correct brand name spelling errors (capturing Brand Goodwill)
    brand_corrections = {
        'maxda': 'mazda',
        'toyouta': 'toyota',
        'vokswagen': 'volkswagen',
        'vw': 'volkswagen',
        'porcshz': 'porsche',
        'alfa-romero': 'alfa-romeo'
    }
    df['brand'] = df['brand'].replace(brand_corrections)
    
    print("\nUnique Brands Cleaned:")
    print(sorted(df['brand'].unique()))
    
    # 3. Create static brand distribution plot
    os.makedirs('static_plots', exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    print("\nGenerating brand-wise average price chart...")
    brand_prices = df.groupby('brand')['price'].mean().sort_values(ascending=False).reset_index()
    plt.figure(figsize=(12, 6))
    sns.barplot(data=brand_prices, x='brand', y='price', palette="viridis")
    plt.xticks(rotation=45, ha='right')
    plt.title("Average Car Price by Brand (Goodwill Impact)", fontsize=14, pad=15)
    plt.xlabel("Car Brand / Manufacturer", fontsize=12)
    plt.ylabel("Average Price ($)", fontsize=12)
    plt.tight_layout()
    plt.savefig("static_plots/car_price_brand_distribution.png", dpi=300)
    plt.close()
    
    # 4. Create numerical correlation heatmap
    print("Generating numerical features correlation heatmap...")
    numerical_cols = ['wheelbase', 'carlength', 'carwidth', 'carheight', 'curbweight', 
                      'enginesize', 'boreratio', 'stroke', 'compressionratio', 
                      'horsepower', 'peakrpm', 'citympg', 'highwaympg', 'price']
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[numerical_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Car Specifications Correlation Matrix", fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig("static_plots/car_price_correlation_heatmap.png", dpi=300)
    plt.close()
    
    # 5. Preprocessing & Split
    # Select features
    numerical_features = ['horsepower', 'enginesize', 'curbweight', 'citympg', 'highwaympg', 'wheelbase', 'carlength', 'carwidth']
    categorical_features = ['brand', 'carbody', 'fueltype', 'aspiration', 'drivewheel']
    
    X = df[numerical_features + categorical_features]
    y = df['price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Setup scikit-learn transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )
    
    # 6. Train and compare models
    models = {
        "Linear Regression": Pipeline(steps=[('preprocessor', preprocessor), ('regressor', LinearRegression())]),
        "Ridge Regression": Pipeline(steps=[('preprocessor', preprocessor), ('regressor', Ridge(alpha=1.0))]),
        "Random Forest Regressor": Pipeline(steps=[('preprocessor', preprocessor), ('regressor', RandomForestRegressor(n_estimators=150, random_state=42))])
    }
    
    best_model_name = None
    best_r2 = -float('inf')
    best_pipeline = None
    
    print("\nTraining and evaluating models:")
    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f" - {name}:")
        print(f"   R² Score: {r2:.4f}")
        print(f"   Mean Absolute Error (MAE): ${mae:.2f}")
        print(f"   Root Mean Squared Error (RMSE): ${rmse:.2f}")
        
        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name
            best_pipeline = pipeline
            
    print(f"\nBest Model: {best_model_name} (R² = {best_r2:.4f})")
    
    # Save the best model pipeline
    print("Saving best model pipeline...")
    joblib.dump(best_pipeline, 'car_price_model.joblib')
    print("Saved 'car_price_model.joblib' successfully!")
    
    # Save the training DataFrame columns and categories for app UI mapping
    joblib.dump(X_train.columns.tolist(), 'car_features_list.joblib')
    joblib.dump(sorted(df['brand'].unique().tolist()), 'car_brands_list.joblib')
    joblib.dump(sorted(df['carbody'].unique().tolist()), 'car_bodies_list.joblib')
    joblib.dump(sorted(df['fueltype'].unique().tolist()), 'car_fuels_list.joblib')
    joblib.dump(sorted(df['aspiration'].unique().tolist()), 'car_aspirations_list.joblib')
    joblib.dump(sorted(df['drivewheel'].unique().tolist()), 'car_drives_list.joblib')
    print("Saved UI metadata files successfully!")
    
    # 7. Generate Prediction Scatter Plot
    print("\nGenerating prediction scatter plot for best model...")
    y_pred_best = best_pipeline.predict(X_test)
    plt.figure(figsize=(7, 5))
    sns.scatterplot(x=y_test, y=y_pred_best, color='#3B82F6', alpha=0.8)
    # Plot line of equality
    max_val = max(y_test.max(), y_pred_best.max())
    min_val = min(y_test.min(), y_pred_best.min())
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='Perfect Fit')
    plt.title(f"Car Price Prediction: Actual vs Predicted ({best_model_name})", fontsize=13, pad=15)
    plt.xlabel("Actual Price ($)", fontsize=11)
    plt.ylabel("Predicted Price ($)", fontsize=11)
    plt.legend()
    plt.tight_layout()
    plt.savefig("static_plots/car_price_prediction_comparison.png", dpi=300)
    plt.close()
    
    # 8. Generate Feature Importance Plot for Random Forest
    if best_model_name == "Random Forest Regressor":
        print("Generating Random Forest feature importance plot...")
        rf_model = best_pipeline.named_steps['regressor']
        preproc = best_pipeline.named_steps['preprocessor']
        
        # Get feature names out of encoder
        feature_names = preproc.get_feature_names_out()
        
        # Clean feature names (remove prefix num__ and cat__)
        clean_names = [f.replace('num__', '').replace('cat__', '') for f in feature_names]
        
        importances = rf_model.feature_importances_
        forest_importances = pd.Series(importances, index=clean_names)
        
        # Take top 10 features
        top_features = forest_importances.sort_values(ascending=False).head(10)
        
        plt.figure(figsize=(8, 5))
        sns.barplot(x=top_features.values, y=top_features.index, palette="mako")
        plt.title("Top 10 Feature Importances in Car Price Prediction", fontsize=14, pad=15)
        plt.xlabel("Relative Importance Weight", fontsize=12)
        plt.tight_layout()
        plt.savefig("static_plots/car_price_feature_importance.png", dpi=300)
        plt.close()
        
    print("\nTraining pipeline execution finished!")

if __name__ == "__main__":
    main()
