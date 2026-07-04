import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

def main():
    print("=== Exploratory Data Analysis ===")
    
    # 1. Load Iris dataset
    iris = load_iris()
    df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                      columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target'])
    df['species'] = df['target'].map(lambda x: iris['target_names'][int(x)])
    
    # Create directory for figures
    os.makedirs('static_plots', exist_ok=True)
    
    # 2. General statistics
    print("\n1. Basic Dataset Information:")
    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")
    print("\nSpecies distribution:")
    print(df['species'].value_counts())
    
    print("\nSummary Statistics:")
    print(df.describe())
    
    # Set premium plotting theme
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
        'figure.titlesize': 16,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10
    })
    
    # 3. Save Pair Plot to show linear and non-linear separability
    print("\n2. Generating Pair Plot...")
    pairplot = sns.pairplot(df.drop(columns=['target']), hue="species", palette="muted", markers=["o", "s", "D"])
    pairplot.fig.suptitle("Iris Feature Pairwise Relationships", y=1.02)
    pairplot.savefig("static_plots/iris_pairplot.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Save Correlation Heatmap
    print("3. Generating Correlation Heatmap...")
    plt.figure(figsize=(6, 5))
    features_df = df.drop(columns=['target', 'species'])
    correlation_matrix = features_df.corr()
    
    # Custom color palette (coolwarm)
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, vmin=-1, vmax=1)
    plt.title("Iris Feature Correlation Matrix", pad=15)
    plt.tight_layout()
    plt.savefig("static_plots/iris_correlation_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Boxplots of all features by species
    print("4. Generating Distribution Boxplots...")
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    titles = ['Sepal Length (cm)', 'Sepal Width (cm)', 'Petal Length (cm)', 'Petal Width (cm)']
    
    for i, ax in enumerate(axes.flat):
        sns.boxplot(data=df, x='species', y=features[i], palette="Set2", ax=ax, width=0.6)
        ax.set_title(titles[i])
        ax.set_xlabel("")
        ax.set_ylabel("")
        
    plt.suptitle("Distribution of Features by Species", y=0.98)
    plt.tight_layout()
    plt.savefig("static_plots/iris_feature_boxplots.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nEDA complete! Generated plots saved to the 'static_plots/' directory:")
    print(" - static_plots/iris_pairplot.png")
    print(" - static_plots/iris_correlation_heatmap.png")
    print(" - static_plots/iris_feature_boxplots.png")

if __name__ == "__main__":
    main()
