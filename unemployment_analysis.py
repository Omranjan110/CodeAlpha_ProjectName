import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_clean_data(file_path):
    print(f"\n--- Loading and cleaning dataset: {os.path.basename(file_path)} ---")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    # Read the dataset
    df = pd.read_csv(file_path)
    print(f"Original shape: {df.shape}")
    print("Original columns:", df.columns.tolist())
    
    # 1. Clean column names (strip whitespaces)
    df.columns = df.columns.str.strip()
    print("Cleaned columns:", df.columns.tolist())
    
    # 2. Drop rows where all elements are missing, or key elements are missing
    # The dataset often has blank filler rows at the end
    df = df.dropna(subset=['Region', 'Date', 'Estimated Unemployment Rate (%)'])
    print(f"Shape after dropping nulls: {df.shape}")
    
    # 3. Clean date column and convert to datetime
    df['Date'] = df['Date'].str.strip()
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    
    # 4. Extract Year, Month, and Month Name for analysis
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Month_Name'] = df['Date'].dt.strftime('%b')
    
    # 5. Clean area or state names if they have leading/trailing spaces
    df['Region'] = df['Region'].str.strip()
    if 'Area' in df.columns:
        df['Area'] = df['Area'].str.strip()
    if 'Region.1' in df.columns:
        df['Region.1'] = df['Region.1'].str.strip()
        
    return df

def segment_covid_periods(df):
    # Segment data by COVID-19 periods in 2020
    # National Lockdown in India was announced on March 24, 2020.
    # Peak lockdown impact was felt in April, May, and June 2020.
    def get_period(date):
        if date < pd.Timestamp('2020-04-01'):
            return 'Pre-Lockdown (Jan-Mar 2020)'
        elif date <= pd.Timestamp('2020-06-30'):
            return 'Lockdown Peak (Apr-Jun 2020)'
        else:
            return 'Recovery Phase (Jul-Nov 2020)'
            
    df['Lockdown_Phase'] = df['Date'].apply(get_period)
    return df

def main():
    # File paths
    file_india = os.path.join("unemployment_data", "Unemployment in India.csv")
    file_detailed = os.path.join("unemployment_data", "Unemployment_Rate_upto_11_2020.csv")
    
    # Load and clean datasets
    df_india = load_and_clean_data(file_india)
    df_detailed = load_and_clean_data(file_detailed)
    
    # Segment lockdown phases
    df_india = segment_covid_periods(df_india)
    df_detailed = segment_covid_periods(df_detailed)
    
    os.makedirs('static_plots', exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    # --- ANALYSIS & VISUALIZATIONS ---
    
    # 1. National Unemployment Trend over Time (Detailed Dataset)
    print("\n1. Generating national unemployment timeline plot...")
    timeline = df_detailed.groupby('Date')['Estimated Unemployment Rate (%)'].mean().reset_index()
    
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=timeline, x='Date', y='Estimated Unemployment Rate (%)', marker='o', color='#E11D48', linewidth=2.5)
    plt.title("National Monthly Average Unemployment Rate in India (2020)", fontsize=14, pad=15)
    plt.xlabel("Timeline", fontsize=12)
    plt.ylabel("Average Unemployment Rate (%)", fontsize=12)
    plt.axvspan(pd.Timestamp('2020-04-01'), pd.Timestamp('2020-06-30'), color='red', alpha=0.1, label='Lockdown Peak')
    plt.legend()
    plt.tight_layout()
    plt.savefig("static_plots/unemployment_national_trend.png", dpi=300)
    plt.close()
    
    # 2. COVID-19 Period Impact Comparison (Detailed Dataset)
    print("2. Generating lockdown phase impact comparison plot...")
    impact = df_detailed.groupby('Lockdown_Phase')['Estimated Unemployment Rate (%)'].mean().reset_index()
    
    plt.figure(figsize=(8, 5))
    bars = sns.barplot(
        data=impact, x='Lockdown_Phase', y='Estimated Unemployment Rate (%)', 
        palette=['#10B981', '#EF4444', '#F59E0B'], order=[
            'Pre-Lockdown (Jan-Mar 2020)', 
            'Lockdown Peak (Apr-Jun 2020)', 
            'Recovery Phase (Jul-Nov 2020)'
        ]
    )
    plt.title("Average Unemployment Rate by COVID-19 Lockdown Phase", fontsize=14, pad=15)
    plt.xlabel("Lockdown Phase", fontsize=12)
    plt.ylabel("Average Unemployment Rate (%)", fontsize=12)
    
    # Add values on top of bars
    for bar in bars.patches:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.3, f"{yval:.2f}%", ha='center', va='bottom', fontweight='semibold')
        
    plt.tight_layout()
    plt.savefig("static_plots/unemployment_covid_impact.png", dpi=300)
    plt.close()
    
    # 3. Rural vs Urban Comparison (India Dataset - contains Area column)
    print("3. Generating Rural vs Urban unemployment comparison boxplot...")
    plt.figure(figsize=(7, 5))
    sns.boxplot(data=df_india, x='Area', y='Estimated Unemployment Rate (%)', palette="Set2", width=0.5)
    plt.title("Rural vs Urban Unemployment Rate Distribution", fontsize=14, pad=15)
    plt.xlabel("Area Sector", fontsize=12)
    plt.ylabel("Unemployment Rate (%)", fontsize=12)
    plt.tight_layout()
    plt.savefig("static_plots/unemployment_rural_vs_urban.png", dpi=300)
    plt.close()
    
    # 4. State-Wise Heatmap over 2020 (Detailed Dataset)
    print("4. Generating State-wise monthly heatmap...")
    # Sort months chronologically
    df_detailed = df_detailed.sort_values('Date')
    state_pivot = df_detailed.pivot_table(
        index='Region', 
        columns='Month_Name', 
        values='Estimated Unemployment Rate (%)',
        aggfunc='mean'
    )
    # Reorder columns to calendar order in 2020 (Jan to Oct/Nov)
    months_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov']
    available_months = [m for m in months_order if m in state_pivot.columns]
    state_pivot = state_pivot[available_months]
    
    plt.figure(figsize=(10, 10))
    sns.heatmap(state_pivot, annot=True, cmap="YlOrRd", fmt=".1f", linewidths=0.5, cbar_kws={'label': 'Unemployment Rate (%)'})
    plt.title("State-Wise Average Monthly Unemployment Rate Heatmap (2020)", fontsize=14, pad=15)
    plt.xlabel("Month in 2020", fontsize=12)
    plt.ylabel("State / Region", fontsize=12)
    plt.tight_layout()
    plt.savefig("static_plots/unemployment_state_heatmap.png", dpi=300)
    plt.close()
    
    # Print Key Insights to console
    print("\n--- Key Insights from Analysis ---")
    overall_avg = df_detailed['Estimated Unemployment Rate (%)'].mean()
    lockdown_avg = df_detailed[df_detailed['Lockdown_Phase'] == 'Lockdown Peak (Apr-Jun 2020)']['Estimated Unemployment Rate (%)'].mean()
    pre_lockdown_avg = df_detailed[df_detailed['Lockdown_Phase'] == 'Pre-Lockdown (Jan-Mar 2020)']['Estimated Unemployment Rate (%)'].mean()
    recovery_avg = df_detailed[df_detailed['Lockdown_Phase'] == 'Recovery Phase (Jul-Nov 2020)']['Estimated Unemployment Rate (%)'].mean()
    
    print(f"Overall Average Unemployment Rate: {overall_avg:.2f}%")
    print(f"Pre-Lockdown Average (Jan-Mar): {pre_lockdown_avg:.2f}%")
    print(f"Lockdown Peak Average (Apr-Jun): {lockdown_avg:.2f}% (An increase of {lockdown_avg - pre_lockdown_avg:.2f} percentage points)")
    print(f"Recovery Phase Average (Jul-Nov): {recovery_avg:.2f}%")
    
    # Find worst hit states during lockdown
    lockdown_df = df_detailed[df_detailed['Lockdown_Phase'] == 'Lockdown Peak (Apr-Jun 2020)']
    worst_hit_states = lockdown_df.groupby('Region')['Estimated Unemployment Rate (%)'].mean().sort_values(ascending=False).head(5)
    print("\nWorst hit states during the lockdown peak (average %):")
    for state, rate in worst_hit_states.items():
        print(f" - {state}: {rate:.2f}%")
        
    # Rural vs Urban average
    rural_avg = df_india[df_india['Area'] == 'Rural']['Estimated Unemployment Rate (%)'].mean()
    urban_avg = df_india[df_india['Area'] == 'Urban']['Estimated Unemployment Rate (%)'].mean()
    print(f"\nNational Sector averages (Jan-Jun 2020):")
    print(f" - Rural Unemployment Rate: {rural_avg:.2f}%")
    print(f" - Urban Unemployment Rate: {urban_avg:.2f}%")
    
    print("\nAnalysis complete! Visual plots saved under 'static_plots/' directory:")
    print(" - static_plots/unemployment_national_trend.png")
    print(" - static_plots/unemployment_covid_impact.png")
    print(" - static_plots/unemployment_rural_vs_urban.png")
    print(" - static_plots/unemployment_state_heatmap.png")

if __name__ == "__main__":
    main()
