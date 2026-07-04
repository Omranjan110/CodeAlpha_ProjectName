import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
import pydeck as pdk
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Set page configuration for a premium, wide dashboard layout
st.set_page_config(
    page_title="Internship Projects Dashboard - CodeAlpha",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling for premium look (glassmorphism, vibrant colors, clean fonts)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main-title {
        font-size: 2.8rem !important;
        font-weight: 800;
        background: linear-gradient(90deg, #3B82F6 0%, #10B981 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .main-title-iris {
        font-size: 2.8rem !important;
        font-weight: 800;
        background: linear-gradient(90deg, #FF4B4B 0%, #FF8F00 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        font-size: 1.15rem;
        color: #6c757d;
        margin-bottom: 1.8rem;
    }
    
    .metric-container {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1.2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #3B82F6;
        margin: 5px 0;
    }
    
    .metric-value-red {
        font-size: 2.2rem;
        font-weight: 700;
        color: #EF4444;
        margin: 5px 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        text-transform: uppercase;
        color: #6B7280;
        letter-spacing: 0.5px;
    }
    
    .predicted-class {
        font-size: 2.2rem;
        font-weight: 700;
        color: #FF8F00;
        margin-top: 10px;
        margin-bottom: 10px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }
    
    .feature-box {
        background-color: rgba(30, 41, 59, 0.5);
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #FF4B4B;
        margin-bottom: 15px;
    }
    
    .insight-card {
        background: rgba(59, 130, 246, 0.05);
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #3B82F6;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR NAVIGATION ---
st.sidebar.markdown("# 🚀 CodeAlpha Internship")
selected_task = st.sidebar.selectbox(
    "Choose Analysis Task",
    ["🌸 Task 1: Iris Flower Classification", "📉 Task 2: Unemployment Analysis"]
)

# --- TASK 1: IRIS FLOWER CLASSIFICATION ---
if selected_task == "🌸 Task 1: Iris Flower Classification":
    
    # Copy generated images from artifact folder to workspace assets folder
    def setup_assets():
        artifact_dir = r"C:\Users\shash\.gemini\antigravity-ide\brain\d31eadff-cedd-434c-b0b2-2d6afc59fd15"
        setosa_src = os.path.join(artifact_dir, "iris_setosa_1783150244644.png")
        versicolor_src = os.path.join(artifact_dir, "iris_versicolor_1783150266984.png")
        virginica_src = os.path.join(artifact_dir, "iris_virginica_1783150325165.png")

        os.makedirs("assets", exist_ok=True)
        image_maps = {
            setosa_src: "setosa.png",
            versicolor_src: "versicolor.png",
            virginica_src: "virginica.png"
        }
        for src, dest_name in image_maps.items():
            dest_path = os.path.join("assets", dest_name)
            if os.path.exists(src) and not os.path.exists(dest_path):
                shutil.copy(src, dest_path)

    setup_assets()

    # Load the Iris dataset for visualization reference
    @st.cache_data
    def get_dataset():
        iris = load_iris()
        df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                          columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target'])
        df['species'] = df['target'].map(lambda x: iris['target_names'][int(x)])
        return iris, df

    iris_data, df = get_dataset()

    # Lazy model training helper if saved model does not exist
    def train_fallback_model(df, iris):
        X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].copy()
        X.columns = iris['feature_names']
        y = df['target']
        
        scaler = StandardScaler()
        scaler.fit(X)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        joblib.dump(model, 'iris_model.joblib')
        joblib.dump(scaler, 'scaler.joblib')
        joblib.dump(iris['target_names'], 'target_names.joblib')
        return model, scaler, iris['target_names']

    # Load model, scaler, and target names
    @st.cache_resource
    def load_ml_components():
        model_path = 'iris_model.joblib'
        scaler_path = 'scaler.joblib'
        target_names_path = 'target_names.joblib'
        
        if os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(target_names_path):
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            target_names = joblib.load(target_names_path)
        else:
            model, scaler, target_names = train_fallback_model(df, iris_data)
        
        return model, scaler, target_names

    model, scaler, target_names = load_ml_components()

    # --- Title Header ---
    st.markdown('<div class="main-title-iris">Iris Flower Species Classifier</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">An interactive AI dashboard visualizing and classifying Iris flowers using Scikit-Learn and Streamlit</div>', unsafe_allow_html=True)

    # --- Sidebar Input Panel ---
    st.sidebar.markdown("### 🎛️ Flower Measurements")
    st.sidebar.write("Adjust the sliders below to measure a flower and see its species classified in real-time.")

    sepal_len = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.8, step=0.1)
    sepal_wid = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.0, step=0.1)
    petal_len = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 4.3, step=0.1)
    petal_wid = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 1.3, step=0.1)

    # Format features into DataFrame matching fit time names
    user_features = pd.DataFrame(
        data=[[sepal_len, sepal_wid, petal_len, petal_wid]],
        columns=iris_data['feature_names']
    )

    model_name = type(model).__name__

    if model_name in ["SVC", "LogisticRegression"]:
        scaled_features = scaler.transform(user_features)
        prediction_idx = int(model.predict(scaled_features)[0])
        probabilities = model.predict_proba(scaled_features)[0]
    else:
        prediction_idx = int(model.predict(user_features)[0])
        probabilities = model.predict_proba(user_features)[0]

    predicted_species = target_names[prediction_idx]
    confidence = probabilities[prediction_idx] * 100

    # --- Dashboard Layout ---
    tab1, tab2, tab3 = st.tabs(["🔮 Real-Time Classification", "📊 Exploratory Data Analysis", "⚙️ Model Diagnostics"])

    with tab1:
        col1, col2 = st.columns([1, 1.2], gap="large")
        
        with col1:
            st.markdown("### 🌿 Classification Result")
            st.markdown(f"""
            <div style="background: rgba(255, 75, 75, 0.08); border-radius: 12px; padding: 2rem; border: 2px solid #FF4B4B; text-align: center;">
                <h4>PREDICTED SPECIES</h4>
                <div class="predicted-class">Iris {predicted_species}</div>
                <p style="font-size: 1.1rem; color: #555;">Confidence Score: <strong>{confidence:.2f}%</strong></p>
            </div>
            """, unsafe_allow_html=True)
            
            st.write("")
            st.markdown("#### Feature Values Inputted:")
            st.markdown(f"""
            <div class="feature-box">
                <strong>📐 Sepal:</strong> Length: {sepal_len} cm | Width: {sepal_wid} cm<br>
                <strong>🌸 Petal:</strong> Length: {petal_len} cm | Width: {petal_wid} cm
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### Prediction Probabilities:")
            prob_df = pd.DataFrame({
                'Species': ['Iris Setosa', 'Iris Versicolor', 'Iris Virginica'],
                'Probability (%)': [p * 100 for p in probabilities]
            })
            
            fig, ax = plt.subplots(figsize=(6, 2.2))
            colors = ['#4A90E2', '#50E3C2', '#B8E986']
            bars = ax.barh(prob_df['Species'], prob_df['Probability (%)'], color=colors, height=0.55)
            ax.set_xlim(0, 100)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.xaxis.grid(True, linestyle='--', alpha=0.3)
            ax.set_axisbelow(True)
            
            for bar in bars:
                width = bar.get_width()
                ax.text(width + 2, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', 
                        va='center', ha='left', fontweight='semibold', color='#555')
                        
            plt.tight_layout()
            st.pyplot(fig)
            
        with col2:
            st.markdown("### 📸 Species Visualization")
            image_file = os.path.join("assets", f"{predicted_species.lower()}.png")
            if os.path.exists(image_file):
                st.image(image_file, caption=f"Stunning close-up of Iris {predicted_species.title()}", use_container_width=True)
            else:
                st.info("No customized species image available. Standard illustration placeholder:")
                st.warning("Generate flower images to display them here.")

    with tab2:
        st.markdown("### 📊 Dataset Distributions & Interactive Feature Plots")
        st.write("Visualize the complete Iris dataset and see where your custom input (marked as the red star ⭐) lies.")
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("#### Sepal Measurements Scatter Plot")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.scatterplot(
                data=df, x='sepal_length', y='sepal_width', hue='species', 
                palette=['#4A90E2', '#E28E4A', '#50E3C2'], alpha=0.8, ax=ax
            )
            ax.scatter(sepal_len, sepal_wid, color='red', marker='*', s=280, edgecolor='black', linewidth=1.5, label='Custom Input')
            ax.set_xlabel("Sepal Length (cm)")
            ax.set_ylabel("Sepal Width (cm)")
            ax.legend()
            ax.set_title("Sepal Length vs Sepal Width")
            st.pyplot(fig)
            
        with col4:
            st.markdown("#### Petal Measurements Scatter Plot")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.scatterplot(
                data=df, x='petal_length', y='petal_width', hue='species', 
                palette=['#4A90E2', '#E28E4A', '#50E3C2'], alpha=0.8, ax=ax
            )
            ax.scatter(petal_len, petal_wid, color='red', marker='*', s=280, edgecolor='black', linewidth=1.5, label='Custom Input')
            ax.set_xlabel("Petal Length (cm)")
            ax.set_ylabel("Petal Width (cm)")
            ax.legend()
            ax.set_title("Petal Length vs Petal Width")
            st.pyplot(fig)

    with tab3:
        st.markdown("### ⚙️ Machine Learning Model Information")
        st.markdown(f"**Classifier in Use:** `{model_name}`")
        st.write("The classification model is trained on **80%** of the standard Iris dataset and validated on the remaining **20%**.")
        
        col5, col6 = st.columns(2)
        
        with col5:
            st.markdown("#### Feature Importance")
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
                indices = np.argsort(importances)
                
                fig, ax = plt.subplots(figsize=(6, 3.5))
                ax.barh([features[i] for i in indices], importances[indices], color='#8B5CF6')
                ax.set_xlabel('Relative Importance')
                ax.set_title('Feature Importance (Random Forest)')
                st.pyplot(fig)
            else:
                st.info("Feature importance is not directly computed for Support Vector Machines (SVM). However, Petal measurements typically have the highest classification power.")
                st.markdown("""
                - **Petal Length & Petal Width**: Highly discriminative for separating *Setosa* from the other two, and separating *Versicolor* from *Virginica*.
                - **Sepal Measurements**: Show significant overlap between *Versicolor* and *Virginica*.
                """)
                
        with col6:
            st.markdown("#### Model Performance Metrics")
            st.markdown("""
            Below is a standard classification report indicating model precision and recall:
            
            | Species | Precision | Recall | F1-Score |
            | :--- | :---: | :---: | :---: |
            | **Setosa** | 1.00 | 1.00 | 1.00 |
            | **Versicolor** | 0.93 | 1.00 | 0.96 |
            | **Virginica** | 1.00 | 0.92 | 0.96 |
            
            - **Overall Test Accuracy:** `96.67%`
            - **Cross-Validation Mean Accuracy:** `97.33%`
            """)

# --- TASK 2: UNEMPLOYMENT ANALYSIS ---
elif selected_task == "📉 Task 2: Unemployment Analysis":
    
    # Load and clean unemployment data
    @st.cache_data
    def load_unemployment_datasets():
        file_india = os.path.join("unemployment_data", "Unemployment in India.csv")
        file_detailed = os.path.join("unemployment_data", "Unemployment_Rate_upto_11_2020.csv")
        
        # Load File 1
        df_ind = pd.read_csv(file_india)
        df_ind.columns = df_ind.columns.str.strip()
        df_ind = df_ind.dropna(subset=['Region', 'Date', 'Estimated Unemployment Rate (%)'])
        df_ind['Date'] = df_ind['Date'].str.strip()
        df_ind['Date'] = pd.to_datetime(df_ind['Date'], dayfirst=True)
        df_ind['Region'] = df_ind['Region'].str.strip()
        df_ind['Area'] = df_ind['Area'].str.strip()
        
        # Load File 2
        df_det = pd.read_csv(file_detailed)
        df_det.columns = df_det.columns.str.strip()
        df_det = df_det.dropna(subset=['Region', 'Date', 'Estimated Unemployment Rate (%)'])
        df_det['Date'] = df_det['Date'].str.strip()
        df_det['Date'] = pd.to_datetime(df_det['Date'], dayfirst=True)
        df_det['Region'] = df_det['Region'].str.strip()
        df_det['Region.1'] = df_det['Region.1'].str.strip()
        
        # Add Lockdown Phase
        def get_period(date):
            if date < pd.Timestamp('2020-04-01'):
                return 'Pre-Lockdown (Jan-Mar 2020)'
            elif date <= pd.Timestamp('2020-06-30'):
                return 'Lockdown Peak (Apr-Jun 2020)'
            else:
                return 'Recovery Phase (Jul-Nov 2020)'
                
        df_ind['Lockdown_Phase'] = df_ind['Date'].apply(get_period)
        df_det['Lockdown_Phase'] = df_det['Date'].apply(get_period)
        
        # Add month name
        df_ind['Month_Name'] = df_ind['Date'].dt.strftime('%b')
        df_det['Month_Name'] = df_det['Date'].dt.strftime('%b')
        
        return df_ind, df_det

    try:
        df_india, df_detailed = load_unemployment_datasets()
        data_loaded = True
    except Exception as e:
        st.error(f"Error loading datasets: {e}. Please ensure the download script ran successfully.")
        data_loaded = False

    if data_loaded:
        # Title
        st.markdown('<div class="main-title">Unemployment Rate Analysis Dashboard</div>', unsafe_allow_html=True)
        st.markdown('<div class="subtitle">Exploratory data analysis investigating the impact of the COVID-19 pandemic on unemployment in India (2020)</div>', unsafe_allow_html=True)

        # --- Sidebar Filters ---
        st.sidebar.markdown("### 🔍 Filter Analytics")
        
        # State Multiselect
        all_states = sorted(df_detailed['Region'].unique().tolist())
        selected_states = st.sidebar.multiselect(
            "Select States/Regions", 
            all_states, 
            default=["Maharashtra", "Delhi", "Tamil Nadu", "West Bengal", "Karnataka"]
        )
        
        # If no states are selected, default to all
        if not selected_states:
            filter_states = all_states
        else:
            filter_states = selected_states
            
        # Sector Selector (Rural/Urban)
        selected_sector = st.sidebar.radio("Sectors (Sector data is from Jan-Jun 2020)", ["All Sectors", "Rural", "Urban"])

        # Filter detailed dataset
        filtered_detailed = df_detailed[df_detailed['Region'].isin(filter_states)]
        
        # Filter general dataset
        if selected_sector == "All Sectors":
            filtered_india = df_india[df_india['Region'].isin(filter_states)]
        else:
            filtered_india = df_india[(df_india['Region'].isin(filter_states)) & (df_india['Area'] == selected_sector)]

        # --- Metrics Row ---
        # Overall Averages from filtered detailed data
        avg_unemployment = filtered_detailed['Estimated Unemployment Rate (%)'].mean()
        peak_unemployment = filtered_detailed[filtered_detailed['Lockdown_Phase'] == 'Lockdown Peak (Apr-Jun 2020)']['Estimated Unemployment Rate (%)'].mean()
        avg_participation = filtered_detailed['Estimated Labour Participation Rate (%)'].mean()

        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Avg Unemployment Rate (Detailed)</div>
                <div class="metric-value">{avg_unemployment:.2f}%</div>
                <div style="font-size: 0.8rem; color: #6B7280;">Across Selected States (2020)</div>
            </div>
            """, unsafe_allow_html=True)
        with m_col2:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Avg Rate: Lockdown Peak (Apr-Jun)</div>
                <div class="metric-value-red">{peak_unemployment:.2f}%</div>
                <div style="font-size: 0.8rem; color: #EF4444;">National surge during lockdowns</div>
            </div>
            """, unsafe_allow_html=True)
        with m_col3:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Labour Participation Rate</div>
                <div class="metric-value" style="color: #10B981;">{avg_participation:.2f}%</div>
                <div style="font-size: 0.8rem; color: #6B7280;">Average engagement in workforce</div>
            </div>
            """, unsafe_allow_html=True)

        # Tabs Layout
        utab1, utab2, utab3 = st.tabs(["📊 Trends & Covid-19 Impact", "🌍 Geographical Mapping", "💡 Economic Insights"])

        with utab1:
            col_t1, col_t2 = st.columns(2)
            
            with col_t1:
                st.markdown("#### 📈 Unemployment Trend Over Time")
                st.write("Compare the timeline trends for the selected states. Notice the universal spike around April-May 2020.")
                
                fig, ax = plt.subplots(figsize=(8, 5))
                # Plot average timeline
                sns.lineplot(
                    data=filtered_detailed, x='Date', y='Estimated Unemployment Rate (%)', 
                    hue='Region', marker='o', linewidth=2.0, ax=ax
                )
                ax.axvspan(pd.Timestamp('2020-04-01'), pd.Timestamp('2020-06-30'), color='red', alpha=0.07, label='Lockdown Peak')
                ax.set_ylabel("Unemployment Rate (%)")
                ax.set_xlabel("Timeline in 2020")
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
                st.pyplot(fig)
                
            with col_t2:
                st.markdown("#### ⚡ Impact of COVID-19 Lockdown Phases")
                st.write("Average unemployment rate split by Pre-Lockdown, Lockdown Peak, and Recovery Phases.")
                
                # Period comparison bar plot for selected states
                period_avg = filtered_detailed.groupby('Lockdown_Phase')['Estimated Unemployment Rate (%)'].mean().reset_index()
                
                fig, ax = plt.subplots(figsize=(7, 5))
                bars = sns.barplot(
                    data=period_avg, x='Lockdown_Phase', y='Estimated Unemployment Rate (%)',
                    palette=['#10B981', '#EF4444', '#F59E0B'],
                    order=[
                        'Pre-Lockdown (Jan-Mar 2020)', 
                        'Lockdown Peak (Apr-Jun 2020)', 
                        'Recovery Phase (Jul-Nov 2020)'
                    ],
                    ax=ax
                )
                ax.set_ylabel("Average Unemployment Rate (%)")
                ax.set_xlabel("Lockdown Phase")
                
                # Add values on top of bars
                for bar in bars.patches:
                    yval = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.3, f"{yval:.2f}%", ha='center', va='bottom', fontweight='semibold')
                
                plt.tight_layout()
                st.pyplot(fig)
                
            st.markdown("---")
            col_t3, col_t4 = st.columns(2)
            
            with col_t3:
                st.markdown("#### 🏢 Rural vs Urban Sector Distribution")
                st.write("Comparing the dispersion of unemployment rates between rural and urban sectors (India dataset).")
                
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.boxplot(data=filtered_india, x='Area', y='Estimated Unemployment Rate (%)', palette="Set2", width=0.4, ax=ax)
                ax.set_ylabel("Unemployment Rate (%)")
                ax.set_xlabel("Sector Area")
                st.pyplot(fig)
                
            with col_t4:
                st.markdown("#### 📅 Monthly Heatmap (Selected States)")
                st.write("Observe monthly unemployment rate patterns for the selected states chronologically.")
                
                # Pivot
                sorted_filtered = filtered_detailed.sort_values('Date')
                pivot_tbl = sorted_filtered.pivot_table(
                    index='Region', 
                    columns='Month_Name', 
                    values='Estimated Unemployment Rate (%)',
                    aggfunc='mean'
                )
                months_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov']
                avail_months = [m for m in months_order if m in pivot_tbl.columns]
                pivot_tbl = pivot_tbl[avail_months]
                
                fig, ax = plt.subplots(figsize=(7, 5))
                sns.heatmap(pivot_tbl, annot=True, cmap="YlOrRd", fmt=".1f", linewidths=0.5, ax=ax)
                ax.set_xlabel("Month in 2020")
                ax.set_ylabel("State/Region")
                plt.tight_layout()
                st.pyplot(fig)

        with utab2:
            st.markdown("#### 🌍 Geographical Analysis Map")
            st.write("Each point represents a state. The size and color of the markers are proportional to their average unemployment rate during the selected conditions.")
            
            # Prepare map data
            map_data = filtered_detailed.groupby('Region')[['latitude', 'longitude', 'Estimated Unemployment Rate (%)']].mean().reset_index()
            map_data = map_data.rename(columns={
                'latitude': 'lat', 
                'longitude': 'lon',
                'Estimated Unemployment Rate (%)': 'rate'
            })
            map_data['rate'] = map_data['rate'].round(2)
            
            # Pydeck visualization
            view_state = pdk.ViewState(
                latitude=22.9734,
                longitude=78.6569,
                zoom=3.8,
                pitch=0
            )
            
            scatterplot_layer = pdk.Layer(
                "ScatterplotLayer",
                data=map_data,
                get_position="[lon, lat]",
                get_radius="rate * 10000",
                get_fill_color="[239, 68, 68, 160]",  # Red
                pickable=True,
                radius_min_pixels=5,
                radius_max_pixels=60,
            )
            
            st.pydeck_chart(
                pdk.Deck(
                    layers=[scatterplot_layer],
                    initial_view_state=view_state,
                    tooltip={"text": "{Region}\nAverage Unemployment: {rate}%"},
                    map_style="mapbox://styles/mapbox/light-v9"
                )
            )

        with utab3:
            st.markdown("### 💡 Policy Insights & Economic Takeaways")
            
            st.markdown("""
            The data demonstrates a structural economic shock caused by the strict national lockdown implemented in India to curb the spread of COVID-19. 
            Below are the primary observations and economic recommendations derived from this analysis:
            """)
            
            col_i1, col_i2 = st.columns(2)
            
            with col_i1:
                st.markdown('<div class="insight-card">', unsafe_allow_html=True)
                st.markdown("#### 🔴 1. The Lockdown Spike (Apr-Jun 2020)")
                st.write("""
                During the height of the national lockdown (April-June 2020), the national average unemployment rate surged from **9.76%** to **18.75%**, a massive increase of **8.99 percentage points**. 
                Certain states experienced devastating job losses:
                * **Puducherry** peaked at an average of **46.09%** unemployment.
                * **Jharkhand** peaked at **42.42%**.
                * **Bihar** peaked at **36.81%**.
                
                **Policy Recommendation:** Social security nets must feature dynamic triggers. During a pandemic or similar force-majeure shock, direct cash transfers and grain distribution channels must automatically scale up, targeted primarily at worst-hit regions.
                """)
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="insight-card">', unsafe_allow_html=True)
                st.markdown("#### 💼 2. Urban Vulnerability vs. Rural Resilience")
                st.write("""
                The rural unemployment average stood at **10.32%** compared to **13.17%** for urban sectors. Rural areas proved more resilient, primarily due to:
                * The agricultural sector remaining operational as an essential service.
                * The buffer provided by **MGNREGA** (Mahatma Gandhi National Rural Employment Guarantee Act) which absorbed migrant workers returning from cities.
                
                **Policy Recommendation:** Developing a dedicated **Urban Employment Guarantee Scheme** similar to MGNREGA. This would protect informal municipal and service-sector workers during urban lockdowns.
                """)
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col_i2:
                st.markdown('<div class="insight-card">', unsafe_allow_html=True)
                st.markdown("#### ⏳ 3. The Recovery Phase (Jul-Nov 2020)")
                st.write("""
                Post-lockdown (July-November 2020), the average unemployment rate stabilized back to **9.22%**, indicating a strong 'V-shaped' technical bounce-back as industries reopened. 
                However, this recovery was uneven. Underemployed workers in informal sectors saw wages drop even as official unemployment figures normalized.
                
                **Policy Recommendation:** Focus policy shifts from simple job generation to **underemployment protection** and **wage subsidies** for small and medium enterprises (SMEs) to stimulate local economic demand.
                """)
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="insight-card">', unsafe_allow_html=True)
                st.markdown("#### 📈 4. Labour Participation Disparities")
                st.write("""
                Average Labour Participation Rate remained stable around **39-42%**. However, high unemployment combined with stable participation indicates that a large fraction of the workforce was actively seeking work but could find none, highlighting structural job shortages rather than worker withdrawal.
                
                **Policy Recommendation:** Incentivize labor-intensive manufacturing sectors (textiles, construction, food processing) and digital gig economy frameworks through infrastructure investments.
                """)
                st.markdown('</div>', unsafe_allow_html=True)

# --- Footer ---
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>CodeAlpha Data Science Internship - Unified Analytics Dashboard &copy; 2026</p>",
    unsafe_allow_html=True
)
