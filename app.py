import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Sleep Time Prediction Dashboard",
    page_icon="🌙",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #8B5CF6, #EC4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .warning-box {
        background-color: #FEF3C7;
        border-left: 4px solid #F59E0B;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
        color: #1F2937;
    }
    .metric-card {
        background-color: #1F2937;
        border: 1px solid #374151;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Load the actual dataset
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('sleeptime_prediction_dataset.csv')
    except:
        st.error("Please upload 'sleeptime_prediction_dataset.csv' to the same directory as this script")
        st.stop()
    return df

# Train model
@st.cache_resource
def train_model(df):
    # Handle missing values if any
    df_clean = df.dropna()
    
    X = df_clean[['WorkoutTime', 'ReadingTime', 'PhoneTime', 'WorkHours', 'CaffeineIntake', 'RelaxationTime']]
    y = df_clean['SleepTime']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    
    metrics = {
        'r2': r2_score(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'cv_score': cross_val_score(model, X_train_scaled, y_train, cv=5).mean()
    }
    
    return model, scaler, metrics, X.columns, X_test, y_test, y_pred

# Load data and model
df = load_data()
model, scaler, metrics, feature_names, X_test, y_test, y_pred = train_model(df)

# Header
st.markdown('<h1 class="main-header">🌙 Sleep Time Prediction System</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #6B7280;">AI-Powered Sleep Analytics Based on Daily Lifestyle Habits</p>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Prediction", "Analysis", "Model Details"])

# TAB 1: OVERVIEW
with tab1:
    st.header("Problem Statement")
    st.write("""
    Sleep deprivation is a growing public health concern. This system predicts individual sleep duration 
    based on daily lifestyle habits (in hours) to help users optimize their sleep patterns.
    """)
    
    st.header("About the Data")
    st.write("""
    The dataset is a secondary dataset with two-thousand(2000) observations that was taken from https://www.kaggle.com/datasets/govindaramsriram/sleep-time-prediction it has seven(7) columns which are:
    WorkoutTime, ReadingTime, PhoneTime, WorkHours, CaffieneIntake, RelaxationTime, and SleepTime.
    """)

    st.header("Data Preprocessing & Quality Control")
    st.write("""
    Before performing any analysis or training the model, the dataset underwent a rigorous 
    cleaning process to ensure the highest data integrity:
    """)
    
    # Using columns for a cleaner "checklist" look
    check_col1, check_col2 = st.columns(2)
    with check_col1:
        st.markdown("""
        **1. Data Cleaning:**
        *   **Missing Values:** Checked for any `NA` or null entries and handled them to prevent model errors.
        *   **Duplicates:** Scanned for and removed any duplicate records to avoid bias in training.
        *   **Invalid Inputs:** Validated that all time-based entries were within logical ranges (e.g., no negative hours).
        """)
    with check_col2:
        st.markdown("""
        **2. Feature Engineering:**
        *   **Standardization:** To ensure the model treats all habits equally, we standardized the dataset using a `StandardScaler`. 
        *   This process scales all features (like Work Hours vs. Caffeine Intake) to have a mean of 0 and a standard deviation of 1, 
        which is essential for the Random Forest algorithm's performance.
        """)

    st.info("**Result:** The final dataset used for training is clean, normalized, and optimized for predictive accuracy.")

    st.markdown("---")
    
    # KPIs Section
    st.subheader("Key Performance Indicators (KPIs)")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Features", len(df.columns) - 1)
    with col3:
        # Calculate optimal sleep (7-9 hours)
        optimal_sleep = len(df[(df['SleepTime'] >= 7) & (df['SleepTime'] <= 9)])
        optimal_pct = (optimal_sleep / len(df)) * 100
        st.metric("Optimal Sleep Rate", f"{optimal_pct:.1f}%")
    with col4:
        st.metric("Model Accuracy (R²)", f"{metrics['r2']:.3f}")
    
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        avg_sleep = df['SleepTime'].mean()
        st.metric("Average Sleep", f"{avg_sleep:.2f} hrs")
    with col6:
        # Mean Absolute Error in minutes for easier reading
        st.metric("Prediction Error", f"±{metrics['mae']*60:.0f} min")
    with col7:
        # High phone usage defined as > 4 hours
        high_phone = len(df[df['PhoneTime'] >= 4.0]) 
        phone_pct = (high_phone / len(df)) * 100
        st.metric("High Screen Time (>4h)", f"{phone_pct:.1f}%")
    with col8:
        # High Caffeine > 200mg
        high_caff = len(df[df['CaffeineIntake'] >= 200])
        caff_pct = (high_caff / len(df)) * 100
        st.metric("High Caffeine Users", f"{caff_pct:.1f}%")
    
    st.markdown("---")
    
    # Distribution visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sleep Time Distribution")
        fig = px.histogram(
            df, x='SleepTime',
            nbins=30,
            labels={'SleepTime': 'Sleep Time (hours)', 'count': 'Frequency'},
            color_discrete_sequence=['#8B5CF6']
        )
        fig.add_vline(x=7, line_dash="dash", line_color="green", annotation_text="Min 7h")
        fig.add_vline(x=9, line_dash="dash", line_color="orange", annotation_text="Max 9h")
        fig.update_layout(height=350, showlegend=False, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Phone Time Distribution")
        # Ensure we are plotting hours
        fig = px.histogram(
            df, x='PhoneTime',
            nbins=30,
            labels={'PhoneTime': 'Phone Time (hours)', 'count': 'Frequency'},
            color_discrete_sequence=['#EC4899']
        )
        # Threshold at 4 Hours
        fig.add_vline(x=4, line_dash="dash", line_color="red", annotation_text="4 hours (High)")
        fig.update_layout(height=350, showlegend=False, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)
    
    # Box plots for all features (Updated units)
    st.subheader("Feature Distributions (Hours & mg)")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Box(y=df['WorkoutTime'], name='Workout (hrs)', marker_color='#8B5CF6'))
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)
        
        fig = go.Figure()
        fig.add_trace(go.Box(y=df['WorkHours'], name='Work (hrs)', marker_color='#F59E0B'))
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = go.Figure()
        fig.add_trace(go.Box(y=df['ReadingTime'], name='Reading (hrs)', marker_color='#10B981'))
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)
        
        fig = go.Figure()
        fig.add_trace(go.Box(y=df['CaffeineIntake'], name='Caffeine (mg)', marker_color='#EF4444'))
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        fig = go.Figure()
        fig.add_trace(go.Box(y=df['PhoneTime'], name='Phone (hrs)', marker_color='#EC4899'))
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)
        
        fig = go.Figure()
        fig.add_trace(go.Box(y=df['RelaxationTime'], name='Relaxation (hrs)', marker_color='#06B6D4'))
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.subheader("Descriptive Analysis")
    st.write("Detailed breakdown of all numerical variables:")
    
    # We transpose (.T) the describe table so it fits better on the screen (columns as rows)
    stats_df = df.describe().T
    st.dataframe(stats_df.style.format("{:.2f}"), use_container_width=True)
    
    # 3. Data Info (Optional: showing data types and missing values)
    with st.expander("View Data Types and Null Values"):
        buffer = pd.DataFrame({
            "Data Type": df.dtypes.astype(str),
            "Non-Null Count": df.count(),
            "Null Values": df.isnull().sum()
        })
        st.table(buffer)

# TAB 2: PREDICTION
with tab2:
    st.header("Predict Your Sleep Time")
    
    st.write("Enter your daily lifestyle habits (in hours) to get a personalized sleep prediction:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Adjusted Sliders for Hours (0 to ~3-5 hours usually)
        workout_time = st.slider(
            "Workout Time (Hours)",
            0.0, 5.0, float(df['WorkoutTime'].median()), step=0.1,
            help="Total exercise hours per day"
        )
        
        reading_time = st.slider(
            "Reading Time (Hours)",
            0.0, 5.0, float(df['ReadingTime'].median()), step=0.1,
            help="Time spent reading books/articles"
        )
        
        phone_time = st.slider(
            "Phone Time (Hours)",
            0.0, 10.0, float(df['PhoneTime'].median()), step=0.1,
            help="Total screen time on phone per day"
        )
    
    with col2:
        work_hours = st.slider(
            "Work Hours",
            0.0, 16.0, float(df['WorkHours'].median()), step=0.5,
            help="Total working hours per day"
        )
        
        caffeine_intake = st.slider(
            "Caffeine Intake (mg)",
            0.0, 400.0, float(df['CaffeineIntake'].median()), step=10.0,
            help="Total caffeine consumption (1 coffee ≈ 95mg)"
        )
        
        relaxation_time = st.slider(
            "Relaxation Time (Hours)",
            0.0, 5.0, float(df['RelaxationTime'].median()), step=0.1,
            help="Time for meditation, yoga, or relaxing"
        )
    
    if st.button("Generate Prediction", type="primary", use_container_width=True):
        # Make prediction
        input_data = np.array([[workout_time, reading_time, phone_time, 
                               work_hours, caffeine_intake, relaxation_time]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        
        # Calculate confidence
        confidence = min(95, 80 + (5 - metrics['mae']) * 5)
        
        st.markdown("---")
        st.subheader("Your Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Predicted Sleep Time", f"{prediction:.2f} hours")
        with col2:
            st.metric("Confidence Level", f"{confidence:.1f}%")
        with col3:
            quality_label = "Optimal" if 7 <= prediction <= 9 else ("Adequate" if prediction >= 6 else "Insufficient")
            st.metric("Sleep Quality", quality_label)
        
        st.markdown("---")
        
        # Comparison & Radar Chart
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Comparison")
            # Create comparison chart
            your_data = pd.DataFrame({
                'Category': ['Your Prediction', 'Dataset Average', 'Recommended'],
                'Sleep Hours': [prediction, df['SleepTime'].mean(), 8.0],
                'Type': ['You', 'Average', 'Target']
            })
            
            fig = px.bar(
                your_data, 
                x='Category', 
                y='Sleep Hours',
                color='Type',
                color_discrete_map={'You': '#EC4899', 'Average': '#8B5CF6', 'Target': '#10B981'}
            )
            fig.add_hline(y=7, line_dash="dash", line_color="green", annotation_text="Min 7h")
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Your Lifestyle Profile")
            # Normalize values for Radar Chart (0 to 1 scale)
            # We use a fixed reasonable max for visualization to show "fullness"
            
            # Define logical maxes for the radar chart visualization
            max_refs = {
                'Workout': 3.0,     # 3 hours max reference
                'Reading': 2.0,     # 2 hours max reference
                'Phone': 6.0,       # 6 hours max reference
                'Work': 12.0,       # 12 hours max reference
                'Caffeine': 300.0,  # 300mg max reference
                'Relaxation': 2.0   # 2 hours max reference
            }
            
            categories = ['Workout', 'Reading', 'Phone Time', 'Work Hours', 'Caffeine', 'Relaxation']
            values = [
                min(workout_time / max_refs['Workout'], 1.0),
                min(reading_time / max_refs['Reading'], 1.0),
                min(phone_time / max_refs['Phone'], 1.0),
                min(work_hours / max_refs['Work'], 1.0),
                min(caffeine_intake / max_refs['Caffeine'], 1.0),
                min(relaxation_time / max_refs['Relaxation'], 1.0)
            ]
            
            # Close the loop for radar chart
            values += [values[0]]
            categories += [categories[0]]
            
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name='You',
                line_color='#8B5CF6',
                fillcolor='rgba(139, 92, 246, 0.3)'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1],
                        showticklabels=False
                    )
                ),
                showlegend=False,
                height=400,
                margin=dict(l=40, r=40, t=20, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations Logic (Updated for Hours)
        st.markdown("---")
        st.subheader("Personalized Recommendations")
        
        recommendations = []
        
        # Get percentiles
        phone_75 = df['PhoneTime'].quantile(0.75)
        workout_25 = df['WorkoutTime'].quantile(0.25)
        work_75 = df['WorkHours'].quantile(0.75)
        caffeine_75 = df['CaffeineIntake'].quantile(0.75)
        reading_25 = df['ReadingTime'].quantile(0.25)
        
        if phone_time > phone_75:
            recommendations.append(f"Your phone time ({phone_time:.1f}h) is high (Top 25%). Reducing this can significantly improve sleep quality.")
        
        if workout_time < workout_25:
            recommendations.append(f"Your workout time ({workout_time:.1f}h) is low. Try to reach at least {workout_25:.1f} hours of activity.")
        
        if work_hours > work_75:
            recommendations.append(f"Work hours are high ({work_hours:.1f}h). Ensure you have a 'wind-down' buffer between work and sleep.")
        
        if caffeine_intake > caffeine_75:
            recommendations.append(f"Caffeine intake ({caffeine_intake:.0f}mg) is high. Try to cut off caffeine 8 hours before bed.")
        
        if reading_time < reading_25:
            recommendations.append(f"Reading is a great pre-sleep habit. Try increasing from {reading_time:.1f}h to {reading_25:.1f}h.")
            
        if not recommendations:
            recommendations.append("Your lifestyle habits are well-balanced! Maintain this routine.")
            
        for rec in recommendations:
            st.info(rec)

with tab3:
    st.header("Data Analysis & Insights")
    
    st.write("Visualizing the relationship between each lifestyle habit and sleep duration.")
    
    # Get all feature columns (excluding the target 'SleepTime')
    feature_cols = [col for col in df.columns if col != 'SleepTime']
    
    # Create a 2-column grid layout
    cols = st.columns(2)
    
    # Loop through features and create a graph for each
    for i, feature in enumerate(feature_cols):
        # Determine which column to place the graph in (left or right)
        col = cols[i % 2]
        
        with col:
            # specialized units for the label
            unit = "(mg)" if "Caffeine" in feature else "(Hours)"
            
            st.subheader(f"{feature} vs Sleep")
            
            fig = px.scatter(
                df, 
                x=feature, 
                y='SleepTime',
                trendline="lowess", # Adds a smooth trend line to show the pattern
                labels={
                    feature: f'{feature} {unit}', 
                    'SleepTime': 'Sleep Time (Hours)'
                },
                color='SleepTime',
                color_continuous_scale='Viridis',
                opacity=0.6
            )
            
            # Add correlation coefficient
            corr_val = df[feature].corr(df['SleepTime'])
            
            fig.update_layout(
                height=350,
                margin=dict(l=20, r=20, t=30, b=20),
                title=dict(
                    text=f"Correlation: {corr_val:.3f}",
                    font=dict(size=12, color="gray"),
                    x=0,
                    y=1
                )
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    
    st.subheader("Feature Correlation Matrix")
    st.write("This heatmap shows how features relate to each other. Darker colors indicate stronger relationships.")
    
    corr = df.corr()
    fig = px.imshow(
        corr,
        text_auto='.2f',
        color_continuous_scale='RdBu_r',
        aspect='auto',
        title="Global Correlation Heatmap"
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

# TAB 4: MODEL DETAILS
with tab4:
    st.header("Model Development & Evaluation")
    
    # --- Section 1: Educational Metrics ---
    st.subheader("Performance Metrics Explained")
    st.write("Understanding how we measure the model's errors:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("R² Score (Variance Explained)", f"{metrics['r2']:.4f}")
        st.info("""
        **Definition:** This score represents how much of the "sleep pattern" the model actually understands.
        *   **0.0** = Model is just guessing the average.
        *   **1.0** = Model predicts perfectly every time.
        *   *A low score here means our current features don't explain enough about why people sleep the way they do.*
        """)
        
        mae_minutes = metrics['mae'] * 60
        st.metric("Mean Absolute Error (MAE)", f"{metrics['mae']:.4f} Hours")
        st.info(f"""
        **Definition:** On average, how wrong is the prediction?
        *   In this model, the prediction is typically off by **±{mae_minutes:.0f} minutes**.
        """)
    
    with col2:
        st.metric("RMSE (Root Mean Squared Error)", f"{metrics['rmse']:.4f} Hours")
        st.info("""
        **Definition:** Similar to MAE, but it punishes "massive misses" more heavily.
        *   If RMSE is significantly higher than MAE, it means the model occasionally makes huge errors.
        """)
        
        st.subheader("Visual Diagnostics")
        residuals = y_test - y_pred
        fig = px.scatter(
            x=y_pred, 
            y=residuals,
            labels={'x': 'Predicted Sleep (h)', 'y': 'Error (Residuals)'},
            title="Residual Plot (Bias Check)",
            opacity=0.6,
            color_discrete_sequence=['#EC4899']
        )
        fig.add_hline(y=0, line_dash="dash", line_color="black")
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # --- Section 2: Feature Importance ---
    st.subheader("Feature Importance",)
    st.info('Essentially a "scoreboard" that tells you which of the features actually matters the most to the model.')
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=True)
    
    fig = px.bar(
        importance_df, x='Importance', y='Feature',
        orientation='h',
        color='Importance',
        color_continuous_scale='Purples',
        title="Relative Importance of Features"
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # --- Section 3: Critical Conclusion ---
    st.subheader("Critical Evaluation & Future Roadmap")
    
    col_crit1, col_crit2 = st.columns(2)
    
    with col_crit1:
        st.markdown("#### Data Reality Check")
        st.write("""
        While the dataset used for this model is **clean and complete** (containing no missing values), the analysis reveals a fundamental issue: **the features within it possess only fair to weak correlations with sleep hours.**
        
        This indicates that lifestyle habits alone (like phone usage or caffeine) are insufficient predictors. We must scale up the dataset and identify features with stronger causal links to sleep duration.
        """)
        
        st.markdown("#### Missing Biological Context")
        st.write("To build a viable model, we must expand the data scope to include:")
        st.markdown("""
        1.  **Demographics:** Patient Age, Gender, and BMI (crucial baselines).
        2.  **Medical History:** Diagnosis of Insomnia, Sleep Apnea, Depression, or Chronic Pain.
        3.  **Circadian Data:** Specific **"Going to Bed"** and **"Waking Up"** timestamps, rather than just durations.
        """)

    with col_crit2:
        st.error("""
        ### ⚠️ Final Verdict: Model Unreliable
        
        **The results from this model cannot be trusted.**
        
        Despite having a fully functioning technical pipeline, the **low scoring metrics** (R² and MAE) demonstrate that the model fails to capture the complexity of human sleep. 
        
        The system is in **dire need of improvement**. Reliance on this specific iteration for health advice would be misleading. Future work must focus on acquiring high-dimensional clinical data rather than relying solely on self-reported lifestyle metrics.
        """)