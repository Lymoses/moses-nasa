import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt

# Streamlit application starts here
st.set_page_config(page_title="Water Level Visualization Tool", layout="wide")
st.title("🌊 Water Level Visualization Tool for Farmers")

# Sidebar for Area Selection (always visible)
st.sidebar.header("Settings")
area = st.sidebar.selectbox("Select Area", ["Nairobi, Kenya", "Dar es Salaam, Tanzania", "Kampala, Uganda", "Addis Ababa, Ethiopia", "Bujumbura, Burundi", "Juba, South Sudan"])

# Create a dummy dataset for demonstration
np.random.seed(42)  # For reproducibility
data = {
    "Year": np.arange(2000, 2031),  # Years from 2000 to 2030
    "Water_Level": np.random.randint(50, 150, size=31)  # Random water levels
}
df = pd.DataFrame(data)

# Date Picker for selecting specific date
selected_date = st.sidebar.date_input("Select a Date", pd.to_datetime("2024-10-01"))

# Extracting month and year from the selected date
selected_month = selected_date.month
selected_year = selected_date.year

# Year Range Slider (for historical data)
start_year = int(df['Year'].min())
end_year = int(df['Year'].max())

start_year, end_year = st.sidebar.slider("Select Year Range", start_year, end_year, (start_year, end_year))

# Statistics section
st.sidebar.subheader("Statistics")
average_water_level = df['Water_Level'].mean()
max_water_level = df['Water_Level'].max()
min_water_level = df['Water_Level'].min()

st.sidebar.write(f"**Average Water Level**: {average_water_level:.2f}")
st.sidebar.write(f"**Max Water Level**: {max_water_level}")
st.sidebar.write(f"**Min Water Level**: {min_water_level}")

# Predict water levels on button click
def predict_water_levels(start_year, end_year):
    future_years = np.arange(start_year, end_year + 1)

    # Ensure the data is filtered by the selected year range
    filtered_df = df[(df['Year'] >= start_year) & (df['Year'] <= end_year)]
    
    # Check if filtered_df is not empty before fitting the model
    if not filtered_df.empty:
        # Predict water levels
        model = LinearRegression()
        model.fit(filtered_df["Year"].values.reshape(-1, 1), filtered_df["Water_Level"])
        predictions = model.predict(future_years.reshape(-1, 1))
        return predictions, filtered_df
    return None, None

# Get predictions for the selected year range
predictions, filtered_df = predict_water_levels(start_year, end_year)

# Real-time prediction for the selected day
if not df.empty and not filtered_df.empty:
    current_day_prediction = np.nan
    if selected_year in filtered_df['Year'].values:
        current_day_prediction = filtered_df.loc[filtered_df['Year'] == selected_year, 'Water_Level'].values[0]
    else:
        # If the year is outside the historical range, predict using the model
        model = LinearRegression()
        model.fit(filtered_df["Year"].values.reshape(-1, 1), filtered_df["Water_Level"])
        current_day_prediction = model.predict(np.array([[selected_year]]))[0]

    st.subheader(f"Predicted Water Level for {selected_date.strftime('%Y-%m-%d')}: {current_day_prediction:.2f}")

    # Create the visualization for historical and predicted water levels
    fig = go.Figure()

    # Add historical water levels to the graph
    fig.add_trace(go.Scatter(
        x=filtered_df["Year"],
        y=filtered_df["Water_Level"],
        mode='lines+markers',
        name='Historical Water Level',
        line=dict(color='blue', width=2)
    ))

    # Add predicted water levels to the graph
    if predictions is not None:
        fig.add_trace(go.Scatter(
            x=np.arange(start_year, end_year + 1),
            y=predictions,
            mode='lines+markers',
            name='Predicted Water Level',
            line=dict(color='green', width=2)
        ))

    # Update layout of the figure
    fig.update_layout(
        title=f"Water Levels in {area} from {start_year} to {end_year}",
        xaxis_title='Year',
        yaxis_title='Water Level',
        plot_bgcolor='lightgrey',
        legend=dict(x=0, y=1, traceorder='normal'),
        font=dict(size=12),
        height=400
    )

    # Display the graph
    st.plotly_chart(fig)

    # Yearly Change Visualization
    st.subheader("Yearly Change in Water Levels")

    # Calculate year-on-year changes
    filtered_df['Change'] = filtered_df['Water_Level'].diff()
    filtered_df['Change_Color'] = np.where(filtered_df['Change'] < 0, 'red', 'blue')

    # Create a bar chart for changes
    bar_fig = go.Figure()
    bar_fig.add_trace(go.Bar(
        x=filtered_df["Year"][1:],  # Exclude the first year for change calculation
        y=filtered_df['Change'][1:],
        marker_color=filtered_df['Change_Color'][1:],  # Color bars based on increase/decrease
        name='Water Level Change'
    ))

    # Update layout for the bar chart
    bar_fig.update_layout(
        title=f"Water Level Change in {area}",
        xaxis_title='Year',
        yaxis_title='Change in Water Level',
        plot_bgcolor='lightgrey',
        height=400
    )

    # Display the bar chart
    st.plotly_chart(bar_fig)

    # Color-coded boxes to indicate water level changes
    st.subheader("Water Level Change Categories")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"<div style='background-color: lightblue; padding: 20px; border-radius: 5px; text-align: center;'>"
            f"<strong>Increase</strong><br>Water level increased</div>",
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            f"<div style='background-color: lightcoral; padding: 20px; border-radius: 5px; text-align: center;'>"
            f"<strong>Decrease</strong><br>Water level decreased</div>",
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"<div style='background-color: lightgreen; padding: 20px; border-radius: 5px; text-align: center;'>"
            f"<strong>Significant Increase</strong><br>Water level increased significantly</div>",
            unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            f"<div style='background-color: orange; padding: 20px; border-radius: 5px; text-align: center;'>"
            f"<strong>Significant Decrease</strong><br>Water level decreased significantly</div>",
            unsafe_allow_html=True
        )

    # Remote sensing style heatmap visualization
    st.subheader("Water Level Heatmap")
    
    # Create a heatmap data
    heatmap_data = np.random.rand(12, 6) * 100  # Simulate data for each month across different areas
    heatmap_df = pd.DataFrame(heatmap_data, columns=["Nairobi", "Dar es Salaam", "Kampala", "Addis Ababa", "Bujumbura", "Juba"], index=[f"Month {i+1}" for i in range(12)])

    # Create the heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_df, annot=True, cmap="coolwarm", linewidths=.5)
    plt.title("Monthly Water Levels (Heatmap)")
    plt.xlabel("Areas")
    plt.ylabel("Months")
    plt.xticks(rotation=45)
    
    # Show the heatmap in Streamlit
    st.pyplot(plt)

    # Summary report section
    st.subheader("Summary Report")
    
    report_text = (
        f"**Area Selected**: {area}\n\n"
        f"**Date Selected**: {selected_date.strftime('%Y-%m-%d')}\n\n"
        f"**Water Levels Summary (from {start_year} to {end_year})**:\n"
        f"- Average Water Level: {average_water_level:.2f}\n"
        f"- Maximum Water Level: {max_water_level}\n"
        f"- Minimum Water Level: {min_water_level}\n"
        f"- Predicted Water Level on {selected_date.strftime('%Y-%m-%d')}: {current_day_prediction:.2f}\n\n"
        f"**Recommendations**:\n"
        f"- Monitor the changes in water levels monthly to ensure effective water management.\n"
        f"- Implement practices to mitigate significant decreases in water levels.\n"
    )
    
    st.markdown(report_text)
else:
    st.write("No data available to display.")
