import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import openai
import os
import google.generativeai as genai
import time
from sklearn.linear_model import LinearRegression
import numpy as np
import re
import requests
# Set API keys securely
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Set page config
st.set_page_config(page_title="Planet at Risk", layout="wide", page_icon="favicon.ico")

# Custom CSS
st.markdown("""
    <style>
    html, body, [class*="css"] {
        background-color: #000000;
        color: #ffffff;
    }
    .animated-title {
        text-align: center;
        color: #2E8B57;
        font-size: 4em;
        font-weight: bold;
        animation: pulse 2s infinite, glow 2s ease-in-out infinite alternate;
        text-decoration: underline;
        text-underline-offset: 10px;
        cursor: pointer;
    }
    @keyframes pulse {
        0% { color: #2E8B57; }
        50% { color: #3CB371; }
        100% { color: #2E8B57; }
    }
    @keyframes glow {
        from {
            text-shadow: 0 0 10px #2E8B57, 0 0 20px #2E8B57, 0 0 30px #3CB371, 0 0 40px #3CB371;
        }
        to {
            text-shadow: 0 0 20px #3CB371, 0 0 30px #2E8B57, 0 0 40px #2E8B57, 0 0 50px #3CB371;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <h1 class='animated-title'>
        🌍 Planet at Risk
    </h1>
""", unsafe_allow_html=True)

# Define file paths
data_dir = "data"
temp_path = f"{data_dir}/GLB.Ts+dSST.csv"
disasters_path = f"{data_dir}/number-of-natural-disaster-events.csv"
forest_path = f"{data_dir}/annual-change-forest-area.csv"
co2_path = f"{data_dir}/annual-co2-emissions-per-country.csv"
glacier_path = f"{data_dir}/mass-us-glaciers.csv"

# Load data
def load_data(path, **kwargs):
    try:
        return pd.read_csv(path, **kwargs)
    except Exception as e:
        st.warning(f"⚠️ Could not load file: {path}. Error: {e}")
        return None

# Plot helper
def seaborn_lineplot(df, x, y, title, xlabel, ylabel, color="blue"):
    fig, ax = plt.subplots(figsize=(10, 5), facecolor='black')
    ax.set_facecolor('black')
    sns.set_style("darkgrid", {"axes.facecolor": "black"})
    sns.lineplot(data=df, x=x, y=y, ax=ax, color=color)
    ax.set_title(title, fontsize=18, weight='bold', color='white')
    ax.set_xlabel(xlabel, color='white')
    ax.set_ylabel(ylabel, color='white')
    ax.tick_params(axis='x', colors='white', labelsize=10)
    ax.tick_params(axis='y', colors='white', labelsize=10)
    st.pyplot(fig)

# Load datasets
temp_df = load_data(temp_path, skiprows=1)
disasters_df = load_data(disasters_path)
forest_df = load_data(forest_path)
co2_df = load_data(co2_path)
glacier_df = load_data(glacier_path)

# Tabs for Navigation
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🌍 Dashboard", "📢 Awareness & Solutions", "🤖 Ask Planet AI", "🔮 Disaster Forecast", "📚 Credits"])

# ----------------- 🌍 Dashboard -------------------
with tab1:
    st.sidebar.markdown("""
    ### 📊 Dashboard Features
    - 🌡️ Global Temperature Anomalies
    - 🌪️ Natural Disaster Events
    - 🌳 Forest Area Change
    - 💨 CO₂ Emissions
    - 🧊 Glacier Mass Loss
    """)

    show_temp = st.sidebar.checkbox("🌡️ Show Temperature Anomalies", True)
    show_disasters = st.sidebar.checkbox("🌪️ Show Natural Disasters", True)
    show_forest = st.sidebar.checkbox("🌳 Show Forest Area Change", True)
    show_co2 = st.sidebar.checkbox("💨 Show CO₂ Emissions", True)
    show_glacier = st.sidebar.checkbox("🧊 Show Glacier Mass Loss", True)

    if show_temp and temp_df is not None:
        try:
            temp_df.rename(columns={temp_df.columns[0]: "Year"}, inplace=True)
            temp_df = temp_df.drop(columns=[col for col in temp_df.columns if "J-D" not in col and col != "Year"], errors='ignore')
            temp_df['Year'] = pd.to_numeric(temp_df['Year'], errors='coerce')
            temp_df['Anomaly (°C)'] = pd.to_numeric(temp_df['J-D'], errors='coerce') / 100.0
            temperature_df = temp_df[['Year', 'Anomaly (°C)']].dropna()
            seaborn_lineplot(temperature_df, 'Year', 'Anomaly (°C)', "Global Temperature Anomalies", "Year", "Anomaly (°C)", color="darkorange")
        except Exception as e:
            st.warning(f"⚠️ Error parsing NASA temperature data: {e}")

    if show_disasters and disasters_df is not None:
        filtered_disasters = disasters_df[disasters_df['Entity'] == 'All disasters']
        if 'Year' in filtered_disasters.columns and 'Disasters' in filtered_disasters.columns:
            seaborn_lineplot(filtered_disasters, 'Year', 'Disasters', "Natural Disaster Events", "Year", "Number of Disasters", color="crimson")
        else:
            st.warning("⚠️ Required columns missing in disaster data.")

    if show_forest and forest_df is not None:
        countries = forest_df['Entity'].unique().tolist()
        selected_country = st.selectbox("🌳 Select Country for Forest Area Change", countries)
        country_forest = forest_df[forest_df['Entity'] == selected_country]
        if 'Year' in country_forest.columns and 'Annual net change in forest area' in country_forest.columns:
            seaborn_lineplot(country_forest, 'Year', 'Annual net change in forest area', f"Forest Area Change in {selected_country}", "Year", "Forest Change (hectares)", color="forestgreen")

    if show_co2 and co2_df is not None:
        co2_countries = co2_df['Entity'].unique().tolist()
        selected_co2_country = st.selectbox("💨 Select Country for CO₂ Emissions", co2_countries)
        country_co2 = co2_df[co2_df['Entity'] == selected_co2_country]
        if 'Year' in country_co2.columns and 'Annual CO₂ emissions' in country_co2.columns:
            seaborn_lineplot(country_co2, 'Year', 'Annual CO₂ emissions', f"CO₂ Emissions in {selected_co2_country}", "Year", "Emissions (tonnes)", color="#CCCCCC")

    if show_glacier and glacier_df is not None:
        glacier_names = glacier_df['Entity'].unique().tolist()
        selected_glacier = st.selectbox("🧊 Select Glacier for Mass Loss", glacier_names)
        glacier_data = glacier_df[glacier_df['Entity'] == selected_glacier]
        if 'Year' in glacier_data.columns and 'Cumulative mass balance' in glacier_data.columns:
            seaborn_lineplot(glacier_data, 'Year', 'Cumulative mass balance', f"Mass Loss - {selected_glacier}", "Year", "Cumulative Mass Loss (Gt)", color="deepskyblue")

# ----------------- 📢 Awareness & Solutions -------------------
with tab2:
    st.header("📢 Awareness and Ways to Help 🌎")
    st.markdown("""
    **How You Can Help Fight Climate Change:**
    - 🚲 Use eco-friendly transport (bike, walk, carpool)
    - 🔌 Reduce energy consumption (turn off lights, efficient appliances)
    - 🌱 Plant more trees
    - 🍽️ Reduce food waste
    - 🛍️ Use reusable bags, bottles, and containers
    - 🗳️ Support climate-positive policies and leaders

    Together, small actions create a huge impact!
    """)

# ----------------- 🤖 Ask Planet AI -------------------




with tab3:
    st.header("🤖 Ask Planet AI about Climate, Earth & Solutions!")
    st.markdown("Feel free to ask anything about climate change, disasters, CO₂, deforestation, OR current temperatures!")

    user_input = st.text_input("Ask your question:")

    if user_input:
        with st.spinner("Thinking... 🌎"):
            try:
                # No more temperature special handling, everything goes to Gemini
                model = genai.GenerativeModel('gemini-2.0-flash-lite')

                system_prompt = (
                    "You are PlanetAI, an expert in climate change, natural disasters, environmental protection, CO₂ emissions, deforestation, "
                    "and sustainability. Only answer questions related to these topics. "
                    "If the user's question is irrelevant (like gaming, politics, or gossip), politely reply: "
                    "'I'm here to assist only with climate, environment, and Earth-related topics! 🌍' "
                    "Here is the user's question:\n"
                    f"{user_input}"
                )

                response = model.generate_content(system_prompt)
                full_response = response.text

                output_placeholder = st.empty()
                displayed_text = ""
                cursor_visible = True

                for char in full_response:
                    displayed_text += char
                    if cursor_visible:
                        output_placeholder.markdown(f"🧠 {displayed_text}|")
                    else:
                        output_placeholder.markdown(f"🧠 {displayed_text} ")
                    cursor_visible = not cursor_visible
                    time.sleep(0.02)

                output_placeholder.markdown(f"🧠 {displayed_text}")
                st.success("Done! ✅")

            except Exception as e:
                st.error(f"⚠️ Error: {e}")




# ----------------- 🔮 Disaster Forecast -------------------
with tab4:
    st.header("🔮 Disaster Forecast for the Future")

    if disasters_df is not None:
        st.subheader("📈 Projected Number of Natural Disasters (2025–2040)")
        
        filtered_disasters = disasters_df[disasters_df['Entity'] == 'All disasters']
        X = filtered_disasters['Year'].values.reshape(-1,1)
        y = filtered_disasters['Disasters'].values

        model = LinearRegression()
        model.fit(X, y)

        future_years = np.arange(2025, 2041).reshape(-1,1)
        future_preds = model.predict(future_years)

        forecast_df = pd.DataFrame({'Year': future_years.flatten(), 'Predicted Disasters': future_preds})
        seaborn_lineplot(forecast_df, 'Year', 'Predicted Disasters', "Projected Natural Disasters", "Year", "Predicted Number of Disasters", color="violet")

    with st.spinner("Generating future forecast..."):
        try:
            model = genai.GenerativeModel('gemini-2.0-flash-lite')
            prompt = "Predict the trend of global natural disasters between 2025 and 2040 considering rising climate change impacts. Give a short futuristic summary."
            response = model.generate_content(prompt)
            forecast_text = response.text
            st.success("Forecast ready! ✅")
            st.markdown(f"📝 **AI Prediction Summary:**\n\n{forecast_text}")
        except Exception as e:
            st.error(f"⚠️ Gemini API error: {e}")

# ----------------- 📚 Credits -------------------
with tab5:
    st.header("📚 Credits")
    st.markdown("""
    - **Global Temperature Data**: NASA GISTEMP v4
    - **Natural Disaster Data**: Our World in Data
    - **Forest Area Change**: Our World in Data
    - **CO₂ Emissions**: Our World in Data
    - **Glacier Mass Loss**: Our World in Data

    *This dashboard is built for educational awareness. 🌏 Made with ❤️ by Abhimanyu & Abeer.*
    """)
