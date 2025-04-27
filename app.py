import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import openai
import os

# Set page config
st.set_page_config(page_title="Planet at Risk", layout="wide", page_icon="🌍")

# OpenAI Client Initialization
openai_client = st.secrets["OPENAI_API_KEY"]

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

# Load data paths
data_dir = "data"
temp_path = f"{data_dir}/GLB.Ts+dSST.csv"
disasters_path = f"{data_dir}/number-of-natural-disaster-events.csv"
forest_path = f"{data_dir}/annual-change-forest-area.csv"
co2_path = f"{data_dir}/annual-co2-emissions-per-country.csv"
glacier_path = f"{data_dir}/mass-us-glaciers.csv"

# Load datasets
def load_data(path, **kwargs):
    try:
        return pd.read_csv(path, **kwargs)
    except Exception as e:
        st.warning(f"⚠️ Could not load {path}. Error: {e}")
        return None

# AI Assistant
def ask_ai(prompt):
    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant specializing in climate change and environmental issues."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ Error contacting AI: {e}"

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

# Animated Title
st.markdown("""
    <h1 class='animated-title'>
        🌍 Planet at Risk
    </h1>
""", unsafe_allow_html=True)

# Sidebar Navigation
page = st.sidebar.radio("Navigate", ("🌍 Dashboard", "📢 Awareness & Solutions", "📚 Credits"))

# Load all datasets
temp_df = load_data(temp_path, skiprows=1)
disasters_df = load_data(disasters_path)
forest_df = load_data(forest_path)
co2_df = load_data(co2_path)
glacier_df = load_data(glacier_path)

# Pages
if page == "🌍 Dashboard":
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
            st.warning(f"⚠️ Error parsing temperature data: {e}")

    if show_disasters and disasters_df is not None:
        filtered_disasters = disasters_df[disasters_df['Entity'] == 'All disasters']
        if 'Year' in filtered_disasters.columns and 'Disasters' in filtered_disasters.columns:
            seaborn_lineplot(filtered_disasters, 'Year', 'Disasters', "Natural Disaster Events", "Year", "Number of Disasters", color="crimson")
        else:
            st.warning("⚠️ Columns missing in disasters dataset.")

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

elif page == "📢 Awareness & Solutions":
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

    st.subheader("💬 Ask AI for Climate Advice:")
    user_question = st.text_input("Ask anything about fighting climate change:")
    if st.button("Ask AI"):
        if user_question:
            ai_response = ask_ai(user_question)
            st.success(ai_response)
        else:
            st.warning("Please enter a question.")

elif page == "📚 Credits":
    st.header("📚 Credits")
    st.markdown("""
    - **Global Temperature Data**: NASA GISTEMP v4
    - **Natural Disaster Data**: Our World in Data
    - **Forest Area Change**: Our World in Data
    - **CO₂ Emissions**: Our World in Data
    - **Glacier Mass Loss**: Our World in Data

    *This dashboard is built for educational awareness. 🌏 Made with ❤️ by Abhimanyu & Abeer.*
    """)

