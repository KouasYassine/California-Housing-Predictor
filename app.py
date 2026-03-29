import streamlit as st
import pandas as pd
import joblib
import os
import shap
import matplotlib.pyplot as plt
import time

st.set_page_config(page_title="Housing Predictor Pro", layout="wide")

languages = {
    "Français": {
        "main_title": "California Housing AI - Analyse de Valeur",
        "sidebar_header": "Paramètres du Quartier",
        "med_inc": "Revenu Médian (x10k $)",
        "age": "Âge moyen des maisons",
        "rooms": "Nombre moyen de pièces",
        "pop": "Population totale",
        "loc_header": "Localisation",
        "predict_btn": "Estimer le prix",
        "loading": "Analyse des données par l'IA...",
        "result_label": "Prix estimé",
        "why_title": "Pourquoi ce prix ? (Interprétabilité)",
        "help_title": "Aide à l'interprétation",
        "help_content": "Ce modèle utilise l'algorithme **XGBoost** optimisé. L'analyse **SHAP** décompose l'estimation : chaque barre montre l'impact d'une caractéristique par rapport à la moyenne du marché.",
        "caption": "Rouge = Augmente le prix | Bleu = Diminue le prix",
        "features": ['Revenu', 'Âge', 'Pièces', 'Chambres', 'Pop', 'Occupation', 'Lat', 'Long']
    },
    "English": {
        "main_title": "California Housing AI - Value Analysis",
        "sidebar_header": "Neighborhood Parameters",
        "med_inc": "Median Income (x10k $)",
        "age": "Average House Age",
        "rooms": "Average Rooms",
        "pop": "Total Population",
        "loc_header": "Location",
        "predict_btn": "Estimate Price",
        "loading": "AI is analyzing data...",
        "result_label": "Estimated Price",
        "why_title": "Why this price? (Interpretability)",
        "help_title": "Interpretation Help",
        "help_content": "This model uses an optimized **XGBoost** algorithm. The **SHAP** analysis breaks down the estimation: each bar shows how a feature moved the price relative to the market average.",
        "caption": "Red = Increases price | Blue = Decreases price",
        "features": ['Income', 'Age', 'Rooms', 'Bedrooms', 'Pop', 'Occupancy', 'Lat', 'Long']
    }
}

selected_lang = st.sidebar.selectbox("Language / Langue", ["Français", "English"], key="lang_selector")
t = languages[selected_lang]

st.title(t["main_title"])
st.markdown("---")

@st.cache_resource
def load_model():
    return joblib.load('models/housing_model.pkl')

if os.path.exists('models/housing_model.pkl'):
    model = load_model()
    
    st.sidebar.header(t["sidebar_header"])
    
    med_inc = st.sidebar.slider(t["med_inc"], 0.5, 50.0, 3.5, key=f"inc_{selected_lang}")
    house_age = st.sidebar.slider(t["age"], 1, 52, 28, key=f"age_{selected_lang}")
    ave_rooms = st.sidebar.slider(t["rooms"], 1.0, 15.0, 5.0, key=f"rms_{selected_lang}")
    population = st.sidebar.number_input(t["pop"], value=1500, key=f"pop_{selected_lang}")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader(t["loc_header"])
    lat = st.sidebar.slider("Latitude", 32.5, 42.5, 34.0, key=f"lat_{selected_lang}")
    lon = st.sidebar.slider("Longitude", -124.3, -114.3, -118.0, key=f"lon_{selected_lang}")
    
    input_data = pd.DataFrame({
        'MedInc': [med_inc], 
        'HouseAge': [house_age], 
        'AveRooms': [ave_rooms],
        'AveBedrms': [1.0], 
        'Population': [population], 
        'AveOccup': [3.0],
        'Latitude': [lat], 
        'Longitude': [lon]
    })

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader(t['predict_btn'])
        if st.button(t["predict_btn"], key="predict_action"):
            with st.spinner(t["loading"]):
                time.sleep(0.4) 
                prediction = model.predict(input_data)[0]
            
            st.toast("Done / Terminé")
            st.success(f"### {t['result_label']} : {prediction * 100000:,.0f} $")
            
            st.markdown("---")
            st.subheader(t["why_title"])
            
            try:
                scaler = model.named_steps['scaler']
                regressor = model.named_steps['regressor']
                
                input_scaled = scaler.transform(input_data)
                explainer = shap.TreeExplainer(regressor)
                shap_values = explainer.shap_values(input_scaled)

                fig, ax = plt.subplots(figsize=(8, 5))
                features_display = t["features"]
                
                colors = ['#ff4b4b' if x > 0 else '#0068c9' for x in shap_values[0]]
                impact_series = pd.Series(shap_values[0], index=features_display).sort_values()
                
                ax.barh(impact_series.index, impact_series.values, color=[colors[features_display.index(i)] for i in impact_series.index])
                ax.set_xlabel("Impact")
                ax.grid(axis='x', linestyle='--', alpha=0.7)
                
                st.pyplot(fig)
                st.caption(t["caption"])
            except Exception as e:
                st.error(f"SHAP Error: {e}")

    with col2:
        st.subheader(t["help_title"])
        st.info(t["help_content"])
        
        st.write("**Technical Details / Détails techniques:**")
        st.code(f"""
Model: XGBoost Regressor
Language: {selected_lang}
Explanation: SHAP TreeExplainer
        """)

else:
    st.error("Error: models/housing_model.pkl not found.")