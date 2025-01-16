import pandas as pd
import streamlit as st
import plotly.express as px
import folium
from folium.plugins import HeatMap
from sklearn.linear_model import LinearRegression
import numpy as np

# **Chargement des données**
@st.cache_data
def load_data():
    try:
        yield_history = pd.read_csv("/Users/mac/Documents/projet_agricole/data/historique_rendements.csv", parse_dates=["date"])
        monitoring_data = pd.read_csv("/Users/mac/Documents/projet_agricole/data/monitoring_cultures.csv", parse_dates=["date"])
        weather_data = pd.read_csv("/Users/mac/Documents/projet_agricole/data/meteo_detaillee.csv", parse_dates=["date"])
        soil_data = pd.read_csv("/Users/mac/Documents/projet_agricole/data/sols.csv")

        yield_history["annee"] = yield_history["date"].dt.year
        monitoring_data.drop_duplicates(inplace=True)
        soil_data.drop_duplicates(inplace=True)

        return yield_history, monitoring_data, weather_data, soil_data
    except FileNotFoundError as e:
        st.error(f"Fichier introuvable : {e}")
        st.stop()

# **Visualisation Plotly : Évolution des rendements**
def create_yield_history_plot(yield_history):
    grouped_data = yield_history.groupby("annee")["rendement_final"].mean().reset_index()
    fig = px.line(
        grouped_data,
        x="annee",
        y="rendement_final",
        title="Évolution Historique des Rendements",
        labels={"annee": "Année", "rendement_final": "Rendement (t/ha)"},
    )
    fig.update_traces(mode="lines+markers")
    return fig

# **Visualisation Plotly : Prédiction des rendements**
def create_yield_prediction_plot(yield_history):
    grouped_data = yield_history.groupby("annee")["rendement_final"].mean().reset_index()

    X = grouped_data["annee"].values.reshape(-1, 1)
    y = grouped_data["rendement_final"].values

    model = LinearRegression()
    model.fit(X, y)

    future_years = np.arange(X[-1][0] + 1, X[-1][0] + 6).reshape(-1, 1)
    future_predictions = model.predict(future_years)

    future_data = pd.DataFrame({"annee": future_years.flatten(), "rendement_final": future_predictions})
    combined_data = pd.concat([grouped_data, future_data])

    fig = px.line(
        combined_data,
        x="annee",
        y="rendement_final",
        title="Prédiction des Rendements Futurs",
        labels={"annee": "Année", "rendement_final": "Rendement Prévu (t/ha)"},
    )
    fig.update_traces(mode="lines+markers")
    return fig

# **Visualisation Plotly : Tendance Hebdomadaire NDVI**
def create_ndvi_temporal_plot(monitoring_data):
    monitoring_data["week"] = monitoring_data["date"].dt.to_period("W").apply(lambda r: r.start_time)
    aggregated_data = monitoring_data.groupby("week")["ndvi"].mean().reset_index()

    fig = px.line(
        aggregated_data,
        x="week",
        y="ndvi",
        title="NDVI : Tendance Hebdomadaire",
        labels={"week": "Semaine", "ndvi": "NDVI"},
    )
    fig.update_traces(mode="lines+markers")
    return fig

# **Visualisation Plotly : Matrice de Stress Hydrique**
def create_stress_matrix(monitoring_data):
    fig = px.scatter(
        monitoring_data,
        x="stress_hydrique",
        y="lai",
        color="stress_hydrique",
        size="lai",
        title="Matrice de Stress Hydrique",
        labels={"stress_hydrique": "Stress Hydrique", "lai": "LAI (Indice de Surface Foliaire)"},
        color_continuous_scale="Turbo",
    )
    return fig

# **Carte Folium : Carte de chaleur et informations sur les sols**
def create_map(monitoring_data, soil_data):
    center_lat = monitoring_data["latitude"].mean()
    center_lon = monitoring_data["longitude"].mean()
    folium_map = folium.Map(location=[center_lat, center_lon], zoom_start=12)

    # Carte de chaleur NDVI
    heat_data = monitoring_data[["latitude", "longitude", "ndvi"]].dropna().values
    HeatMap(heat_data, radius=15).add_to(folium_map)

    # Informations sur les sols
    for _, row in soil_data.iterrows():
        popup = f"""
        <b>Parcelle:</b> {row['parcelle_id']}<br>
        <b>Type de Sol:</b> {row['type_sol']}<br>
        <b>Surface:</b> {row['surface_ha']} ha<br>
        <b>PH:</b> {row['ph']}<br>
        """
        folium.Marker(
            location=[row["latitude"], row["longitude"]],
            popup=popup,
            icon=folium.Icon(color="green", icon="info-sign"),
        ).add_to(folium_map)

    return folium_map

# **Tableau de Bord Streamlit**
def main():
    st.title("Tableau de Bord Agricole Intégré")

    # Chargement des données
    yield_history, monitoring_data, weather_data, soil_data = load_data()

    # Visualisation des rendements historiques
    st.header("Évolution des Rendements")
    st.plotly_chart(create_yield_history_plot(yield_history), use_container_width=True)

    # Visualisation des prédictions de rendements
    st.header("Prédiction des Rendements Futurs")
    st.plotly_chart(create_yield_prediction_plot(yield_history), use_container_width=True)

    # Visualisation de la tendance NDVI
    st.header("NDVI : Tendance Hebdomadaire")
    st.plotly_chart(create_ndvi_temporal_plot(monitoring_data), use_container_width=True)

    # Matrice de stress hydrique
    st.header("Matrice de Stress Hydrique")
    st.plotly_chart(create_stress_matrix(monitoring_data), use_container_width=True)

    # Carte interactive
    st.header("Carte Interactif : NDVI et Sols")
    folium_map = create_map(monitoring_data, soil_data)
    st.components.v1.html(folium_map._repr_html_(), height=600)

if __name__ == "__main__":
    main()