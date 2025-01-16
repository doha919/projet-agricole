import folium
from folium import plugins
from branca.colormap import LinearColormap
import pandas as pd

class AgriculturalMap:
    def __init__(self, monitoring_data, yield_history_data):
        """
        Initialise la carte avec les données nécessaires
        """
        self.monitoring_data = monitoring_data
        self.yield_history_data = yield_history_data
        self.map = None
        self.yield_colormap = LinearColormap(
            colors=['red', 'yellow', 'green'],
            vmin=0,
            vmax=12  # Rendement maximum en tonnes/ha
        )

    def create_base_map(self):
        """
        Crée la carte de base avec les couches appropriées
        """
        center_lat = self.monitoring_data['latitude'].mean()
        center_lon = self.monitoring_data['longitude'].mean()
        self.map = folium.Map(location=[center_lat, center_lon], zoom_start=12)
        folium.TileLayer('CartoDB positron').add_to(self.map)

    def add_yield_history_layer(self):
        """
        Ajoute une couche visualisant l’historique des rendements
        """
        for _, row in self.yield_history_data.iterrows():
            location = [row['latitude'], row['longitude']]
            mean_yield = row['rendement_estime']
            trend = mean_yield * 1.05  # Exemple d'une simple tendance
            popup_content = f"""
            <b>Parcelle:</b> {row['parcelle_id']}<br>
            <b>Rendement estimé:</b> {mean_yield:.2f} t/ha<br>
            <b>Tendance:</b> {trend:.2f} t/ha<br>
            """
            folium.CircleMarker(
                location=location,
                radius=8,
                color=self.yield_colormap(mean_yield),
                fill=True,
                fill_opacity=0.7,
                popup=folium.Popup(popup_content, max_width=300)
            ).add_to(self.map)

    def add_current_ndvi_layer(self):
        """
        Ajoute une couche de la situation NDVI actuelle
        """
        for _, row in self.monitoring_data.iterrows():
            location = [row['latitude'], row['longitude']]
            popup_content = f"""
            <b>Parcelle:</b> {row['parcelle_id']}<br>
            <b>NDVI:</b> {row['ndvi']:.2f}<br>
            <b>LAI:</b> {row['lai']:.2f}<br>
            """
            folium.CircleMarker(
                location=location,
                radius=6,
                color='blue',
                fill=True,
                fill_opacity=0.6,
                popup=folium.Popup(popup_content, max_width=300)
            ).add_to(self.map)

    def add_risk_heatmap(self):
        """
        Ajoute une carte de chaleur des zones à risque
        """
        heat_data = [
            [row['latitude'], row['longitude'], row['ndvi']]
            for _, row in self.monitoring_data.iterrows()
        ]
        plugins.HeatMap(heat_data, min_opacity=0.4, radius=15).add_to(self.map)

# Exemple d'utilisation
def main():
    # Mettez ici les chemins vers vos fichiers de données locaux
    monitoring_data_path = "/Users/mac/Documents/projet_agricole/data/monitoring_cultures.csv"
    yield_history_data_path = "/Users/mac/Documents/projet_agricole/data/historique_rendements.csv"

    monitoring_data = pd.read_csv(monitoring_data_path)
    yield_history_data = pd.read_csv(yield_history_data_path)

    # Ajouter latitude et longitude si elles manquent dans les rendements
    if 'latitude' not in yield_history_data.columns or 'longitude' not in yield_history_data.columns:
        yield_history_data = yield_history_data.merge(
            monitoring_data[['parcelle_id', 'latitude', 'longitude']].drop_duplicates(),
            on='parcelle_id', how='left'
        )

    agri_map = AgriculturalMap(monitoring_data, yield_history_data)
    agri_map.create_base_map()
    agri_map.add_yield_history_layer()
    agri_map.add_current_ndvi_layer()
    agri_map.add_risk_heatmap()

    agri_map.map.save("agricultural_map.html")
    print("Carte sauvegardée sous 'agricultural_map.html'.")

if __name__ == "__main__":
    main()
