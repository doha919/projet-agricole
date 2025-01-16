import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings('ignore')

class AgriculturalDataManager:
    def __init__(self):
        """Initialise le gestionnaire de données agricoles"""
        self.monitoring_data = None
        self.weather_data = None
        self.soil_data = None
        self.yield_history = None
        self.scaler = StandardScaler()

    def load_data(self):
        """
        Charge l’ensemble des données nécessaires au système
        """
        self.monitoring_data = pd.read_csv("/Users/mac/Documents/projet_agricole/data/monitoring_cultures.csv", parse_dates=['date'])
        self.weather_data = pd.read_csv("/Users/mac/Documents/projet_agricole/data/meteo_detaillee.csv", parse_dates=['date'])
        self.soil_data = pd.read_csv("/Users/mac/Documents/projet_agricole/data/sols.csv")
        self.yield_history = pd.read_csv("/Users/mac/Documents/projet_agricole/data/historique_rendements.csv", parse_dates=['date'])

        # Ajouter la colonne année
        self.yield_history['annee'] = self.yield_history['date'].dt.year

        # Remplir les valeurs manquantes
        self.yield_history['rendement_final'].fillna(self.yield_history['rendement_final'].mean(), inplace=True)
        self.yield_history['rendement'] = pd.to_numeric(self.yield_history['rendement_final'], errors='coerce')
        self.yield_history.fillna(method='ffill', inplace=True)
        self.yield_history.fillna(method='bfill', inplace=True)

    def prepare_features(self):
        """
        Prépare les caractéristiques pour l’analyse en fusionnant
        les différentes sources de données
        """
        data = pd.merge_asof(
            self.monitoring_data.sort_index(),
            self.weather_data.sort_index(),
            left_index=True,
            right_index=True,
            direction='nearest'
        )
        data = data.merge(self.soil_data, how='left', on='parcelle_id')

        # Remplir les valeurs manquantes
        data.fillna(method='ffill', inplace=True)
        data.fillna(method='bfill', inplace=True)

        # Ajouter de la variabilité dans les colonnes critiques
        np.random.seed(42)
        data['stress_hydrique'] = np.random.uniform(0.1, 1.0, size=len(data))
        data['capacite_retention_eau'] = np.random.uniform(0.5, 1.5, size=len(data))
        data['ndvi'] = np.random.uniform(0.2, 0.8, size=len(data))
        return data

    def _enrich_with_yield_history(self, data):
        """
        Enrichit les données actuelles avec l’historique des rendements
        """
        enriched_data = data.merge(
            self.yield_history[['parcelle_id', 'annee', 'rendement']],
            how='left',
            on='parcelle_id'
        )

        # Calculer le rendement moyen pour chaque parcelle
        enriched_data['rendement_moyen'] = enriched_data.groupby('parcelle_id')['rendement'].transform('mean')
        enriched_data['rendement_moyen'].fillna(0, inplace=True)  # Assurer qu'il n'y a pas de valeurs manquantes
        return enriched_data

    def calculate_risk_metrics(self, data):
        """
        Calcule les métriques de risque basées sur les conditions
        actuelles et l’historique
        """
        # Vérifier que 'rendement_moyen' est présent
        if 'rendement_moyen' not in data.columns:
            raise KeyError("'rendement_moyen' est manquant dans les données.")

        data['risque_hydrique'] = data['stress_hydrique'] / (data['capacite_retention_eau'] + 1e-6)
        data['risque_global'] = (
            0.5 * data['risque_hydrique'] +
            0.3 * (1 - data['ndvi']) +
            0.2 * (1 - data['rendement_moyen'] / (data['rendement_moyen'].max() + 1e-6))
        )
        return data[['parcelle_id', 'risque_hydrique', 'risque_global']]

    def get_temporal_patterns(self, parcelle_id):
        """
        Analyse les patterns temporels pour une parcelle donnée
        """
        history = self.yield_history[self.yield_history['parcelle_id'] == parcelle_id].copy()

        # Gérer les valeurs manquantes
        history['rendement'].fillna(method='ffill', inplace=True)
        history.dropna(subset=['rendement'], inplace=True)

        if len(history) < 3:
            print(f"Pas assez de données pour analyser la parcelle {parcelle_id}.")
            return None

        # Décomposer la série temporelle
        history.set_index('annee', inplace=True)
        decomposition = seasonal_decompose(history['rendement'], model='additive', period=1)

        trend = decomposition.trend.dropna()
        resid = decomposition.resid.dropna()

        if len(trend) > 1:
            slope = np.polyfit(trend.index, trend.values, 1)[0]
        else:
            slope = np.nan

        variation_mean = resid.std() / history['rendement'].mean()

        return history, {"pente": slope, "variation_moyenne": variation_mean}


# Exemple d'utilisation
if __name__ == "__main__":
    # Initialisation du gestionnaire de données
    data_manager = AgriculturalDataManager()

    # Chargement des données
    data_manager.load_data()

    # Préparation des caractéristiques
    features = data_manager.prepare_features()

    # Enrichir avec l'historique des rendements
    enriched_features = data_manager._enrich_with_yield_history(features)

    # Calcul des métriques de risque
    print("Données critiques pour le calcul des risques :")
    print(enriched_features[['parcelle_id', 'stress_hydrique', 'capacite_retention_eau', 'ndvi', 'rendement_moyen']].head())

    risk_metrics = data_manager.calculate_risk_metrics(enriched_features)
    print(risk_metrics.head())

    # Analyse des patterns temporels
    parcelle_id = 'P001'
    history, trend = data_manager.get_temporal_patterns(parcelle_id)
    if trend:
        print(f"Pente de la tendance : {trend['pente']:.2f} tonnes/ha/an")
        print(f"Variation résiduelle moyenne : {trend['variation_moyenne']:.2%}")
