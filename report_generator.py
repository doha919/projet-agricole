import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
from weasyprint import HTML
import os

class DataManager:
    def __init__(self):
        self.yield_history = None
        self.monitoring_data = None
        self.weather_data = None
        self.soil_data = None

    def load_data(self):
        try:
            self.yield_history = pd.read_csv("/Users/mac/Documents/projet_agricole/data/historique_rendements.csv", parse_dates=['date'])
            self.monitoring_data = pd.read_csv("/Users/mac/Documents/projet_agricole/data/monitoring_cultures.csv", parse_dates=['date'])
            self.weather_data = pd.read_csv("/Users/mac/Documents/projet_agricole/data/monitoring_cultures.csv", parse_dates=['date'])
            self.soil_data = pd.read_csv("/Users/mac/Documents/projet_agricole/data/sols.csv")
        except FileNotFoundError as e:
            print(f"Erreur lors du chargement des fichiers : {e}")

class AgriculturalAnalyzer:
    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)

    def analyze_yield_factors(self, parcelle_id):
        monitoring_data = self.data_manager.monitoring_data
        soil_data = self.data_manager.soil_data
        weather_data = self.data_manager.weather_data

        # Filtrer les données pour la parcelle spécifique
        parcel_data = monitoring_data[monitoring_data['parcelle_id'] == parcelle_id]
        parcel_data = parcel_data.merge(soil_data, on='parcelle_id', how='left')

        # Vérifier et compléter les colonnes nécessaires
        required_features = ['temperature', 'ph', 'stress_hydrique', 'lai']
        for feature in required_features:
            if feature not in parcel_data.columns and feature in weather_data.columns:
                parcel_data = parcel_data.merge(weather_data[['date', feature]], on='date', how='left')

        missing_features = [f for f in required_features if f not in parcel_data.columns]
        if missing_features:
            print(f"Colonnes manquantes : {', '.join(missing_features)}")
            return None

        if parcel_data.empty:
            print(f"Pas de données pour la parcelle {parcelle_id}.")
            return None

        # Préparer les données pour l'entraînement
        X = parcel_data[required_features].fillna(0)
        y = parcel_data['biomasse_estimee'].fillna(0)

        if len(y) == 0:
            print(f"Pas de données suffisantes pour la parcelle {parcelle_id}.")
            return None

        self.model.fit(X, y)
        importances = self.model.feature_importances_
        return pd.DataFrame({'Feature': required_features, 'Importance': importances}).sort_values(by='Importance', ascending=False)

class AgriculturalReportGenerator:
    def __init__(self, analyzer, data_manager):
        self.analyzer = analyzer
        self.data_manager = data_manager

    def generate_parcelle_report(self, parcelle_id):
        parcel_analysis = self.analyzer.analyze_yield_factors(parcelle_id)

        if parcel_analysis is None or parcel_analysis.empty:
            print(f"Pas de données suffisantes pour la parcelle {parcelle_id}.")
            return

        os.makedirs("reports", exist_ok=True)
        figure_path = self._generate_report_figures(parcelle_id, parcel_analysis)
        markdown_content = self._create_markdown_report(parcelle_id, parcel_analysis, figure_path)
        self._convert_to_pdf(markdown_content, parcelle_id)

    def _generate_report_figures(self, parcelle_id, parcel_analysis):
        figure_path = f"reports/{parcelle_id}_importance_factors.png"
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=parcel_analysis, palette='viridis')
        plt.title(f"Importance des Facteurs pour la Parcelle {parcelle_id}")
        plt.savefig(figure_path)
        plt.close()
        return figure_path

    def _create_markdown_report(self, parcelle_id, parcel_analysis, figure_path):
        markdown_content = f"""
        <h1>Rapport Agronomique - Parcelle {parcelle_id}</h1>
        <h2>Analyse des Facteurs de Rendement</h2>
        <p>Les facteurs suivants ont été identifiés comme les plus influents sur le rendement :</p>
        {parcel_analysis.to_html(index=False)}
        <h2>Recommandations</h2>
        <ul>
            <li>Optimiser la gestion des facteurs les plus influents.</li>
            <li>Continuer à surveiller régulièrement les données agronomiques.</li>
        </ul>
        <h2>Visualisation</h2>
        <img src="{figure_path}" alt="Importance des Facteurs">
        """
        return markdown_content

    def _convert_to_pdf(self, markdown_content, parcelle_id):
        html_file = f"reports/{parcelle_id}_report.html"
        pdf_file = f"reports/{parcelle_id}_report.pdf"

        # Sauvegarder le contenu HTML
        with open(html_file, 'w') as f:
            f.write(markdown_content)

        # Convertir en PDF
        HTML(html_file).write_pdf(pdf_file)
        print(f"Rapport PDF généré : {pdf_file}")

if __name__ == "__main__":
    data_manager = DataManager()
    data_manager.load_data()

    analyzer = AgriculturalAnalyzer(data_manager)
    report_generator = AgriculturalReportGenerator(analyzer, data_manager)

    # Génération d'un rapport pour une parcelle spécifique
    report_generator.generate_parcelle_report("P001")
