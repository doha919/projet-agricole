import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta


class AgriculturalAnalyzer:
    def __init__(self, data_manager):
        """
        Initialize the analyzer with the data manager.
        This class uses historical and current data to generate insights.
        """
        self.data_manager = data_manager
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)

    def analyze_yield_factors(self, parcelle_id):
        """
        Analyze factors influencing yields for a given parcelle.
        """
        print(f"\n--- Analyzing Yield Factors for Parcelle: {parcelle_id} ---")

        # Load and filter data for the parcelle
        parcelle_data = self.data_manager.prepare_features()
        parcelle_data = parcelle_data[parcelle_data['parcelle_id'] == parcelle_id]

        if parcelle_data.empty:
            print(f"Error: No data available for parcelle {parcelle_id}.")
            return None

        print("Parcelle Data Loaded:\n", parcelle_data.head())
        print("Available Columns:", parcelle_data.columns)

        # Check for rendement columns
        yield_data = None
        if 'rendement' in parcelle_data.columns:
            yield_data = parcelle_data['rendement']
        elif 'rendement_final' in parcelle_data.columns:
            yield_data = parcelle_data['rendement_final']
        else:
            print("Error: Column 'rendement' or 'rendement_final' is missing.")
            return None

        features = parcelle_data.drop(columns=['parcelle_id', 'rendement', 'rendement_final'], errors='ignore')

        print("Features for Training:\n", features.head())
        print("Yield Data:\n", yield_data.head())

        # Train the Random Forest model
        self.model.fit(features, yield_data)
        feature_importances = pd.DataFrame({
            'Feature': features.columns,
            'Importance': self.model.feature_importances_
        }).sort_values(by='Importance', ascending=False)

        print("\nFeature Importances Computed:\n", feature_importances)
        return feature_importances

    def _calculate_yield_correlations(self, yield_data, weather_data, soil_data):
        """
        Calculate correlations between yields and environmental factors.
        """
        print("\n--- Calculating Correlations Between Yields and Factors ---")
        merged_data = pd.concat([yield_data, weather_data, soil_data], axis=1)

        # Filter numeric columns for correlation
        numeric_data = merged_data.select_dtypes(include=[np.number])

        if 'rendement' not in numeric_data.columns:
            print("Error: 'rendement' column is missing in the numeric data.")
            return None

        correlations = numeric_data.corr()['rendement'].sort_values(ascending=False)
        print("Computed Correlations:\n", correlations)
        return correlations

    def _analyze_performance_trend(self, parcelle_data):
        """
        Analyze performance trend for a parcelle.
        """
        print("\n--- Analyzing Performance Trend ---")
        if 'date' not in parcelle_data.columns or 'rendement' not in parcelle_data.columns:
            print("Error: Required columns ('date', 'rendement') are missing.")
            return None

        parcelle_data = parcelle_data.sort_values(by='date')
        print("Sorted Parcelle Data:\n", parcelle_data[['date', 'rendement']].head())

        yield_trend = np.polyfit(parcelle_data['date'].map(datetime.toordinal), parcelle_data['rendement'], 1)
        print(f"Yield Trend (Slope, Intercept): {yield_trend}")
        return yield_trend

    def _analyze_yield_stability(self, yield_series):
        """
        Analyze yield stability over time.
        """
        print("\n--- Analyzing Yield Stability ---")
        mean_yield = np.mean(yield_series)
        std_yield = np.std(yield_series)
        stability_index = 1 - (std_yield / mean_yield)
        print(f"Stability Index: {stability_index:.2f} (Mean: {mean_yield}, Std: {std_yield})")
        return stability_index

    def _calculate_stability_index(self, yield_series):
        """
        Calculate a custom stability index, including trend.
        """
        print("\n--- Calculating Custom Stability Index ---")
        parcelle_data = self.data_manager.yield_history[['date', 'rendement']].copy()
        parcelle_data = parcelle_data.loc[parcelle_data['rendement'].isin(yield_series)]

        trend = self._analyze_performance_trend(parcelle_data)
        stability = self._analyze_yield_stability(yield_series)

        stability_index = stability * (1 + trend[0])
        print(f"Final Stability Index: {stability_index}")
        return stability_index


if __name__ == "__main__":
    from data_manager import AgriculturalDataManager

    # Initialize data manager
    data_manager = AgriculturalDataManager()
    print("Loading Data...")
    data_manager.load_data()

    if data_manager.yield_history.empty:
        print("Error: Yield history data is missing or not loaded correctly.")
    else:
        analyzer = AgriculturalAnalyzer(data_manager)

        # Example parcelle ID
        parcelle_id = 'P001'
        print(f"\n--- Running Analysis for Parcelle: {parcelle_id} ---")

        # Analyze yield factors
        feature_importances = analyzer.analyze_yield_factors(parcelle_id)

        # Calculate correlations
        correlations = analyzer._calculate_yield_correlations(
            data_manager.yield_history['rendement'],
            data_manager.weather_data,
            data_manager.soil_data
        )

        # Analyze performance trend
        parcelle_data = data_manager.yield_history[data_manager.yield_history['parcelle_id'] == parcelle_id]
        trend = analyzer._analyze_performance_trend(parcelle_data)

        # Analyze yield stability
        stability_index = analyzer._analyze_yield_stability(parcelle_data['rendement'])

        # Calculate custom stability index
        custom_stability_index = analyzer._calculate_stability_index(parcelle_data['rendement'])