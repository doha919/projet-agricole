import pandas as pd
import numpy as np
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, HoverTool, LinearColorMapper
from bokeh.plotting import figure, curdoc
from bokeh.transform import linear_cmap
from bokeh.palettes import Plasma256, Turbo256
from sklearn.linear_model import LinearRegression


class AgriculturalDataManager:
    def __init__(self):
        self.yield_history = None
        self.monitoring_data = None
        self.weather_data = None

    def load_data(self):
        """
        Charge toutes les données nécessaires.
        """
        self.yield_history = pd.read_csv("data/historique_rendements.csv", parse_dates=['date'])
        self.monitoring_data = pd.read_csv("data/monitoring_cultures.csv", parse_dates=['date'])
        self.weather_data = pd.read_csv("data/meteo_detaillee.csv", parse_dates=['date'])

        self.yield_history['annee'] = self.yield_history['date'].dt.year
        self.yield_history = self.yield_history.infer_objects()

        numeric_cols = self.yield_history.select_dtypes(include=['float64', 'int64']).columns
        self.yield_history[numeric_cols] = self.yield_history[numeric_cols].interpolate()

        self.monitoring_data.drop_duplicates(inplace=True)


class AgriculturalDashboard:
    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.yield_source = ColumnDataSource(data=self.data_manager.yield_history)
        self.monitoring_source = ColumnDataSource(data=self.data_manager.monitoring_data)

    def create_yield_history_plot(self):
        """
        Crée un graphique de l'historique des rendements.
        """
        p = figure(
            title="Évolution Historique des Rendements",
            x_axis_label="Année",
            y_axis_label="Rendement (t/ha)",
            height=400,
            tools="pan,wheel_zoom,box_zoom,reset",
        )
        grouped_data = self.data_manager.yield_history.groupby('annee')['rendement_final'].mean().reset_index()
        source = ColumnDataSource(grouped_data)

        p.line(x='annee', y='rendement_final', source=source, line_width=3, color="#D35400", legend_label="Rendement Annuel")
        p.scatter(x='annee', y='rendement_final', source=source, size=10, color="#C0392B", legend_label="Points de Données", marker="circle")

        hover = HoverTool(tooltips=[("Année", "@annee"), ("Rendement", "@rendement_final")])
        p.add_tools(hover)
        p.legend.location = "top_left"
        return p

    def create_ndvi_temporal_plot(self):
        """
        Crée un graphique montrant l’évolution temporelle du NDVI.
        """
        monitoring_data = self.data_manager.monitoring_data.copy()
        monitoring_data['week'] = monitoring_data['date'].dt.to_period('W').apply(lambda r: r.start_time)
        aggregated_data = monitoring_data.groupby('week')['ndvi'].mean().reset_index()
        source = ColumnDataSource(aggregated_data)

        p = figure(
            title="NDVI : Tendance Hebdomadaire",
            x_axis_label="Date",
            y_axis_label="NDVI",
            x_axis_type="datetime",
            height=400,
            tools="pan,wheel_zoom,box_zoom,reset",
        )

        p.line(x='week', y='ndvi', source=source, line_width=3, color="#1ABC9C", legend_label="NDVI Moyen")
        p.scatter(x='week', y='ndvi', source=source, size=8, color="#16A085", legend_label="NDVI Points", marker="triangle")

        hover = HoverTool(tooltips=[("Semaine", "@week{%F}"), ("NDVI", "@ndvi{0.2f}")], formatters={"@week": "datetime"})
        p.add_tools(hover)

        p.legend.location = "top_left"
        return p

    def create_stress_matrix(self):
        """
        Crée une matrice de stress combinant stress hydrique et climatique.
        """
        monitoring_data = self.data_manager.monitoring_data.copy()
        source = ColumnDataSource(monitoring_data)

        p = figure(
            title="Matrice de Stress Hydrique",
            x_axis_label="Stress Hydrique",
            y_axis_label="LAI (Indice de Surface Foliaire)",
            height=400,
            tools="pan,wheel_zoom,box_zoom,reset",
        )
        mapper = linear_cmap(field_name='stress_hydrique', palette=Turbo256,
                             low=monitoring_data['stress_hydrique'].min(),
                             high=monitoring_data['stress_hydrique'].max())

        p.scatter(x='stress_hydrique', y='lai', source=source, size=12, color=mapper, alpha=0.8)
        hover = HoverTool(tooltips=[("Stress Hydrique", "@stress_hydrique"), ("LAI", "@lai")])
        p.add_tools(hover)

        return p

    def create_yield_prediction_plot(self):
        """
        Crée un graphique de prédiction des rendements.
        """
        historical_data = self.data_manager.yield_history.groupby('annee')['rendement_final'].mean().reset_index()
        X = historical_data[['annee']].values
        y = historical_data['rendement_final'].values

        model = LinearRegression()
        model.fit(X, y)

        future_years = np.arange(X[-1][0] + 1, X[-1][0] + 6).reshape(-1, 1)
        predicted_yields = model.predict(future_years)

        future_data = pd.DataFrame({'annee': future_years.flatten(), 'rendement_final': predicted_yields})
        combined_data = pd.concat([historical_data, future_data])
        source = ColumnDataSource(combined_data)
        future_source = ColumnDataSource(future_data)

        p = figure(
            title="Prédiction des Rendements Futurs",
            x_axis_label="Année",
            y_axis_label="Rendement Prévu (t/ha)",
            height=400,
            tools="pan,wheel_zoom,box_zoom,reset",
        )

        # Ajout des données historiques
        p.line(x='annee', y='rendement_final', source=source, line_width=3, color="#2C3E50", legend_label="Historique")
        p.scatter(x='annee', y='rendement_final', source=source, size=10, color="#34495E", legend_label="Données Historiques", marker="circle")

        # Ajout des prédictions
        p.line(x='annee', y='rendement_final', source=future_source, line_width=3, color="#E74C3C", legend_label="Prédiction")
        p.scatter(x='annee', y='rendement_final', source=future_source, size=10, color="#E67E22", legend_label="Données Prévues", marker="square")

        hover = HoverTool(tooltips=[("Année", "@annee"), ("Rendement", "@rendement_final")])
        p.add_tools(hover)

        p.legend.location = "top_left"
        return p

    def create_layout(self):
        """
        Organise les graphiques dans une mise en page cohérente.
        """
        yield_plot = self.create_yield_history_plot()
        ndvi_plot = self.create_ndvi_temporal_plot()
        stress_matrix = self.create_stress_matrix()
        prediction_plot = self.create_yield_prediction_plot()
        return column(yield_plot, ndvi_plot, stress_matrix, prediction_plot)


# Initialisation et exécution
data_manager = AgriculturalDataManager()
data_manager.load_data()

dashboard = AgriculturalDashboard(data_manager)
layout = dashboard.create_layout()

# Ajout de la mise en page au document Bokeh
curdoc().add_root(layout)
