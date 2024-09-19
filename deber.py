import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from scipy import stats
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Configuración de la página (más ancha)
st.set_page_config(layout="wide")


# Cargar los datos
@st.cache_data
def load_data():
    file_url = 'https://docs.google.com/spreadsheets/d/1mgDS76mbR9moLdNRShISSTpZIniAFcyJ/export?format=xlsx'
    df = pd.read_excel(file_url, sheet_name='Sheet1')
    return df


df = load_data()

# Crear pestañas para la aplicación
tab1, tab2 = st.tabs(["Análisis Financiero", "Evaluación de Significancia Estadística"])

# -------------------------------------------
# PESTAÑA 1: ANÁLISIS FINANCIERO
# -------------------------------------------
with tab1:
    st.title("Análisis Financiero de Cooperativas")

    # Filtros en la parte superior
    st.subheader("Filtros")
    col1, col2 = st.columns(2)
    with col1:
        cooperativas_seleccionadas = st.multiselect(
            "Selecciona las Cooperativas", df["Cooperativa"].unique(),
            default=df["Cooperativa"].unique(), key="cooperativas_filtro_tab1"
        )
    with col2:
        rango_fechas = st.date_input(
            "Selecciona el rango de fechas", [df['Fecha'].min(), df['Fecha'].max()],
            key="rango_fechas_tab1"
        )

    # Filtrar el DataFrame
    df_filtered = df[(df["Cooperativa"].isin(cooperativas_seleccionadas)) &
                     (df["Fecha"] >= pd.to_datetime(rango_fechas[0])) &
                     (df["Fecha"] <= pd.to_datetime(rango_fechas[1]))]

    # --- Gráfico de Calor: Número de cooperativas activas en el tiempo ---
    st.header("Gráfico de Calor: Cooperativas activas en el tiempo")

    # Crear una tabla pivote para contar cooperativas por mes y año
    df['AñoMes'] = df['Fecha'].dt.to_period('M').astype(str)
    df_heatmap = df.groupby(['AñoMes', 'Cooperativa']).size().reset_index(name='Count')
    df_pivot = df_heatmap.pivot_table(index='AñoMes', columns='Cooperativa', values='Count', fill_value=0)

    # Crear el heatmap con Plotly
    fig_heatmap = px.imshow(df_pivot.T,
                            labels=dict(x="Mes/Año", y="Cooperativa", color="Número de Cooperativas Activas"),
                            aspect="auto", color_continuous_scale="Blues")
    fig_heatmap.update_layout(title="Número de Cooperativas Activas por Mes", height=600)
    st.plotly_chart(fig_heatmap, use_container_width=True)


    # Función para crear gráficos de líneas individuales
    def plot_line_chart_individual(df, indicador, titulo):
        fig = go.Figure()
        df_temp = df[df['Indicador'] == indicador]
        for cooperativa in df_temp['Cooperativa'].unique():
            df_coop = df_temp[df_temp['Cooperativa'] == cooperativa]
            fig.add_trace(go.Scatter(x=df_coop['Fecha'], y=df_coop['Valor'], mode='lines+markers', name=cooperativa))

        fig.update_layout(title=titulo, template="plotly_dark", xaxis_title='Fecha', yaxis_title='Valor', height=400)
        return fig


    # Función para crear gráficos de barras o dispersión cuando sea más adecuado
    def plot_scatter_or_bar(df, indicador, titulo, graph_type='scatter'):
        fig = go.Figure()
        df_temp = df[df['Indicador'] == indicador]
        for cooperativa in df_temp['Cooperativa'].unique():
            df_coop = df_temp[df_temp['Cooperativa'] == cooperativa]
            if graph_type == 'scatter':
                fig.add_trace(
                    go.Scatter(x=df_coop['Fecha'], y=df_coop['Valor'], mode='markers+lines', name=cooperativa))
            else:
                fig.add_trace(go.Bar(x=df_coop['Fecha'], y=df_coop['Valor'], name=cooperativa))

        fig.update_layout(title=titulo, template="plotly_white", xaxis_title='Fecha', yaxis_title='Valor', height=400)
        return fig


    # --- Gráfico de Líneas y Barras según el indicador ---
    st.header("Gráficos Individuales por Indicador")

    # Gráficos de Suficiencia Patrimonial y Calidad de Activos (Líneas)
    indicadores_suficiencia = [
        "( PATRIMONIO + RESULTADOS ) / ACTIVOS INMOVILIZADOS",
        "ACTIVOS IMPRODUCTIVOS NETOS / TOTAL ACTIVOS",
        "ACTIVOS PRODUCTIVOS / TOTAL ACTIVOS",
        "ACTIVOS PRODUCTIVOS / PASIVOS CON COSTO"
    ]
    st.subheader("Suficiencia Patrimonial, Estructura y Calidad de Activos")
    for indicador in indicadores_suficiencia:
        fig = plot_line_chart_individual(df_filtered, indicador, indicador)
        st.plotly_chart(fig, use_container_width=True)

    # Gráficos de Índices de Morosidad (Dispersión en lugar de líneas)
    st.subheader("Índices de Morosidad")
    indicadores_morosidad = [
        "MOROSIDAD DE LA CARTERA DE CREDITO PRODUCTIVO",
        "MOROSIDAD DE LA CARTERA DE CONSUMO",
        "MOROSIDAD DE LA CARTERA DE CREDITO INMOBILIARIO",
        "MOROSIDAD DE LA CARTERA DE MICROCREDITO",
        "MOROSIDAD DE LA CARTERA DE VIVIENDA DE INTERES SOCIAL Y PUBLICO",
        "MOROSIDAD DE LA CARTERA DE CREDITO EDUCATIVO",
        "MOROSIDAD DE LA CARTERA TOTAL"
    ]

    for indicador in indicadores_morosidad:
        fig = plot_scatter_or_bar(df_filtered, indicador, indicador, graph_type='scatter')
        st.plotly_chart(fig, use_container_width=True)

    # Gráficos de Cobertura de Provisiones para Cartera Improductiva (Barras)
    st.subheader("Cobertura de Provisiones para Cartera Improductiva")
    indicadores_cobertura = [
        "COBERTURA DE LA CARTERA DE CREDITO PRODUCTIVO",
        "COBERTURA DE LA CARTERA DE CREDITO CONSUMO",
        "COBERTURA DE LA CARTERA DE CREDITO INMOBILIARIO",
        "COBERTURA DE LA CARTERA DE MICROCREDITO",
        "COBERTURA DE LA CARTERA DE VIVIENDA DE IINTERES PUBLICO",
        "COBERTURA DE LA CARTERA DE CREDITO EDUCATIVO",
        "COBERTURA DE LA CARTERA PROBLEMÁTICA"
    ]

    for indicador in indicadores_cobertura:
        fig = plot_scatter_or_bar(df_filtered, indicador, indicador, graph_type='bar')
        st.plotly_chart(fig, use_container_width=True)

    # --- Gráfico de Intermediación financiera ---
    st.subheader("Intermediación Financiera")
    indicador_intermediacion = "CARTERA BRUTA / (DEPOSITOS A LA VISTA + DEPOSITOS A PLAZO)"
    fig_intermediacion = plot_line_chart_individual(df_filtered, indicador_intermediacion, "Intermediación Financiera")
    st.plotly_chart(fig_intermediacion, use_container_width=True)

    # --- Gráficos de Rendimiento de la cartera ---
    st.subheader("Rendimiento de la Cartera")
    indicadores_rendimiento = [
        "RENDIMIENTO DE LA CARTERA DE CREDITO PRODUCTIVO POR VENCER",
        "RENDIMIENTO DE LA CARTERA DE CREDITO CONSUMO",
        "RENDIMIENTO DE LA CARTERA DE CREDITO INMOBILIARIO POR VENCER",
        "RENDIMIENTO DE LA CARTERA DE MICROCREDITO POR VENCER",
        "RENDIMIENTO DE LA CARTERA DE VIVIENDA DE IINTERES PUBLICO POR VENCER",
        "RENDIMIENTO DE LA CARTERA DE CREDITO EDUCATIVO POR VENCER",
        "CARTERAS DE CRÉDITOS  REFINANCIADAS",
        "CARTERAS DE CRÉDITOS  REESTRUCTURADAS",
        "CARTERA POR VENCER TOTAL"
    ]

    for indicador in indicadores_rendimiento:
        fig = plot_line_chart_individual(df_filtered, indicador, indicador)
        st.plotly_chart(fig, use_container_width=True)

    # --- Gráficos
    # --- Gráficos de Vulnerabilidad del patrimonio ---
    st.subheader("Vulnerabilidad del Patrimonio")
    indicadores_vulnerabilidad = [
        "CARTERA IMPRODUCTIVA DESCUBIERTA / (PATRIMONIO + RESULTADOS)",
        "CARTERA IMPRODUCTIVA / PATRIMONIO (DIC)",
        "FK = (PATRIMONIO + RESULTADOS - INGRESOS EXTRAORDINARIOS) / ACTIVOS TOTALES",
        "INDICE DE CAPITALIZACION NETO: FK / FI"
    ]

    for indicador in indicadores_vulnerabilidad:
        fig = plot_line_chart_individual(df_filtered, indicador, indicador)
        st.plotly_chart(fig, use_container_width=True)

    # --- Gráfico de Activos improductivos ---
    st.subheader("Activos Improductivos")
    indicador_activos_improductivos = "FI = 1 + (ACTIVOS IMPRODUCTIVOS / ACTIVOS TOTALES)"
    fig_activos_improductivos = plot_line_chart_individual(df_filtered, indicador_activos_improductivos, "Activos Improductivos")
    st.plotly_chart(fig_activos_improductivos, use_container_width=True)

# -------------------------------------------
# PESTAÑA 2: EVALUACIÓN DE SIGNIFICANCIA ESTADÍSTICA
# -------------------------------------------
with tab2:
    st.title("Evaluación de Significancia Estadística")

    # Filtros en la parte superior para entidad financiera y rango de fechas
    st.subheader("Filtros")
    col1, col2 = st.columns(2)
    with col1:
        # Filtros con keys únicos
        cooperativas_seleccionadas = st.multiselect(
            "Selecciona las Cooperativas", df["Cooperativa"].unique(),
            default=df["Cooperativa"].unique(), key="cooperativas_filtro_tab2"
        )
    with col2:
        rango_fechas = st.date_input(
            "Selecciona el rango de fechas", [df['Fecha'].min(), df['Fecha'].max()],
            key="rango_fechas_tab2"
        )

    # Filtrar el DataFrame
    df_filtered = df[(df["Cooperativa"].isin(cooperativas_seleccionadas)) &
                     (df["Fecha"] >= pd.to_datetime(rango_fechas[0])) &
                     (df["Fecha"] <= pd.to_datetime(rango_fechas[1]))]

    # Selección de variables para análisis
    st.subheader("Selección de variables para el análisis")
    variables_disponibles = df_filtered['Indicador'].unique()
    variables_seleccionadas = st.multiselect(
        "Selecciona las variables para análisis", variables_disponibles, default=variables_disponibles, key="variables_analisis_tab2"
    )

    # Filtrar DataFrame solo con las variables seleccionadas
    df_variables = df_filtered[df_filtered['Indicador'].isin(variables_seleccionadas)]
    df_pivot = df_variables.pivot_table(index=["Fecha", "Cooperativa"], columns="Indicador", values="Valor").reset_index()

    # --------------------
    # Correlación de Spearman
    # --------------------
    st.subheader("Correlación de Spearman")
    if not df_pivot.empty and len(variables_seleccionadas) > 1:
        # Calcular la correlación de Spearman
        correlacion_spearman = df_pivot[variables_seleccionadas].corr(method='spearman')

        # Graficar el heatmap con Plotly
        fig_heatmap_spearman = px.imshow(correlacion_spearman, labels=dict(x="Variable", y="Variable", color="Correlación de Spearman"),
                                         aspect="auto", color_continuous_scale="RdBu_r")
        fig_heatmap_spearman.update_layout(title="Mapa de calor de la correlación de Spearman", height=900)  # Ajustar altura
        st.plotly_chart(fig_heatmap_spearman, use_container_width=True)
    else:
        st.warning("No hay suficientes datos para calcular la correlación o selecciona más variables.")

    # --------------------
    # Regresión múltiple
    # --------------------
    st.subheader("Modelo de Regresión Múltiple")

    variables_independientes = st.multiselect("Selecciona las variables independientes", variables_seleccionadas, key="independientes_regresion_tab2")
    variable_dependiente = st.selectbox("Selecciona la variable dependiente", variables_seleccionadas, key="dependiente_regresion_tab2")

    if len(variables_independientes) > 0 and variable_dependiente:
        # Eliminar filas con valores faltantes
        df_pivot = df_pivot.dropna(subset=variables_independientes + [variable_dependiente])

        # Definir X (independientes) y y (dependiente)
        X = df_pivot[variables_independientes]
        y = df_pivot[variable_dependiente]

        # Modelo de regresión usando statsmodels para obtener p-values
        X = sm.add_constant(X)  # Agregar una constante para el intercepto
        modelo = sm.OLS(y, X).fit()

        # Coeficientes de la regresión
        st.write("**Coeficientes de la regresión múltiple:**")
        st.write(modelo.summary())  # Mostrar resumen con coeficientes y p-values

        # Mostrar R^2 y ajustar el scatter plot entre la predicción y la realidad
        y_pred = modelo.predict(X)
        fig_regresion = px.scatter(x=y, y=y_pred, labels={'x': 'Valores Reales', 'y': 'Valores Predichos'})
        fig_regresion.update_layout(title="Valores Reales vs Predichos (Regresión Múltiple)", height=400)
        st.plotly_chart(fig_regresion, use_container_width=True)
    else:
        st.warning("Selecciona las variables independientes y dependiente.")

    # --------------------
    # Árbol de decisión
    # --------------------
    st.subheader("Modelo de Árbol de Decisión")

    # Selección de variables para el Árbol de Decisión
    variables_arbol = st.multiselect("Selecciona las variables para el modelo de Árbol de Decisión", variables_seleccionadas, key="variables_arbol_tab2")

    if len(variables_arbol) > 1:
        # Eliminar filas con valores faltantes
        df_arbol = df_pivot.dropna(subset=variables_arbol + [variable_dependiente])

        # Definir X (independientes) y y (dependiente)
        X_arbol = df_arbol[variables_arbol]
        y_arbol = df_arbol[variable_dependiente]

        # Crear y entrenar el modelo de Árbol de Decisión
        arbol = DecisionTreeRegressor(random_state=42)
        arbol.fit(X_arbol, y_arbol)

        # Mostrar la importancia de las variables
        st.write("**Importancia de las variables en el Árbol de Decisión:**")
        importancia_vars = pd.DataFrame({
            'Variable': variables_arbol,
            'Importancia': arbol.feature_importances_
        }).sort_values(by='Importancia', ascending=False)
        st.table(importancia_vars)

        # Predicciones y coeficiente R^2
        y_pred_arbol = arbol.predict(X_arbol)
        r2_arbol = r2_score(y_arbol, y_pred_arbol)
        st.write(f"**Coeficiente de determinación R² del Árbol de Decisión: {r2_arbol:.4f}**")

        # Visualización del Árbol de Decisión con Plotly
        fig_arbol = go.Figure()

        # Obtener profundidad y nodos del árbol
        n_nodos = arbol.tree_.node_count
        profundidad = arbol.get_depth()

        # Añadir la visualización básica del árbol
        fig_arbol.add_trace(go.Scatter(
            x=list(range(n_nodos)),
            y=arbol.tree_.threshold,
            mode='markers',
            marker=dict(size=10, color='royalblue', symbol='circle'),
            text=[f"Nodo {i}" for i in range(n_nodos)]
        ))

        # Ajustar el gráfico
        fig_arbol.update_layout(
            title=f"Árbol de Decisión con profundidad {profundidad}",
            xaxis_title="Nodos",
            yaxis_title="Umbral de división",
            height=600
        )

        st.plotly_chart(fig_arbol, use_container_width=True)

    else:
        st.warning("Selecciona al menos dos variables para aplicar el modelo de Árbol de Decisión.")
