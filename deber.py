import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

# Configuración de la página para hacerla más amplia
st.set_page_config(layout="wide")

# URL de la imagen (Logo o diseño)
image_url = "eig_logo-removebg-preview.png"

# Cargar base de datos
@st.cache_data
def load_data():
    url = "https://drive.google.com/uc?id=1buFDsjoe4iRYCtoqyKIuU-n2_c1q6Zki"  # Archivo cargado
    df = pd.read_csv(url)
    df['Appt Start Time'] = pd.to_datetime(df['Appt Start Time'])
    df['Hour'] = df['Appt Start Time'].dt.hour  # Extraer la hora del día
    df['Day of Week'] = df['Appt Start Time'].dt.day_name()  # Extraer el día de la semana
    return df

# Cargar datos
df = load_data()

# Configurar la imagen de encabezado
st.image(image_url, use_column_width=True)

# Título del dashboard
st.title("Dashboard de Pacientes con Múltiples Gráficos en Plotly y Mapa de Calor")

# Filtros
clinics = df["Clinic Name"].unique()
sources = df["Admit Source"].unique()

selected_clinic = st.selectbox("Selecciona la clínica", options=clinics)
selected_source = st.selectbox("Selecciona la fuente del paciente", options=sources)
selected_date_range = st.date_input("Selecciona el rango de fechas", [df['Appt Start Time'].min(), df['Appt Start Time'].max()])

# Filtrar datos según los filtros seleccionados
filtered_data = df[
    (df["Clinic Name"] == selected_clinic) &
    (df["Admit Source"] == selected_source) &
    (df["Appt Start Time"] >= pd.to_datetime(selected_date_range[0])) &
    (df["Appt Start Time"] <= pd.to_datetime(selected_date_range[1]))
]

# Mapa de calor del número de pacientes por día y hora (NO MODIFICAR ESTE)
st.subheader("Mapa de Calor: Número de Pacientes por Hora y Día de la Semana")

# Agrupar por día de la semana y hora, y contar el número de pacientes
heatmap_data = filtered_data.groupby(["Day of Week", "Hour"]).size().reset_index(name='Patient Count')

# Crear una tabla dinámica (pivot table) para el heatmap
heatmap_pivot = heatmap_data.pivot_table(index="Day of Week", columns="Hour", values="Patient Count", fill_value=0)

# Reordenar los días de la semana para que aparezcan en orden (de lunes a domingo)
days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
heatmap_pivot = heatmap_pivot.reindex(days_order)

# Gráfico de calor
fig1, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(heatmap_pivot, cmap="YlGnBu", ax=ax, annot=True, fmt="g", cbar_kws={'label': 'Número de Pacientes'})
ax.set_title("Número de Pacientes por Hora y Día de la Semana", fontsize=16)
ax.set_xlabel("Hora del Día")
ax.set_ylabel("Día de la Semana")

# Mostrar el gráfico en Streamlit
st.pyplot(fig1)

# Gráfico 1: Scatterplot de tiempo de espera y calificación de satisfacción
st.subheader("Scatterplot: Relación entre Tiempo de Espera y Calificación de Satisfacción")

fig2 = px.scatter(filtered_data, x="Wait Time Min", y="Care Score", color="Care Score", title="Scatterplot de Tiempo de Espera vs. Calificación de Satisfacción")
st.plotly_chart(fig2, use_container_width=True)

# Gráfico 2: Gráfico de barras del número de pacientes por día de la semana
st.subheader("Gráfico de Barras: Número de Pacientes por Día de la Semana")

day_counts = filtered_data['Day of Week'].value_counts().reset_index()
day_counts.columns = ['Day of Week', 'Count']

fig3 = px.bar(day_counts, x='Day of Week', y='Count', color='Day of Week', title="Pacientes por Día de la Semana")
st.plotly_chart(fig3, use_container_width=True)

# Gráfico 3: Histograma del tiempo de espera
st.subheader("Histograma: Distribución del Tiempo de Espera")

fig4 = px.histogram(filtered_data, x="Wait Time Min", nbins=20, title="Distribución del Tiempo de Espera")
st.plotly_chart(fig4, use_container_width=True)

# Gráfico 4: Pie chart de la distribución de fuentes de admisión
st.subheader("Gráfico Circular: Distribución de Fuentes de Admisión")

admit_source_counts = filtered_data['Admit Source'].value_counts().reset_index()
admit_source_counts.columns = ['Admit Source', 'Count']

fig5 = px.pie(admit_source_counts, values='Count', names='Admit Source', title="Distribución de Fuentes de Admisión")
st.plotly_chart(fig5, use_container_width=True)

# Gráfico 5: Boxplot del tiempo de espera por clínica
st.subheader("Boxplot: Tiempo de Espera por Clínica")

fig6 = px.box(filtered_data, x="Clinic Name", y="Wait Time Min", color="Clinic Name", title="Boxplot de Tiempo de Espera por Clínica")
st.plotly_chart(fig6, use_container_width=True)

# Gráfico 6: Treemap de pacientes por fuente de admisión y clínica
st.subheader("Treemap: Distribución de Pacientes por Fuente de Admisión y Clínica")

# Eliminar valores nulos antes de crear el treemap
df_clean = df.dropna(subset=['Admit Source', 'Clinic Name'])

# Usar el conteo de pacientes como valor en el treemap
df_clean['Patient Count'] = 1
fig7 = px.treemap(df_clean, path=['Admit Source', 'Clinic Name'], values='Patient Count', title="Distribución por Fuente de Admisión y Clínica")
st.plotly_chart(fig7, use_container_width=True)

# Gráfico 7: Gráfico de líneas del número de pacientes por hora del día
st.subheader("Gráfico de Líneas: Pacientes por Hora del Día")

hourly_patient_counts = filtered_data.groupby('Hour').size().reset_index(name='Patient Count')

fig8 = px.line(hourly_patient_counts, x='Hour', y='Patient Count', title="Pacientes por Hora del Día")
st.plotly_chart(fig8, use_container_width=True)

# Gráfico 8: Gráfico de áreas acumuladas del número de pacientes por día y hora
st.subheader("Gráfico de Áreas Acumuladas: Pacientes por Día y Hora")

daily_hourly_patient_counts = filtered_data.groupby(['Day of Week', 'Hour']).size().reset_index(name='Patient Count')

fig9 = px.area(daily_hourly_patient_counts, x='Hour', y='Patient Count', color='Day of Week', line_group='Day of Week', title="Pacientes por Día y Hora")
st.plotly_chart(fig9, use_container_width=True)

# Gráfico 9: Heatmap de correlación entre variables numéricas
st.subheader("Mapa de Calor: Correlación entre Variables Numéricas")

numeric_cols = ['Care Score', 'Wait Time Min']
corr = filtered_data[numeric_cols].corr()

fig10, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(corr, annot=True, cmap="Blues", ax=ax)
st.pyplot(fig10)
