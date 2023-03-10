import pandas as pd
import streamlit as st
import csv
import os
from utils import utilities, pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from time import sleep
import calendar
import locale


locale.setlocale(locale.LC_ALL, ("es_ES", "UTF-8"))
sns.set(style='darkgrid')

# Carpeta de carga de datos
upload = 'data\\static\\files'
statics = 'data\\static\\'
predicted = 0 # Controla si se ha cargado la predicción
agrupamiento_seleccionado = None # Controla si se ha seleccionado un agrupamiento
visualizacion_seleccionada = None # Controla si se ha seleccionado un tipo de visualización

# Configuración de la página
titulo_web = 'Predict Ocup'
st.set_page_config(page_title=titulo_web, page_icon=":eyeglasses:", layout='wide')


# Título de la web e instrucciones
st.title('Predicciones del % de Ocupación')

st.markdown("""
 * Use el menu a la izquierda para seleccionar o cargar datos
 * La predicción aparecerá bajo este epígrafe
""")


@st.cache_data
def load_csv(file_path):
    with st.spinner('Loading data..'):
        # Analiza el csv para detectar el separador usado
        with open(file_path, 'r', encoding='utf-8') as csvfile:
            dialect = csv.Sniffer().sniff(csvfile.readline()) # Se lee la primera línea del csv con un sniffer
            separador = dialect.delimiter # Detectamos el separador de columna que se ha usado
        
        df_input = pd.DataFrame()
        df_input=pd.read_csv(file_path,sep=separador ,engine='python', encoding='utf-8',
                                parse_dates=True,
                                infer_datetime_format=True)
        sleep(3)
    return df_input

@st.cache_data
def genera_agrupamiento(df, agrupamiento):
    df_grouped = df
    df_grouped["Fecha"] = pd.to_datetime(df_grouped["Fecha"])
    df_grouped["Mes"] = df_grouped["Fecha"].dt.month
    if agrupamiento == 0:
        df_grouped =  df_grouped.groupby(['Mes'], as_index=False).agg({'Prediccion': 'mean'})
        df_grouped["Mes"] = df_grouped["Mes"].apply(lambda x: calendar.month_abbr[x])
    elif agrupamiento == 1:
        df_grouped = df_grouped.groupby(['AreaClinica','Mes'], as_index=False).agg({'Prediccion': 'mean'})
        df_grouped["Mes"] = df_grouped["Mes"].apply(lambda x: calendar.month_abbr[x])
    elif agrupamiento == 2:
        df_grouped = df_grouped.groupby(['DoctorID','Mes'], as_index=False).agg({'Prediccion': 'mean'})
        df_grouped["Mes"] = df_grouped["Mes"].apply(lambda x: calendar.month_abbr[x])
    
    return df_grouped


# DEFINICIÓN DEL SIDEBAR

st.sidebar.markdown("## Selecciona el origen de datos.")

# Selector de origen de datos
origen_seleccionado = st.sidebar.selectbox('Selecciona el origen de datos',
                                    ['Selecciona', 'Carga de CSV', 'Por aproximación'])


# Si se selecciona CSV
if origen_seleccionado == 'Carga de CSV':
    archivo = st.sidebar.file_uploader(label='Seleccionar CSV',
                                       accept_multiple_files=False,
                                       type=['csv']
                                       )
    if archivo is None:
        None

    else:        
        with open(os.path.join(os.getcwd(), upload,archivo.name),"wb") as f:
            f.write(archivo.getbuffer())
                
        df = load_csv(os.path.join(upload,archivo.name))
        st.write("Muestra del arhivo cargado.")
        st.write(df.iloc[5:11])
        
# Si se selecciona Aproximación      
elif origen_seleccionado == 'Por aproximación':
    porcentaje = st.sidebar.slider("Porcentaje de visitas recibidas", min_value=0.00, max_value=100.00)
    if st.sidebar.button(label="Validar porcentaje"):
        df_aprox = utilities.pickle_to_df(statics, "Agenda23.pkl")
        df_aprox["TotalVisitasEfectivas"] = round(df_aprox["TotalHuecos"]/(100/porcentaje), 0)
        st.write("Muestra del arhivo cargado.")
        st.write(df_aprox.iloc[5:11])


# Selector de descarga o visualización
visualizacion_seleccionada = st.selectbox('Seleccione visualización o descarga',
                                          ['Selecciona', 'Por pantalla', 'Descargar'])
        
# Selector de tipo de agrupamiento
agrupamiento_seleccionado = st.radio('Seleccionte tipo de agrupamiento',
                                     ('Sin agrupamiento', 'Por especialidad y mes', 'Por médico y mes'))
    
st.write("Click en el botón para predecir.")
if st.button(label="Predict"):
    with st.spinner('Predicting data..'):
        
        if origen_seleccionado == 'Carga de CSV':
            prediction = pipeline.pipeline(load_csv(os.path.join(upload,archivo.name))) # Aplica pipeline al archivo cargado y genera la predicción
            df["Prediccion"] = prediction*100 # Añade el resultado de la predicción multiplicado, como columna al dataframe de datos
            df.to_csv(os.path.join(upload,'prediccion.csv')) # Genera el csv con la predicción añadida al dataframe original
            predicted = 1
            sleep(5)
        elif origen_seleccionado == 'Por aproximación':
            df = utilities.pickle_to_df(statics, "Agenda23.pkl")
            df["TotalVisitasEfectivas"] = round(df["TotalHuecos"]/(100/porcentaje), 0)
            prediction = pipeline.pipeline(df) # Aplica pipeline al archivo cargado y genera la predicción
            df["Prediccion"] = prediction*100 # Añade el resultado de la predicción multiplicado, como columna al dataframe de datos
            df.to_csv(os.path.join(upload,'prediccion.csv')) # Genera el csv con la predicción añadida al dataframe original
            predicted = 1
            sleep(5)
        
        if visualizacion_seleccionada=='Descargar':
            st.download_button("Press to Download",os.path.join(upload,'prediccion.csv'),'prediccion.csv',"text/csv",key='download-csv')
        
        # Muestra de dataframe de acuerdo a parámentros seleccionados por usuario
        if agrupamiento_seleccionado == 'Sin agrupamiento' and visualizacion_seleccionada == 'Por pantalla' and predicted == 1:
            col1, col2 = st.columns([1,1])
            df_mensual = genera_agrupamiento(df, 0)
            
            plot, ax = plt.subplots()
            ax = sns.lineplot(data= df_mensual, x="Mes", y="Prediccion", markers=True, marker='o')
            ax.set(ylim=(0,None))
            plt.title("Evolución del % de Ocupación previsto", loc='center')
            plt.ylabel("% Ocupación")
            plt.xlabel("Mes")
            
            col1.dataframe(df, use_container_width=True)
            col2.pyplot(fig=plot)
            

        elif agrupamiento_seleccionado == 'Por especialidad y mes' and visualizacion_seleccionada == 'Por pantalla' and predicted == 1:
            col1, col2 = st.columns([1,1])
            df_mensual = genera_agrupamiento(df, 0)
            df_agrupado=genera_agrupamiento(df, 1)

            plot, ax = plt.subplots()
            ax = sns.lineplot(data= df_mensual, x="Mes", y="Prediccion", markers=True, marker='o')
            ax.set(ylim=(0,150))
            plt.title("Evolución del % de Ocupación previsto", loc='center')
            plt.ylabel("% Ocupación")
            plt.xlabel("Mes")
            
            col1.dataframe(df_agrupado.style.background_gradient(cmap='YlOrRd', subset=["Prediccion"]), use_container_width=True)
            col2.pyplot(fig=plot)
                
        elif agrupamiento_seleccionado == 'Por médico y mes' and visualizacion_seleccionada == 'Por pantalla' and predicted == 1: 
            col1, col2 = st.columns([1,1])
            df_mensual = genera_agrupamiento(df, 0)
            df_agrupado=genera_agrupamiento(df, 2)
            
            plot, ax = plt.subplots()
            ax = sns.lineplot(data= df_mensual, x="Mes", y="Prediccion", markers=True, marker='o')
            ax.set(ylim=(0,100))
            plt.title("Evolución del % de Ocupación previsto", loc='center')
            plt.ylabel("% Ocupación")
            plt.xlabel("Mes")
            
            col1.dataframe(df_agrupado.style.background_gradient(subset=['Prediccion'], cmap='YlOrRd'), use_container_width=True)
            col2.pyplot(fig=plot)


# Muestra gráfico de ocupación por especialidad
if agrupamiento_seleccionado == 'Por especialidad y mes' and visualizacion_seleccionada == 'Por pantalla' and predicted == 1:
    df_grouped_year = df_agrupado.loc[df_agrupado["Prediccion"]>0]
    specs = df_grouped_year["AreaClinica"].unique()
    columnas = 2
    filas = (len(specs)//columnas)
    pos = 0
    
    plot, ax = plt.subplots(filas, columnas, figsize=(10,30))
    plot.tight_layout(pad=5.0)
    while pos< len(specs)-1:
        for i in range(filas):
            for j in range(columnas):
                sns.lineplot(data=df_grouped_year[df_grouped_year["AreaClinica"]==specs[pos]], x="Mes", y="Prediccion", ax=ax[i,j], markers=True, marker='o')
                ax[i,j].set(ylim=(0,100))
                ax[i,j].set_title(specs[pos])
                ax[i,j].set_xlabel("")
                ax[i,j].set_ylabel("% Ocupación")
                pos +=1
    
    st.pyplot(plot)

# Muestra gráfico de ocupación por medico
elif agrupamiento_seleccionado == 'Por médico y mes' and visualizacion_seleccionada == 'Por pantalla' and predicted == 1:
    df_grouped_year = df_agrupado.loc[df_agrupado["Prediccion"]>0]
    docs = df_grouped_year["DoctorID"].unique()
    columnas = 2
    filas = (len(docs)//columnas)
    pos = 0

    plot, ax = plt.subplots(filas, columnas, figsize=(15,50))
    plot.tight_layout(pad=3.0)
    while pos< len(docs)-1:
        for i in range(filas):
            for j in range(columnas):
                sns.lineplot(data=df_grouped_year[df_grouped_year["DoctorID"]==docs[pos]], x="Mes", y="Prediccion", ax=ax[i,j], markers=True, marker='o')
                ax[i,j].set(ylim=(0,100))
                ax[i,j].set_title('Dr. '+str(docs[pos]))
                ax[i,j].set_xlabel("")
                ax[i,j].set_ylabel("% Ocupación")
                pos +=1
    
    st.pyplot(plot)
       
output = 0