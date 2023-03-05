# Importación de librerías
import sys 
import os
import seaborn as sns

# Librerias ML
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline

# Librerías de credenciales y utilidades
from utils import utilities

# Modificar configuración de sklearn
from sklearn._config import set_config

def pipeline(df):
   
   set_config(transform_output="pandas")
   ruta_alg = '\\data\\model\\'
    
   drop_columns = ["AreaClinica", "Fecha", "TotalVisitasEfectivas", "TotalHuecos", "VisitasEfect_Tipificada"]
    
   pipeline = Pipeline([
      ("tipificacion", FunctionTransformer(utilities.tipificacion, validate=False, kw_args={'column': 'TotalVisitasEfectivas'})),
      ("cambio_tipo", FunctionTransformer(utilities.cambia_tipo_fecha, validate=False, kw_args={'column': 'Fecha'})),
      ("laborable", FunctionTransformer(utilities.dia_laborable, validate=False, kw_args={'column': 'Fecha'})),
      ("dummies", FunctionTransformer(utilities.getdummies, validate=False, kw_args={'columns': 'AreaClinica'})),
      ("selector", utilities.EliminaColumnas(drop_columns))
      ,("scaler", MinMaxScaler())
      ])

   df_data_scaled = pipeline.fit_transform(df)
    
   columnas_ordenadas = ['DoctorID', 'VisitasAjustada', 'IntervalAgenda', 'MinutosConsulta',
       'EsLaborable', 'Dia', 'Mes', 'Anyo',
       'AreaClinica_ALERGOLOGÍA', 'AreaClinica_APARATO DIGESTIVO',
       'AreaClinica_CARDIOLOGIA ',
       'AreaClinica_CIRUGIA GENERAL Y AP.DIGESTIVO',
       'AreaClinica_CIRUGIA PLASTICA', 'AreaClinica_CIRUGIA VASCULAR',
       'AreaClinica_DERMATOLOGIA', 'AreaClinica_ECOGRAFIA',
       'AreaClinica_ELECTROMIOGRAFIA', 'AreaClinica_ENDOCRINOLOGIA',
       'AreaClinica_FISIOTERAPIA SALUD MUJER', 'AreaClinica_GINECOLOGÍA',
       'AreaClinica_MAMOGRAFÍA', 'AreaClinica_MEDICINA ESTÉTICA',
       'AreaClinica_MEDICINA GENERAL', 'AreaClinica_NEUMOLOGIA',
       'AreaClinica_NEUROLOGIA', 'AreaClinica_NUTRICION',
       'AreaClinica_OFTALMOLOGÍA', 'AreaClinica_OTORRINOLARINGOLOGIA',
       'AreaClinica_PERITO', 'AreaClinica_PODOLOGIA',
       'AreaClinica_PRUEBAS OFTALMOLOGICAS', 'AreaClinica_PSICOLOGÍA',
       'AreaClinica_PSIQUIATRIA', 'AreaClinica_RADIOLOGIA',
       'AreaClinica_REHABILITACION', 'AreaClinica_REUMATOLOGIA',
       'AreaClinica_TRAUMATOLOGIA', 'AreaClinica_UROLOGIA',
       'AreaClinica_Urología/salud sexual y reproductiva/andrología']
    
   ruta = os.getcwd()+ruta_alg
   df_data_scaled = df_data_scaled[columnas_ordenadas]
   
   modelo = utilities.pickle_to_df(ruta, "my_model.pkl")
   prediccion = modelo.predict(df_data_scaled)
   
   return prediccion