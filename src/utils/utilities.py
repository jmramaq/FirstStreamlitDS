
import pandas as pd
import numpy as np
import sqlalchemy as sa
import matplotlib.pyplot as plt
import pyodbc
import seaborn as sns
import pickle
import io
import statsmodels.api as sm
import scipy.stats as ss
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error

q_doctors = ("SELECT DoctorID,TipusUsuari,Abreviatura,Titol,"
             "Nom,Cognoms,AreaClinica,Titulacio,AgendaOrder,"
             "Doctor,IntervalAgenda,DoctorPerPaginaAgenda,"
             "Baixa,DataModificacioRegistre FROM dbo.Doctors "
             "WHERE IsDoctor=1 "
             "AND DoctorID not in (588,79,134,159,349,352,353,474,475,476,603,415,509,525,529,589,491,492)")

q_horarios = "SELECT * FROM dbo.Doctors_Horaris"

q_visitas = ("SELECT [ApuntID],[VisitaID],[IsMutua],"
             "[Data],[Mes],[Any],[DiaSetmana],[DataInici],"
             "[DataFinal],[DoctorID],[TipusVisita],[Estat],"
             "[EstatID],[PacientID],[TipusVisitaID],[Duracio],"
             "[Anulada],[Administracio],[EpisodiID],[MutuaID],"
             "[Mutua],[VisitaEstatID],[NumHistoria],[DelegacioID],"
             "[NomDelegacio],[AbreviaturaDelegacio],[DataAlta],"
             "[MesAlta],[AnyAlta],[IsPrimeraVisita],[TipusVisitaGeneral],"
             "[TipusAtencio] FROM [dbo].[Visita_Sel_Base] "
             "WHERE [Data] >= '01-01-2021' "
             "AND DoctorID not in (588,79,134,159,349,352,353,474,475,476,603,415,509,525,529,589,491,492)")

def df_to_pickle(df,ruta:str,nombre:str="File"):
    """Crea un archivo pickle en la ruta especificadas.
    Contendrá la variable indicada.
    Se identificará con el nombre pasado por parámetro o File.pkl en su defecto

    Args:
        df: Dataframe o variable a almacenar
        ruta (str): Ruta de almacenamiento
        nombre (str): Nombre del archivo. SIN extensión. Defaults to File.pkl
    """
    fichero = open(ruta+nombre+'.pkl', 'wb')
    pickle.dump(df, fichero)
    fichero.close()
    
def pickle_to_df(ruta:str,nombre:str):
    """Carga los datos de un archivo pickle
    Devuelve el dato cargado.

    Args:
        ruta (str): Ruta del archivo a cargar
        nombre (str): Nombre del archivo a cargar

    Returns:
        Data: Datos cargados a partir del archivo pickle
    """
    lectura = open(ruta+nombre, 'rb')
    df=pickle.load(lectura)
    lectura.close()
    return df

def info_a_txt(data, path:str, content:str):
    """Traslada la información del dataframe a un archivo txt
    en una ruta determinada

    Args:
        data (DataFrame): Dataframe del que obtener la información
        path (String): Ruta de almacenamiento
        content (String): Nombre del fichero
    """
    buffer = io.StringIO()
    data.info(verbose=True, show_counts=True, buf=buffer)
    with open(path+'info_'+content+'.txt', 'w', encoding='utf-8') as file:
        file.write(buffer.getvalue())
        
def get_dataframe(query:str, conexion:str):
    """Genera un dataframe a partir de una conexión
    y una consulta a una base de datos

    Args:
        query (String): Consulta SQL para obtención de datos
        conexion (String): Cadena de conexión a BD

    Returns:
        Dataframe: Dataframe con la información recogida
    """
    engine = sa.create_engine(conexion)
    df = pd.read_sql(query, conexion)
    return df

def del_columns(df, columnas:list=[]):
    """Elimina las columnas de un datafrmae

    Args:
        df (DataFrame): Dataframe del que eliminar las columnas
        columnas (list, optional): Lista de columnas a eliminar. Defaults to [].

    Returns:
        _type_: _description_
    """
    df = df.drop(axis='columns', columns=columnas)
    return df
    

def presenta_datos(df, spec, path):
    print("Datos para:",spec)
    print("El % de ocupación se mueve entre {} y {}".format(round(df["PorcentOcupacion"][:10].max()*100,2), round(df["PorcentOcupacion"][:datetime.now().month-1].min()*100,2)))
    print("Y el % de ocupación medio anual es: ",round(df["PorcentOcupacion"][:datetime.now().month-1].mean()*100,2))
    plt.figure()
    sns.set_theme(style="darkgrid")
    fig = sns.lineplot(df, x='Data', y='PorcentOcupacion')
    fig.set(xlabel='Mes', ylabel='Ocupación', title=spec)
    fig.savefig(path+spec+'.png')
    
def nulos_en_tbl(df):
    """Muestra información de valores nulos en el dataframe.
    Cuántos hay en cada columna y qué % del total suponen

    Args:
        df (DataFrame): Dataframe objetivo.
    """
    for columna in df:
        print("Valores nulos en columna {}: {}. Un {}% del total"
              .format(columna, df[columna].isna().sum(), round((df[columna].isna().sum()*100)/df.shape[0],2)))

def hipotesis(serie1, serie2=None, tipo_test:str='', alternativa:str="two-sided"):
    """Ejecuta el test de hipotesis identificado por parámetro "tipo_test"
    anderson: Test de Anderson Darling
    shapiro: Test de Shapiro Wilk
    ztest: zTest de hipótesis

    Args:
        serie1 (pandas serie o Dataframe): Serie sobre la que aplicar el test
        tipo_test (str): Tipo de test a ejecutar
        serie2 (pandas Serie o Dataframe, optional): Serie sobre la que realizar la comparación del test. Defaults to None.
        alternativa (str, optional): En el caso de test, la alterativa a usar. Defaults to "two-sided".

    Returns:
        None
    """
    if tipo_test not in ['anderson','shapiro','ztest', 'mann']:
        print("El test especificado no está definido.")
        return None
    
    estadistico = None
    p_valor = None
    crit_val = []
    
    # Por las características de los valores de retorno, el test de Anderson lo tratamos de modo independiente.
    if tipo_test == 'anderson':
        estadistico,crit_val,_ = ss.anderson(serie1)
        p_valor = crit_val[2] # Nos interesa siempre el valor del 5%
        if estadistico > p_valor:
            print("El valor critico {} es menor que el estadístico {}. Por tanto, rechazamos la hipótesis nula.".format(p_valor, estadistico))
        else:
            print("El valor critico {} es mayor que el estadístico {}. Por tanto, se acepta la hipótesis nula".format(p_valor, estadistico))
        return None
    # Para el resto de tipos de test aceptados:
    elif tipo_test == 'shapiro':
        estadistico,p_valor = ss.shapiro(serie1)
    elif tipo_test == 'ztest':
        estadistico, p_valor = sm.stats.ztest(serie1, serie2, alternative = alternativa)
    elif tipo_test == 'mann':
        estadistico, p_valor = ss.mannwhitneyu(serie1, serie2, alternative = alternativa)

    print("El p valor del test de hipótesis {} es: {}".format(tipo_test, p_valor))
    if p_valor < 0.05:
        print("Rechazamos la hipótesis nula")
    else:
        print("Se acepta la hipótesis nula")
    

def calcular_VIF(var_predictoras_df):
    var_pred_labels = list(var_predictoras_df.columns)
    num_var_pred = len(var_pred_labels)
    
    lr_model = LinearRegression()
    
    result = pd.DataFrame(index = ['VIF'], columns = var_pred_labels)
    result = result.fillna(0)
    
    for ite in range(num_var_pred):
        x_features = var_pred_labels[:]
        y_feature = var_pred_labels[ite]
        x_features.remove(y_feature)
        
        x = var_predictoras_df[x_features]
        y = var_predictoras_df[y_feature]
        
        lr_model.fit(var_predictoras_df[x_features], var_predictoras_df[y_feature])
        
        result[y_feature] = 1/(1 - lr_model.score(var_predictoras_df[x_features], var_predictoras_df[y_feature]))
    
    return result

def calc_regresion_scores(ytest, ypredicted):
    
    mse = mean_squared_error(ytest, ypredicted)
    mae = mean_absolute_error(ytest, ypredicted)
    mape = mean_absolute_percentage_error(ytest, ypredicted)
    r2 = r2_score(ytest, ypredicted)
    return [mse, mae, mape,r2]

######################
# Funciones pipeline #
######################

class EliminaColumnas():
    def __init__(self,columns):
        self.columns=columns

    def transform(self,X,y=None):
        self.cols = X.columns
        return X.drop(self.columns,axis=1)

    def fit(self, X, y=None):
        return self

def dia_laborable(x, column):
    x["Dia"] = x[column].dt.day
    x["Mes"] = x[column].dt.month
    x["EsLaborable"] = np.where(x[column].dt.dayofweek>4, 0, 1)
    x["Anyo"] = x[column].dt.year
    return x

def cambia_tipo_fecha(x, column):
    x[column] = pd.to_datetime(x[column], dayfirst=True)
    return x

def tipificacion(x, column):
    x["VisitasEfect_Tipificada"] = ss.zscore(x[column])
    q95_ocupacion = x[column].quantile(0.95)
    x["VisitasAjustada"] = x[column]
    x.loc[x["VisitasEfect_Tipificada"]>3,"VisitasAjustada"] = q95_ocupacion
    return x

def getdummies(x, columns):
    dummies = pd.get_dummies(x[columns], prefix="AreaClinica", prefix_sep='_')
    x[dummies.columns] = dummies
    return x

    