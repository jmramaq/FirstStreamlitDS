
import pandas as pd
import numpy as np
import pickle
import scipy.stats as ss



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

    