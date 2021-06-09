# Funciones generales para descarga en mysql

import pandas as pd
import sqlalchemy
from copy import copy
from datetime import timedelta, datetime

# Unit Root 
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.stattools import adfuller

# Autocorrelacion
from statsmodels.stats.stattools import durbin_watson

# Vector Autoregression
from statsmodels.tsa.api import VAR

# Linear Regression
import statsmodels.api as sm

# Normality
from scipy import stats

import sys
sys.path.append("/home/ramon/Documents/PythonFunctions")
from rha import Binance

#### FUNCIONES MYSQL
def set_df(crypto, fr = "h", compute = "sum", fiat= "USDT", verbose = False):

    df = from_mysql(crypto, squema = "Twitter", fr = fr, compute = compute)
    df.columns += "_Twitter"
    if verbose:
        print("--- Twitter descargada con exito ---")

    aux = from_mysql(crypto, squema="Reddit", fr = fr, compute=compute)
    aux.columns += "_Reddit"
    
    if verbose:
        print("--- Reddit descargada con exito ---")


    if len(aux) > 0: #and datetime.strptime(aux.index[-1], "%Y-%m-%d").date() >= (date.today() - timedelta(days = 4) ):
        # df = pd.concat([ df, aux ] , axis = 1).dropna()
        df = pd.merge(df,aux, left_index=True, right_index=True, how = "left")
        df.fillna(0, inplace = True)

    if len(df) == 0:
        df = aux

    df.index = pd.to_datetime( df.index )
    df.sort_index(ascending=True, inplace = True)

    cr = {
        'bitcoin':"BTC",
        "ethereum":"ETH",
        "litecoin":"LTC",
        "dogecoin":"DOGE"
    }

    frr = {
        "h":"1h",
        "d":"1h"
    }

    aux = Binance(cr[crypto], 
            df.index[0],  
            df.index[-1] + timedelta(days = 1), 
            frecuencia = frr[fr],
            fiat = fiat)

    if fr == "d":
        aux.df["Open"] = aux.df.resample("D")["Open"].first()
        aux.df["Close"] = aux.df.resample("D")["Close"].last()
        aux.df["High"] = aux.df.resample("D")["High"].max()
        aux.df["Low"] = aux.df.resample("D")["Low"].min()
        aux.df["Volume"] = aux.df.resample("D")["Volume"].sum()
        aux.df["Quotevolume"] = aux.df.resample("D")["Quotevolume"].sum()

    if verbose:
        print("--- Binance descargada con exito ---")

    df = pd.concat([ df, aux.df ] , axis = 1).dropna()

    return df

def from_mysql(table, fr = "h", squema = 'Twitter', compute='sum'):
    """ 
    table (str): "bitcoin", "ethereum", "litecoin", "dogecoin"
    fr (str): Frecuencia para descargar la informacion
                "h" -> hora
                "m" -> minutos
                Default: "h"
    squema (str): "Twitter" o "Reddit". Defualt: "Twitter"
    compute (str): "sum" o "avg". Default: "sum"

    """

    # Descarga de informacion de MySQL
    engine = sqlalchemy.create_engine('mysql+mysqlconnector://root:Guitarra2021-@localhost/{}'.format(squema))

    sep = {
        'd':{'sep':' ', 'places':'1'},
        'h':{'sep':':', 'places':'1'},
        'm':{'sep':':', 'places':'2'},
    }

    cols = {
        "Twitter":['replies_count', 'retweets_count', 'likes_count', 'sentimiento'],
        "Reddit":["comentarios", "score", "upvote_ratio", "sentimiento_titulo", "sentimiento_texto"]
    }

    # substring of separator and place of separation order
    substring = "substring_index(date,'"+ sep[fr]['sep'] +"',"+ sep[fr]['places'] +")"
    
    # Create str of queries
    query_select = "select "+ substring +" as date"
    
    query_sum = ''
    for i in cols[squema]:
        query_sum += ( ', {}({}) as {}'.format(compute, i, i) )
    
    query_from_order = " from "+ table +" group by "+ substring +" order by "+ substring +" desc "
    
    query = query_select + query_sum + query_from_order

    df = pd.read_sql(query, engine)
    
    df.sort_values(by = 'date', ascending=True,inplace = True)
    df.set_index('date', inplace = True)

    return df

#### FUNCIONES ESTADISTICAS
def unit_root_test(df, if_adf = True, if_kpss = True, adf_regression = 'c', kpss_regression='c'):
    """
    OUTPUT: DataFrame with Unit root test.

    Augmented Dickey-Fuller test and KPSS Test para la presencia de 
    Raices unitarias.
    Un valor de 3, significa que en los tres test (1%, 5%, y 10%)
    se pudo rechazar la Null Hypothesis, y la serie es
    estaiconaria.
    Y 0, no se pudo rechazar, y por ende, la serie no es estacionaria.

    La null hypothesis (la data presenta una raiz unitaria) puede ser rechazada de dos maneras:

    1) El valor critico ([0]) sea menor que los T-values al 1%, 5% y 10%.

    2) El p-value ([1]) sea menor que un nivel de significancia al 5% (0.05).

    
    regression (str): 
                'c':constant only (Default)
                'ct':constant and trend
                'ctt':constant, and linear and quatradict trend
                'nc':no constant, no trend

    """
    aux = copy(df)

    aux1 = aux.apply(lambda x: adfuller(x, regression = adf_regression))
    aux2 = aux.apply(lambda x: kpss(x, regression = adf_regression))

    aux1 = aux1.T[[0,1,4]]
    aux2 = aux2.T[[0,1,3]]

    aux1.rename(columns = {0:"T-Statistic", 1:"P-Value"}, inplace = True)
    aux2.rename(columns = {0:"T-Statistic", 1:"P-Value"}, inplace = True)

    aux1 = pd.concat([aux1.drop([4], axis=1), aux1[4].apply(pd.Series)], axis=1)
    aux2 = pd.concat([aux2.drop([3], axis=1), aux2[3].apply(pd.Series)], axis=1)

    aux1.columns = "ADF_" + aux1.columns
    aux2.columns = "KPSS_" + aux2.columns

    return pd.concat([ aux1, aux2 ], axis = 1)

def normality_test(df, mode = 'shapiro', if_help = False):
    """
    Prueba de normalidad.
    0 -> No demuestra normalidad
    1 -> Demuestra normalidad

    mode: 
        shapiro:
        Shapiro-Wilk. La hipothesis nula es: La data esta normalmente distribuida.
            Si p_value < 0.05, entonces la hipothesis nula puede ser rechazada y la data no tiene
            una distribucion normal.
            Sin embargo, p_value > 0.05 y todavia no tener una distribucion normal.
            Pudo haber un problema para rechazarla debido a los pocos datos ingresados.

        ks:
        Kolmogorov-Smirnov Test.
            Prueba si la distribucion de la data fits a CDF (Funcion de Distribucion Acumulativa).
            Esto no es necesariamente para distribuciones normales.
            Al igual que el test de Shapiro la null hypothesis es si la data es igual a la distribucion 
            de la data contra la que estamos comparando.
            Si p_value < 0.05, se rechaza la null hypothesis, y se concluye que la distribucion 
            de nuestra data no es identica a una de distribucion normal.

        anderson:
        Anderson-Darling Test.
            Prueba si la data viene de cierta distribucion, como el ks.
            Se realiza una prueba a diferentes valores de significancia (15%, 10%, 5%, 2.5%, 1%)
            Si su test statistic esta por debajo del nivel de significancia del 5%
            se puede decir que no hay datos que decir que es NO normal, es decir que es normal.
            Si es mayor, entonces su distribucion si es No normal.

    """

    aux = copy(df)
    aux = aux.dropna()

    new = pd.DataFrame(index = aux.columns)

    if mode == 'shapiro':
        for i in aux.columns:
            new.loc[i, mode + '_Statistic'], new.loc[i, mode + '_Pvalue'] = stats.shapiro(aux[i])
        
        new.loc[new[mode + '_Pvalue'] < 0.05, mode + '_Normality'] = 0
        new[mode + '_Normality'].fillna(1, inplace = True)

    elif mode == 'ks':
        for i in aux.columns:
            new.loc[i, mode + '_Statistic'], new.loc[i, mode + '_Pvalue'] = stats.kstest(aux[i], cdf='norm')
            
        new.loc[new[mode + '_Pvalue'] < 0.05, mode + '_Normality'] = 0
        new[mode + '_Normality'] = new[mode + '_Normality'].fillna(1)
    
    elif mode == 'anderson':
        aux_array = []
        for i in aux.columns:
            # new.loc[i, mode + '_Statistic'], cv, *_ = stats.anderson(aux[i], dist = 'norm')
            # new.loc[i, mode + '_Normality'] = len(cv[cv > 0.05]) / len(cv)

            auxx = stats.anderson(aux[i], dist = 'norm')
            for k ,j in zip(auxx.critical_values, auxx.significance_level):
                new.loc[i, mode + '_Statistic'] = auxx.statistic
                new.loc[i, mode + '_Normality'] = 0

                if j <= 10 and auxx.statistic < k:
                    new.loc[i, mode + '_Normality'] = 10
                elif j <= 5 and auxx.statistic < k: 
                    new.loc[i, mode + '_Normality'] = 5
                elif j <= 1 and auxx.statistic < k: 
                    new.loc[i, mode + '_Normality'] = 1
            

    if if_help:
        print("0 -> No normal")
        print("1 -> Normal")

    return new

def durbin_watson_test(df, mode = 'VAR', target = None):
    """
    El metodo de Durbin-Watson testea la presencia de autocorrelación
    en la Data.
    Si el valor se encuentra entre 1.5 y 2.5, se puede rechazar la 
    null hypothesis, y por ende, no hay presencia de autocorrelación.

    Entre 0 y 2, autocorrelación positiva.
    Entre 2 y 4, autocorrelación negativa.

    mode:   Linear, with Ordinary Least Square estimation.
            VAR, Vector autoregression (Lag 3)
    
    target: if VAR, target puede ser None para autoregression
            If != None, Se genera autoregression con un target
            o variable exogenea.
    """

    aux = copy(df)

    if target is not None:
        X = aux.drop(columns = [target])
        y = pd.DataFrame(aux[target])

    if mode == 'VAR':
        mod = VAR(X, exog = y) if target is not None else VAR(aux)
    
    elif mode == 'Linear':
        mod = sm.OLS(X, y)

    res = mod.fit(3)

    dbt = durbin_watson(res.resid)

    dbt = pd.DataFrame(dbt)

    if target is not None:
        dbt.index = X.columns
        col_name = target
    else:
        dbt.index = aux.columns
        col_name = 'Durbin_Watson'
    
    dbt.columns = [col_name]

    dbt['Auto-Correlation'] = dbt[(dbt[col_name]>= 1.5) & (dbt[col_name]<=2.5)]
    dbt['Auto-Correlation'] = ~dbt['Auto-Correlation'].isnull()
    dbt['Auto-Correlation'] = ~dbt['Auto-Correlation']

    return dbt

if __name__ == "__main__":
    crypto = "BTC"
    fiat = "USDT"
    frecuencia = "1d"

    from datetime import date, timedelta, datetime

    end = datetime.today() #+ timedelta(days = 1)
    start = end - timedelta(days = 2)

    b = Binance(
        crypto,
        start,
        end,
        frecuencia = frecuencia,
        fiat = fiat
    )