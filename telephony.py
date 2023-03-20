import numpy as np
import os.path
import pandas as pd
import plotly.express  as px
import statsmodels.api as sm
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing
import statsmodels.tsa.stattools as st
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pygam import LinearGAM, s, f, l

def train_test_split(ds, train_pct=0.0, test_len = 0):
    length = len(ds)
    if train_pct > 0:
        train_sz = int(length * train_pct)
        test_sz = length - train_sz
    elif test_len > 0:
        train_sz = length - test_len
        test_sz = test_len
    
    return ds[:train_sz], ds[-test_sz:]

###################################################################################################

ds = pd.read_csv('./ds.csv')

t = pd.DataFrame(ds[ds['category'] == 'telephony'])
t['timestamp'] = t['timestamp'].apply(lambda x: pd.to_datetime(x).normalize())

tg = t.groupby(pd.Grouper(key='timestamp', freq='M')).agg({'price':['sum']})
tg.columns = ['total']
tg = tg[4:]
tg = tg.reset_index()
#tg[['year', 'month','day']] = [(ts.year, ts.month, ts.day) for ts in tg['timestamp']]

train, test = train_test_split(tg, test_len = 3)

trend = ExponentialSmoothing(train['total'], trend='mul').fit()
trends = trend.forecast(len(test))
                        
test = pd.merge(test, trends.to_frame('proj'), left_index=True, right_index=True)
chart = pd.concat([train.melt(id_vars='timestamp'), test.melt(id_vars='timestamp')])
px.line(chart, x='timestamp', y='value', color='variable').show()

