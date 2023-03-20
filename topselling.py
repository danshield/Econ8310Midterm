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

freq = 'W'
ds = pd.read_csv('./ds.csv')
ds['timestamp'] = ds['timestamp'].apply(lambda x: pd.to_datetime(x).normalize())

top_selling = ['health_beauty', 'watches_gifts', 'bed_bath_table']
t = pd.DataFrame(ds[ds['category'].isin(top_selling)])[['price','timestamp','category']]

cols    = [(c, t[t['category'] == c]) for c in top_selling]  
series  = [(col[0], col[1].reset_index(drop=True)[['price', 'timestamp']]) for col in cols]
series  = [(s[0], s[1].groupby(pd.Grouper(key='timestamp', freq=freq)).agg({'price':['sum']})) for s in series]

# Have to fix the column names before the joins below will work
for i in range(len(series)):
    series[i][1].columns = [series[i][0]]
tables = [pd.DataFrame(s[1]) for s in series]

t1 = tables[0]
t2 = tables[1]
t3 = tables[2]

x = t1.join(t2, how='left', on='timestamp').join(t3, how='left', on='timestamp')
x = x.reset_index()
x = x[15:-1]
chart = x.melt(id_vars='timestamp')

px.line(chart, x='timestamp', y='value', color='variable').show()

train, test = train_test_split(x, test_len = 13)
test.set_index('timestamp', inplace=True)

model = sm.tsa.VAR(train[top_selling])
#print(model.select_order().summary())

# 12 weeks/3 months seems to give the best results
for i in range(12,13):
    modelFit = model.fit(i)
    fcast = pd.DataFrame(modelFit.forecast(y=train[top_selling][-26:-13].values, steps=13), columns = top_selling)

    figs = [px.line(x = test.index, 
                    y=[test[ts], fcast[ts]],
                    title = ts, labels={
                        'value' : ts,
                        'x' : 'Date',
                        'variable' : 'Series'
                    }) for ts in top_selling]

    for i in range(len(figs)):
        figs[i].data[0].name = 'Truth'
        figs[i].data[1].name = 'Forecast'
        
        figs[i].show()
    