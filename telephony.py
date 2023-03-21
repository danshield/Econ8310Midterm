import pandas as pd
import plotly.express  as px
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

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

# Load the pre-processed data and set the datetime columns
ds = pd.read_csv('./ds.csv')
t = pd.DataFrame(ds[ds['category'] == 'telephony'])
t['timestamp'] = t['timestamp'].apply(lambda x: pd.to_datetime(x).normalize())

# Group the data my monthly sales. There are so few sales in the first 4 momths
# we will drop them.
tg = t.groupby(pd.Grouper(key='timestamp', freq='M')).agg({'price':['sum']})
tg.columns = ['total']
tg = tg[4:]
tg = tg.reset_index()
#tg[['year', 'month','day']] = [(ts.year, ts.month, ts.day) for ts in tg['timestamp']]

# Split the data and create the model and forecast
train, test = train_test_split(tg, test_len = 3)

trend = ExponentialSmoothing(train['total'], trend='mul').fit()
trends = trend.forecast(len(test))
                        
test = pd.merge(test, trends.to_frame('proj'), left_index=True, right_index=True)
chart = pd.concat([train.melt(id_vars='timestamp'), test.melt(id_vars='timestamp')])
px.line(chart, x='timestamp', 
               y='value', 
               color='variable',
               title = 'Fastest Growing Category - Telephony',
               labels = {
                   'value': 'Total Sales',
                   'x' : 'Date',
                   'variable' : 'Telephony'
                   }).show()

