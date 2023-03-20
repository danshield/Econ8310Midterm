import numpy as np
import os.path
import pandas as pd
import plotly.express  as px
import statsmodels.api as sm
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing
import statsmodels.tsa.stattools as st
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pandas as pd
import numpy as np

def get_dataframe(file):
    dsRoot  = "./data"
    return pd.read_csv(os.path.join(dsRoot, file))


def order_date(x):
    return dfOrders[dfOrders['order_id'] == x['order_id']]['order_purchase_timestamp'].iloc[0]


def fix_dataset(ds):
    if any(ds.isnull()):
        ds.dropna(inplace=True)

        
def map_category_code(cat):
    tmp = dfTranslation[dfTranslation['product_category_name'] == cat]['product_category_name_english']
    if tmp.empty:
        return None
    else:
        return tmp.iloc[0]        
        

#def get_pct_change(x, periods):
#    return x['Total'].pct_change(periods=periods).to_list()[-1:][0]
    
    
def get_moving_avg(tbl, col, periods, rolling):
    tmp = tbl[col].pct_change(periods=periods)
    tmp = tmp.rolling(rolling).mean().to_frame()
    return tmp
    
def get_ADF(ds):
    return st.adfuller(ds)
    
       
#dfCust        = get_dataframe("olist_customers_dataset.csv")
#dfGeo         = get_dataframe("olist_geolocation_dataset.csv")
dfItems       = get_dataframe("olist_order_items_dataset.csv")
#dfPayments    = get_dataframe("olist_order_payments_dataset.csv")
#dfReviews     = get_dataframe("olist_order_reviews_dataset.csv")
dfOrders      = get_dataframe("olist_orders_dataset.csv")
dfProducts    = get_dataframe("olist_products_dataset.csv")
#dfSellers     = get_dataframe("olist_sellers_dataset.csv")
dfTranslation = get_dataframe("product_category_name_translation.csv")
#datasets = [dfCust, dfGeo, dfItems, dfPayments, dfReviews, dfOrders, dfProducts, dfSellers, dfTranslation]

#Convert columns to datetime
for i in range(3,8):
    dfOrders[dfOrders.columns[i]] = dfOrders[dfOrders.columns[i]].apply(pd.to_datetime)

dfItems['shipping_limit_date'] = dfItems['shipping_limit_date'].apply(pd.to_datetime)


# add the purchase timestamp to the items data and drpp the time portion
ds = dfItems.merge(dfOrders[['order_id', 'order_purchase_timestamp']], on=['order_id'])
ds['order_purchase_timestamp'] = ds['order_purchase_timestamp'].apply(lambda r:r.normalize())

# add the category info to the items data
ds = ds.merge(dfProducts[['product_id','product_category_name']], on=['product_id'])

# We have merged some empty categories into the table. Rather than create a category,
# just drop the ~1600 records from the 110K+ table.
fix_dataset(ds)
ds['product_category_name'] = ds['product_category_name'].apply(lambda r:map_category_code(r))
ds = ds.rename(columns={"order_purchase_timestamp":"timestamp", 'product_category_name':'category'}) 

###################################################################################################
#
# A three-month forecast of future sales (number and figures)
#
# Get sales/month, ignoring categories. Do this for both $$$ and # of items
sales = ds.groupby(pd.Grouper(key='timestamp', freq='M')).agg({'price':['sum'], 'order_item_id':['count']})
sales = pd.DataFrame(sales[4:-1]) # sporatic data for the first 4 months and the last month
sales.columns = ['total','count']
sales['unit_cost'] = sales['total'].div(sales['count'], axis=0).fillna(0)

px.scatter(sales, y='total', trendline='ols').show()
###################################################################################################

###################################################################################################
#
# Determine the 3 best-selling categories, and create a forecst of growth in those categories
#
# 1. health_beauty
# 2. watches_gifts
# 3. bet_bath_table
#
price = ds.groupby('category').agg({"price":['sum']})
price.columns = ['Total']
price = price.reset_index().sort_values('Total', ascending=False)

px.bar(price[:10], x='category', y='Total').show()


# We've got the dataframe we need. Start grouping/sorting to get category values
grp = ds.groupby([pd.Grouper(key='timestamp', freq='M'), 'category']).agg({ 'price': ['sum'], 'order_item_id': ['count']})
grp.columns = ['Total', 'Items']
grp = grp.reset_index()
###################################################################################################



###################################################################################################
#
# Determine the fastest growing category, and create a forecast for its growth
#
# telephoney is the fastest growing category.
#
f = grp.groupby('category')
#for i in range(1, 7):
#    pct_change = [(tbl[0], get_pct_change(f.get_group(tbl[0]), i)) for tbl in f]
#    pc = pd.DataFrame(pct_change, columns=['category','pct_change'])
#    pc = pc.sort_values('pct_change', ascending=False)
#
#    # Get the top 5 potential
#    tbls = [f.get_group(c) for c in pc[:5]['category']]
#    graph = pd.concat(tbls).reset_index(drop=True).sort_values('timestamp')
#
#    px.scatter(graph, x='timestamp', y='Total', color='category', trendline='ols').show()

    
for i in range(1,4):
    rolling_change = [(tbl[0], get_moving_avg(f.get_group(tbl[0]), 'Total', i, 3)) for tbl in f]
    series = [rc[1].reset_index(drop=True) for rc in rolling_change]
    chart = pd.concat(series, axis=1, ignore_index=True)
    chart.columns = [rc[0] for rc in rolling_change]
    chartT = chart.T

    # Look at all rows with a value in the final period and take the last 5 periods
    # sum the rows and the largest total is the fastest growing category
    rows = chartT[(chartT[len(chart)-1] >= 0) | (chartT[len(chart)-1] < 0)].iloc[:,len(chart)-5:len(chart)]
    graph = rows.sum(axis=1).reset_index()
    graph.columns = ['category', 'moving_avg']
    px.bar(graph.sort_values('moving_avg', ascending=False), x='category', y='moving_avg').show()
###################################################################################################

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
# Predict 90 days for all sales
for freq in ['D','W']:
    tot_sales = ds.groupby(pd.Grouper(key='timestamp', freq=freq)).agg({'price':['sum'], 'order_item_id':['count']}) 
    tot_sales = pd.DataFrame(tot_sales[28:] if freq == 'D' else tot_sales[17:])
    tot_sales.columns = ['total','count'] 

    px.scatter(tot_sales, y='total', trendline='ols').show() 

    test_len = 90 if freq == 'D' else 13
    train, test = train_test_split(tot_sales, test_len=test_len)

    trend = ExponentialSmoothing(train['total'], initialization_method='estimated', trend='add', damped_trend=True).fit()
    trends = trend.forecast(len(test))

    chart = pd.merge(test['total'], trends.to_frame('proj'), left_index=True, right_index=True).reset_index()
    chart = chart.melt(id_vars='timestamp')

    tmp = train.reset_index()
    tmp = tmp[['timestamp','total']].melt(id_vars='timestamp') 

    chart = pd.concat([tmp, chart])
    px.line(chart, x='timestamp', y='value', color='variable').show()

###################################################################################################
# Predict the next 90 days for the fastest growing (telephony)
# 
freq = 'W'
fg = ds[ds['category'] == 'telephony']
fg_sales = fg.groupby(pd.Grouper(key='timestamp', freq = freq)).agg({'price':['sum']})
fg_sales = pd.DataFrame(fg_sales[17:])
fg_sales.columns = ['total']

px.scatter(fg_sales, y='total', trendline = 'ols').show()

test_len = 90 if freq == 'D' else 13
train, test = train_test_split(fg_sales, test_len=test_len)

trend = ExponentialSmoothing(train['total'], trend='add', seasonal="add", seasonal_periods=4).fit()
trends = trend.forecast(len(test))

chart = pd.merge(test['total'], trends.to_frame('proj'), left_index=True, right_index=True).reset_index()
chart = chart.melt(id_vars='timestamp')

tmp = train.reset_index()
tmp = tmp[['timestamp','total']].melt(id_vars='timestamp') 

chart = pd.concat([tmp, chart])
px.line(chart, x='timestamp', y='value', color='variable').show()



t = pd.DataFrame(ds[ds['category'] == 'telephony'][['timestamp','price','category']])
tg = t.groupby(pd.Grouper(key='timestamp', freq='W')).agg({'price':['sum']})
tgd = tg.diff().fillna(0)


