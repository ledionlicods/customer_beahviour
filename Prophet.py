#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import libraries

from fbprophet import Prophet
from fbprophet.plot import plot_plotly
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.offline as py
py.init_notebook_mode()
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


#import datasets
acc=pd.read_csv('ACC.csv',header=None)
parf=pd.read_csv('PARF.csv',header=None)
clothes=pd.read_csv('CL.csv',header=None)
total=pd.read_csv('s2019.csv',header=None)
holidays=pd.read_csv('holidays.csv',header=None)
total.info()


# In[9]:


#create total training set
train_tot= pd.DataFrame()
train_tot['ds'] = pd.to_datetime(total[0])
train_tot['y']=total[1]
train_tot[0:5]


# In[5]:


#create total training set for cosmetics
train_parf= pd.DataFrame()
train_parf['ds'] = pd.to_datetime(parf[0])
train_parf['y']=parf[1]
train_parf


# In[315]:


#create total training set for clothes
train_clothes= pd.DataFrame()
train_clothes['ds'] = pd.to_datetime(clothes[0])
train_clothes['y']=clothes[1]
train_clothes


# In[17]:


#initialize and train baseline Prophet model
prophet_model = Prophet()
prophet_model.fit(train_tot)


# In[18]:


# predict values for 300 days
future= prophet_model.make_future_dataframe(periods=300)
future.tail(2)


# In[19]:


# plot rime series data
fig = plt.figure(facecolor='w', figsize=(10, 6))
plt.plot(train_tot.ds, train_tot.y)


# In[20]:


#plot predicted data
forecast=prophet_model.predict(future)
fig1 =prophet_model.plot(forecast)


# In[21]:


#plot components
fig1 = prophet_model.plot_components(forecast)


# In[13]:


#check changepoints
from fbprophet.plot import add_changepoints_to_plot 
fig= prophet_model.plot(forecast)
a = add_changepoints_to_plot(fig.gca(), prophet_model, forecast)


# In[11]:



# add holiday season data
holiday_season = pd.DataFrame({
  'holiday': holidays[0],
  'ds': pd.to_datetime(holidays[1]),
  'lower_window': -1,
  'upper_window': 0,
})
holiday_season.head()


# In[15]:


#modify prophet algorithm adding holidat and creating a new periodicity and plot new data

pro_holiday= Prophet(holidays=holiday_season,holidays_prior_scale = 25).add_seasonality(name='bimonthly',period=365.25/6,fourier_order=5).add_seasonality(name='weekly',period=7,fourier_order=20)
pro_holiday.fit(train_tot)
future_data = pro_holiday.make_future_dataframe(periods=12, freq = 'm')
 
#forecast the data for future data
forecast_data = pro_holiday.predict(future_data)
pro_holiday.plot(forecast_data);


# In[183]:


#modify prophet algorithm adding holidat and creating a new periodicity and plot new data
pro_holiday= Prophet(holidays=holiday_season,holidays_prior_scale = 10,changepoint_prior_scale=10,seasonality_mode='additive',interval_width = 0.6, growth='linear', yearly_seasonality = False,
                     weekly_seasonality=False).add_seasonality(name='yearly',period=365.25,fourier_order=10).add_seasonality(name='quaterly',period=365.25/4,fourier_order=5)

pro_holiday.fit(train_tot)
future_data = pro_holiday.make_future_dataframe(periods=12, freq = 'm')
 
#forecast the data for future data
forecast_data = pro_holiday.predict(future_data)
pro_holiday.plot(forecast_data);


# In[16]:


#Cross validate the first model
from fbprophet.diagnostics import cross_validation
df_cv2 = cross_validation(pro_holiday, initial='2000 days', period='180 days', horizon = '180 days')
# Python
from fbprophet.diagnostics import performance_metrics
df_ph = performance_metrics(df_cv2)
df_ph[50:100]


# In[78]:


#Cross validate the modified  model
from fbprophet.diagnostics import cross_validation
df_cv1 = cross_validation(prophet_model, initial='2000 days', period='90 days', horizon = '90 days')
from fbprophet.diagnostics import performance_metrics
df_p = performance_metrics(df_cv1)
df_p[50:100]


# In[18]:


#plot MAPE metric for modified model
plt.plot( df_ph['mape'][0:150], 'b') # plotting t, b separately

plt.title('MAPE e Modelit te modifikuar')
plt.xlabel('Diapazoni(Dite)')
plt.ylabel('MAPE')


# In[83]:


#plot time series prediction for original  model compared to truth

from sklearn.metrics import mean_absolute_error


# calculate MAE between expected and predicted values for december
y_true = train_tot[(train_tot['ds'] >= '2019-01-01') & (train_tot['ds'] <= '2019-12-30') ]['y'].values
y_pred = forecast_data[(forecast_data['ds'] >= '2019-01-01') & (forecast_data['ds'] <= '2019-12-30')]['yhat'].values
mae = mean_absolute_error(y_true, y_pred)
print('MAE: %.3f' % mae)
# plot expected vs actual
plt.plot(y_true, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.show()


# In[84]:


## #plot time series prediction for modified   model compared to truth


# calculate MAE between expected and predicted values for december
y_true = train_tot[(train_tot['ds'] >= '2019-01-01') & (train_tot['ds'] <= '2019-12-30') ]['y'].values
y_pred = forecast[(forecast['ds'] >= '2019-01-01') & (forecast['ds'] <= '2019-12-30')]['yhat'].values
mae = mean_absolute_error(y_true, y_pred)
print('MAE: %.3f' % mae)
# plot expected vs actual
plt.plot(y_true, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.show()


# In[1]:


# covert jupyter notebook to the python file 
get_ipython().system('jupyter nbconvert --to script DM_Project.ipynb')


# In[ ]:




