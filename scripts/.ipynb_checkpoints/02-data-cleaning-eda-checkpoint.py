#!/usr/bin/env python
# coding: utf-8

# # COVID-19 Vaccine Forecasting
# 
# ## Part 2 - Data Cleaning

# In[1]:


# Imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from tensorflow.keras.utils import set_random_seed

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.vector_ar.vecm import coint_johansen

import os
import random


# In[2]:


# create function that sets OS `PYTHONHASHSEED` environment variable,
# Python, NumPy, and TensorFlow seeds at a fixed value
def reset_seeds():
    os.environ['PYTHONHASHSEED']=str(42)
    random.seed(42)
    np.random.seed(42)
    set_random_seed(42)

reset_seeds()


# In[3]:


# read in data into a pandas dataframe
# for reproducibility, a CSV was saved of the data that was available at the time this project was completed
df = pd.read_csv('../data/us_state_vaccinations_09_04_21.csv')


# In[4]:


# first look at the data
df.head()


# In[5]:


# set the index
df.set_index(df['date'], inplace=True)

# sort the index
df.sort_index(axis=1, inplace=True)

# check dataframe 
df.head()


# In[6]:


# set the index to datetime
df.index = pd.to_datetime(df.index)


# In[7]:


# confirm that the index is in datetime format
df.info()


# In[8]:


# create a mask to set the dataframe equal to the state of massachusetts
mask = df['location'] == 'Massachusetts'
df = df[mask]


# In[9]:


# check for null values
df.isnull().sum()


# In[10]:


df.head()


# In[11]:


# check indexes of missing values
missingv = df["total_vaccinations"].isnull().to_numpy().nonzero()
print(missingv)


# In[12]:


# interpolate missing values
# since missing values are all float objects, I can interpolate on the whole dataframe
df.interpolate(inplace=True)


# In[13]:


# check for remaining missing values
df.isnull().sum()


# In[14]:


# drop booster-related columns
df.drop(columns = ['total_boosters', 'total_boosters_per_hundred'], axis = 1, inplace=True)

# drop first row
df.dropna(inplace=True)

# check shape
df.shape


# In[15]:


# drop the location column
df.drop('location', axis=1, inplace=True)

# check dataframe
df.head()


# In[16]:


# check for missing dates in the timeseries
pd.date_range(start = "2021-01-13", end = "2021-09-04").difference(df.index)


# There are no remaining missing datapoints in the data!

# In[17]:


# # save clean csv
# df.to_csv("../data/massachusetts_vaccinations_09_04_21_clean.csv", index=False)


# ## Exploratory Data Analysis
# 
# ### Plotting features of interest to evaluate trend

# In[18]:


# Look at each column
df.columns


# In[19]:


# create list of features of interest
features = ["total_distributed", "total_vaccinations", "people_vaccinated", "people_fully_vaccinated", "daily_vaccinations"]


# In[20]:


# plot features
fig, ax = plt.subplots(figsize=(10,8))

ax.yaxis.get_major_formatter().set_scientific(False)

plt.title("Massachusetts Vaccine Data")
plt.xlabel('Date')
plt.ylabel('Vaccines')
for feature in features:
    plt.plot(df[feature], label = feature)
    plt.legend();


# "people_fully_vaccinated", "people_vaccinated", "total_vaccinations", and "total_distributed" all have positive trends.

# In[21]:


# check correlation of features of interest
df[features].corr()


# In[22]:


# visualize correlation of the features
plt.figure(figsize=(8,8))
mask = np.triu(np.ones_like(df[features].corr(), dtype=bool))
plt.title("Correlation of features")
sns.heatmap(df[features].corr(), mask=mask, annot=True, cmap="viridis", center=0,
            square=True, linewidths=.5);


# All features of interest other than "daily_vaccinations" are strongly correlated with each other. In time series, this may not hold any signifiance, as features with low correlation could be highly influential in the target of interest (assuming the time series isn't a random walk).
# 
# Because "people vaccinated" and "people_fully_vaccinated" are affected by whether an individual is getting a single-dose or a double-dose vaccine, those features may be harder to model and predict since I don't have data regarding the proportion of single-dose and double-dose vaccines administered.

# In[23]:


# decompose total vaccinations into trend, seasonal, and residual components.
decomp = seasonal_decompose(df['total_vaccinations'])

# plot the decomposed time series
decomp.plot();


# In[24]:


# plot autocorrelation function for total_vaccinations
plot_acf(df['total_vaccinations'], lags=20)
plt.title("Total Vaccinations Autocorrelation")
plt.show();


# In[25]:


# plot partial autocorrelation function for total_vaccinations
plot_pacf(df['total_vaccinations'], lags=20);
plt.title("Total Vaccinations Partial Autocorrelation")
plt.show();


# Autocorrelation plot confirms the positive trend for "total_vaccinations", and the partial autocorrelation plot shows that time T-1 is a significant predictor of time T.

# In[26]:


# Code written by Joseph Nelson.

def interpret_dftest(dftest):
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value', '#Lags Used', 'Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    return dfoutput

# Run ADF test on original (non-differenced!) data using the 
# constant and trend regression pattern for total vaccinations
interpret_dftest(adfuller(df['total_vaccinations'], regression = 'ct'))


# Adjusted dickey-fuller test suggest that total vaccinations is stationary at an alpha level of 0.05 on original non-differenced data

# In[27]:


# Run ADF test on differenced data using the regular regression pattern for total vaccinations
interpret_dftest(adfuller(df['total_vaccinations'].diff().dropna()))


# In[28]:


# Run ADF test on twice differenced data using the regular regression pattern for total vaccinations
interpret_dftest(adfuller(df['total_vaccinations'].diff().diff().dropna()))


# Adjusted dickey-fuller test suggests that "total_vaccinations" is stationary at an alpha level of 0.05 on undifferenced data.

# In[29]:


# decompose daily vaccinations into trend, seasonal, and residual components.
decomp = seasonal_decompose(df['daily_vaccinations'])

# Plot the decomposed time series.
decomp.plot();


# In[30]:


# plot autocorrelation function for daily_vaccinations
plot_acf(df['daily_vaccinations'], lags=20)
plt.title("Daily Vaccinations Autocorrelation")
plt.show();


# In[31]:


# plot partial autocorrelation function for daily_vaccinations
plot_pacf(df['daily_vaccinations'], lags=20);
plt.title("Daily Vaccinations Partial Autocorrelation")
plt.show();


# Autocorrelation plot shows a positive trend for "daily_vaccinations", and partial autocorrelation plot may suggest some sort of seasonality, although this is most likely just a reflection of the variance in daily vaccinations.

# In[32]:


# Run ADF test on original (non-differenced!) data using the 
# constant and trend regression pattern for daily vaccinations
interpret_dftest(adfuller(df['daily_vaccinations'], regression="ct"))


# In[33]:


# Run ADF test on original (non-differenced!) data using the regular regression pattern for total vaccinations
interpret_dftest(adfuller(df['daily_vaccinations'].diff().dropna()))


# In[34]:


# Run ADF test on differenced data
interpret_dftest(adfuller(df['daily_vaccinations'].diff().diff().dropna()))


# Adjusted dickey-fuller test suggest that daily vaccinations is stationary at an alpha level of 0.05 on  twice-differenced data.
