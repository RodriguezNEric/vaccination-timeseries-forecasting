#!/usr/bin/env python
# coding: utf-8

# # COVID-19 Vaccine Forecasting
# 
# ## Part 1 - Data Collection

# In[151]:


# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[152]:


# load in data directly from raw github URL
url = 'https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/vaccinations/us_state_vaccinations.csv'

# create dataframe from url
df = pd.read_csv(url)

# save dataframe during project update
todays_date = "09_04_21"
df.to_csv(f'../data/us_state_vaccinations_{todays_date}.csv', index=False) 