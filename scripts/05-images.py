#!/usr/bin/env python
# coding: utf-8

# # COVID-19 Vaccine Forecast
# 
# ## Part 5 - Images

# In[1]:


import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# In[2]:


model_1 = keras.models.load_model('../data/model_1.h1')


# In[3]:


# plotting model architecture
tf.keras.utils.plot_model(model_1, to_file="../images/model_1.png", show_shapes=True)


# In[ ]:




