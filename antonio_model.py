#!/usr/bin/env python
# coding: utf-8

# In[147]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import re

# Load data
data = pd.read_csv(r"C:\Google One\Work\football\Master_Training 2023_24.csv")


# In[148]:


data['Duration'] = round(data['Total Time'].apply(lambda x: np.nan if pd.isnull(x)
                                                  else int(x[:2]) * 60 + int(x[3:5]) + int(x[6:8]) / 60), 1)


# In[149]:


data['Z4 relative distance (m/min)'] = round(data['Distance Zone 4 (Absolute)'] / data['Duration'], 1)
data['Z5 relative distance (m/min)'] = round(data['Distance Zone 5 (Absolute)'] / data['Duration'], 1)
data['Z6 relative distance (m/min)'] = round(data['Distance Zone 6 (Absolute)'] / data['Duration'], 1)
data['Relative sprint distance (m/min)'] = round(data['Sprint Distance'] / data['Duration'], 1)


# In[150]:


data_update = data[['Session', 'Day', 'Turnover', 'Player Last Name', 'Drill Title', 'Duration', 'Total Distance',
                    'Distance Zone 4 (Absolute)', 'Distance Zone 5 (Absolute)', 'Distance Zone 6 (Absolute)',
                    'HSR Per Minute (Absolute)', 'Sprint Distance', 'Accelerations Per Min', 'Decelerations Per Min',
                    'Max Speed']].copy()


# In[151]:


data_update.columns = ['Session', 'Day', 'Turnover', 'Player_Name', 'Drill', 'Duration', 'Relative_distance',
                       'Z4_relative_distance', 'Z5_relative_distance', 'Z6_relative_distance', 'Relative_HSD',
                       'Relative_sprint_distance', 'Relative_accelerations', 'Relative_decelerations', 'Max_speed']


# In[152]:


data_update['Session'] = data_update['Session'].astype('category')
data_update['Day'] = pd.Categorical(data_update['Day'], categories=['MD+1', 'MD+2', 'MD+3', 'MD-4', 'MD-3', 'MD-2', 'MD-1', 'MD'])
data_update['Player_Name'] = data_update['Player_Name'].astype('category')
data_update['Drill'] = data_update['Drill'].apply(lambda x: re.sub("Set \\d+", "", x))


# In[153]:


model_vars = ['Relative_distance', 'Z4_relative_distance', 'Z5_relative_distance', 'Z6_relative_distance',
              'Relative_HSD', 'Relative_sprint_distance', 'Relative_accelerations', 'Relative_decelerations',
              'Max_speed']


# In[154]:


data_update['Drill']


# In[155]:


data_update['Drill'] = data_update['Drill'].str.rsplit(',', 1).str[0]


# In[157]:


# # Model data - team analysis per match day


# In[158]:


model_formula = "Max_speed ~ Day + Duration + Relative_distance + Z4_relative_distance + Z5_relative_distance + Z6_relative_distance + Relative_HSD + Relative_sprint_distance + Relative_accelerations + Relative_decelerations"


# In[159]:


model = smf.mixedlm(model_formula, data=data_update, groups=data_update["Drill"]).fit()



# In[ ]:




