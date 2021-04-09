#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Data is provided from Lake County, IL. Demographics which is based on 27 records or observations
# Source: Data.Gov

import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
import chart_studio.plotly as py
import seaborn as sns, bokeh
from scipy import stats
import cufflinks as cf
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

get_ipython().run_line_magic('matplotlib', 'inline')

NoHighSchool = np.array([5.7,2.8,2.5,11.2,6.2,6,3.8,18.8,9.7,3.4,2.1,4.7,4.1,3.9,11.6,5,32.1,3.2,21.1,4.2,11.2,33.6,20.3,3.1,8.7,16.5,7.5])
Poverty = np.array([10.5,4.9,3.5,11.2,5.5,4.9,6.7,13.3,11.4, 5.4,4.5,4.3,3.9,3.5,9,3,30.5,4.4,12.8,1.4,7.2,23.1,13.6,4.5,7.2,14.5,8.9])
Med_Income = np.array([75928,122789,130140,55431,81055,83500,115372,54176,63792,93849,152658,83104,127561,122704,79283,91426,33009,100833,64393,92094,73473,40907,52220,96804,75255,55098,56583])

# plt.title('Poverty vs. No HS Diploma')
# plt.ylabel('Poverty')
# plt.xlabel('No HS Diploma')
# plt.scatter(NoHighSchool,Poverty, label='No HS Education', color = 'purple', alpha=0.5)


BSdegree = np.array([29.3,64.3,71.5,22.7,45.5,45.2,69.5,32.6,23.1,69.6,75.1,39.9,58.2,65.4,40.8,60,12.2,62.6,24.3,44.4,34,14.4,18.2,63.7,23.7,17.2,20.5])
Poverty = np.array([10.5,4.9,3.5,11.2,5.5,4.9,6.7,13.3,11.4,5.4,4.5,4.3,3.9,3.5,9,3,30.5,4.4,12.8,1.4,7.2,23.1,13.6,4.5,7.2,14.5,8.9])

df = pd.read_csv('Demographics.csv')
# df = pd.read_csv('data.csv')
# plt.title('Lake county IL. Demographics ')
# plt.ylabel('Poverty')
# plt.xlabel('BS Degree')
# plt.scatter(BSdegree,Poverty, label='With BS Degrees', color = '#88c999', alpha=1)

# plt.legend()
# plt.show()


# In[2]:


init_notebook_mode(connected=True)


# In[3]:


cf.go_offline()


# # Define a Linear Regression Function in Python to find the relationship b/w variables.
# 
# # In Machine Learning and in statistical modeling that relationship is used to  predict the outcome of future events.
# 

# In[15]:


slope, intercept, r, p, std_err = stats.linregress(BSdegree,Poverty) # Linear Regression

def lin_regress(BSdegree):
  return slope * BSdegree + intercept

my_model = list(map(lin_regress, BSdegree))

plt.scatter(BSdegree,Poverty, color='purple', alpha=0.5)
plt.title('Lake county IL. Demographics ')
plt.ylabel('Poverty')
plt.xlabel('BS Degree')
plt.plot(BSdegree, my_model)
plt.show()
print(r)
print(p)


# In[19]:


slope, intercept, r, p, std_err = stats.linregress(Med_Income,Poverty) # Linear Regression

def lin_regress(Med_Income):
  return slope * Med_Income + intercept

my_model = list(map(lin_regress, Med_Income))

plt.scatter(Med_Income,Poverty, color='purple', alpha = 0.5)

new_income = lin_regress(70000) # Predict Future Values by passing in arguments for Med_Income



plt.title('Lake county IL. Demographics ')
plt.ylabel('Poverty')
plt.xlabel('Medium Income')
plt.plot(Med_Income, my_model)
plt.show()
print(r)
print(new_income)


# In[21]:


slope, intercept, r, p, std_err = stats.linregress(NoHighSchool,Poverty) # Linear Regression

def lin_regress(NoHighSchool):
  return slope * NoHighSchool + intercept

my_model = list(map(lin_regress, NoHighSchool))

plt.scatter(NoHighSchool,Poverty, color='purple', alpha = 0.5)

new_value = lin_regress(25) # Predict Future Values by passing in arguments for NoHighSchool



plt.title('Lake county IL. Demographics ')
plt.ylabel('Poverty')
plt.xlabel('No HS Diploma')
plt.plot(NoHighSchool, my_model)
plt.show()
print(r)
print(new_value)


# In[20]:


print(std_err)


# In[7]:


df.head()


# In[8]:


len(df)


# In[9]:


sns.displot(df['Med_Income'], kde = True, color = 'purple')


# In[10]:


sns.displot(df['Poverty'], kde = True, color = 'gold')


# In[11]:


import plotly.graph_objects as go

import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2011_us_ag_exports.csv')

fig = go.Figure(data=go.Choropleth(
    locations=df['code'], # Spatial coordinates
    z = df['total exports'].astype(float), # Data to be color-coded
    locationmode = 'USA-states', # set of locations match entries in `locations`
    colorscale = 'Reds',
    colorbar_title = "Millions USD",
))

fig.update_layout(
    title_text = '2011 US Agriculture Exports by State',
    geo_scope='usa', # limite map scope to USA
)

fig.show()


# In[12]:


from plotly import __version__


# In[13]:


print(__version__)


# In[14]:


df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2014_world_gdp_with_codes.csv')


# In[15]:


df.head()


# In[16]:


df


# In[17]:


df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2014_world_gdp_with_codes.csv')

fig = go.Figure(data=go.Choropleth(
    locations = df['CODE'],
    z = df['GDP (BILLIONS)'],
    text = df['COUNTRY'],
    colorscale = 'Reds',
    autocolorscale=False,
    reversescale=True,
    marker_line_color='darkgray',
    marker_line_width=0.5,
    colorbar_tickprefix = '$',
    colorbar_title = 'GDP<br>Billions US$',
))

fig.update_layout(
    title_text='2014 Global GDP',
    geo=dict(
        showframe=False,
        showcoastlines=False,
        projection_type='conic equidistant'
    ),
    annotations = [dict(
        x=0.55,
        y=0.1,
        xref='paper',
        yref='paper',
        text='Source: <a href="https://www.cia.gov/library/publications/the-world-factbook/fields/2195.html">\
            CIA World Factbook</a>',
        showarrow = False
    )]
)

fig.show()


# In[43]:


df.iplot(kind='box', size=40)


# In[ ]:





# In[ ]:




