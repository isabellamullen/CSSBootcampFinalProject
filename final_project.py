#!/usr/bin/env python
# coding: utf-8

# ### FINAL PROJECT
# 
#  How does environmental vulnerability influence climate change sentiment?
# 

# ## Import Packages

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
from plotly import express as px
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.linear_model import Lasso
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV


# ## Data Upload

# In[2]:


#Environmental Justice Index
# data dictionary: https://eji.cdc.gov/Documents/Data/2022/EJI_2022_Data_Dictionary_508.pdf
eji = pd.read_csv('California_eji_2022.csv')

#Yale Climate Opinion Maps
# https://climatecommunication.yale.edu/visualizations-data/ycom-us/
co = pd.read_csv('YCOM7_publicdata_2023.csv')


# ## Cleaning Climate Opinion dataset

# variables to keep from Climate Opinion:
# - geoid: merge
# - geoname: county name
# - citizens: Estimated percentage who think citizens themselves should be doing more/much more to address global warming
# - citizensOppose:	Estimated percentage who think citizens themselves should be doing less/much less to address global warming
# - congress:	Estimated percentage who think Congress should be doing more/much more to address global warming
# - congressOppose:	Estimated percentage who think Congress should be doing less/much less to address global warming!
# - corporations:	Estimated percentage who think corporations and industry should be doing more/much more to address global warming
# - corporationsOppose:	Estimated percentage who think corporations and industry should be doing less/much less to address global warming!
# - discuss:	Estimated percentage who discuss global warming occassionally or often with friends and family
# - discussOppose:	Estimated percentage who discuss global warming rarely or never with friends and family!
# - exp:	Estimated percentage who somewhat/strongly agree that they have personally experienced the effects of global warming
# - expOppose:	Estimated percentage who strongly/somewhat disagree that they have personally experienced the effects of global warming!
# - futuregen:	Estimated percentage who think global warming will harm future generations a moderate amount/a great deal
# - futuregenOppose:	Estimated percentage who think global warming will harm future generations not at all/only a little!
# - gwvoteimp:	Estimated percentage who say a candidate's views on GW are important to their vote
# - gwvoteimpOppose:	Estimated percentage who do not say a candidate's views on GW are important to their vote
# - happening:	Estimated percentage who think that global warming is happening
# - happeningOppose:	Estimated percentage who do not think that global warming is happening!
# - harmus:	Estimated percentage who think global warming will harm people in the US a moderate amount/a great deal
# - harmusOppose:	Estimated percentage who think global warming will harm people in the US not at all/only a little
# - human:	Estimated percentage who think that global warming is caused mostly by human activities
# - humanOppose:	Estimated percentage who think that global warming is caused mostly by natural changes in the environment
# - mediaweekly:	Estimated percentage who hear about global warming in the media at least weekly
# - mediaweeklyOppose:	Estimated percentage who hear about global warming in the media several times a year or less often
# - personal:	Estimated percentage who think global warming will harm them personally a moderate amount/a great deal
# - personalOppose:	Estimated percentage who think global warming will harm them personally not at all/only a little!
# - priority:	Estimated percentage who say global warming should be a high priority for the next president and Congress
# - priorityOppose:	Estimated percentage who do not say global warming should be a high priority for the next president and Congress!
# - regulate:	Estimated percentage who somewhat/strongly support regulating CO2 as a pollutant
# - regulateOppose:	Estimated percentage who somewhat/strongly oppose regulating CO2 as a pollutant!
# - worried:	Estimated percentage who are somewhat/very worried about global warming
# - worriedOppose:	Estimated percentage who are not very/not at all worried about global warming!
# 
# 
# 
# Combined variables:
# - Citizens_combined: Estimated percentage who think citizens themselves should be doing more or less to address global warming.
# - Congress_combined: Estimated percentage who think Congress should be doing more or less to address global warming
# - Corporations_combined: Estimated percentage who think corporations and industry should be doing more or less to address global warming
# 
# 
# - Exp_combined: Estimated percentage who responded that they have personally experienced the effects of global warming
# - Futuregen_combined: Estimated percentage who think global warming will harm future generations
# - harmus_combined: Estimated percentage who think global warming will harm people in the US
# - Personal_combined: Estimated percentage who think global warming will harm them personally a moderate amount/a great deal
# 
# 
# - Gwvoteimp_combined: Estimated percentage who say a candidate's views on GW are important to their vote
# - Priority_combined: Estimated percentage who say global warming should be a high priority for the next president and Congress
# - Regulate_combined: Estimated percentage who somewhat/strongly support regulating CO2 as a pollutant
# 
# 
# - Discuss_combined: Estimated percentage who discuss global warming with friends and family
# - Mediaweekly_combined: Estimated percentage who hear about global warming in the media frequently
# - Human_combined: Estimated percentage who think that global warming is caused mostly by human activities
# 
# 
# - Happening_combined: Estimated percentage who think that global warming is happening
# - Worried_combined: Estimated percentage who are not very/not at all worried about global warming
# 
# 

# In[3]:


# cleaning the Climate Opinion dataset. Only keeping colums for California counties.

myvar = "California"
co_clean = co.query('geoname.str.contains(@myvar)')
myvar2 = 'county'
co_clean = co_clean.query('geotype.str.contains(@myvar2)')


# In[4]:


#climate opinion variable cut
co_clean = co_clean[['geoid', 
                     'geoname', 
                     'citizens',
                     'citizensOppose', 
                     'congress',
                     'congressOppose',
                     'corporations',
                     'corporationsOppose',
                     'discuss',
                     'discussOppose',
                     'exp',
                     'expOppose',
                     'futuregen',
                     'futuregenOppose',
                     'gwvoteimp',
                     'gwvoteimpOppose',
                     'happening',
                     'happeningOppose',
                     'harmus',
                     'harmusOppose',
                     'human',
                     'humanOppose',
                     'mediaweekly',
                     'mediaweeklyOppose',
                     'personal',
                     'personalOppose',
                     'priority',
                     'priorityOppose',
                     'regulate',
                     'regulateOppose',
                     'worried',
                     'worriedOppose'                
]]


# In[5]:


#Rename the merge column to GEOID to match eji_clean
co_clean = co_clean.rename(columns={'geoid': 'GEOID'})


# In[6]:


#combining the oppose and support variables from the climate opinion dataset. Create pairs, weigh, and combine.

co_pairs = [
    ('citizens', 'citizensOppose'),
    ('congress', 'congressOppose'),
    ('corporations', 'corporationsOppose'),
    ('discuss', 'discussOppose'),
    ('exp', 'expOppose'),
    ('futuregen', 'futuregenOppose'),
    ('gwvoteimp', 'gwvoteimpOppose'),
    ('happening', 'happeningOppose'),
    ('harmus', 'harmusOppose'),
    ('human', 'humanOppose'),
    ('mediaweekly', 'mediaweeklyOppose'),
    ('personal', 'personalOppose'),
    ('priority', 'priorityOppose'),
    ('regulate', 'regulateOppose'),
    ('worried', 'worriedOppose')
]

for support, oppose in co_pairs:
    combined_var_name = f'{support}_combined'
    co_clean[combined_var_name] = (co_clean[support] * 1) + (co_clean[oppose] * -1)


# In[7]:


#keep only the combined variables. Include GEOID and geoname.
co_clean = co_clean[['GEOID', 'geoname','citizens_combined', 'congress_combined',
       'corporations_combined', 'discuss_combined', 'exp_combined',
       'futuregen_combined', 'gwvoteimp_combined', 'happening_combined',
       'harmus_combined', 'human_combined', 'mediaweekly_combined',
       'personal_combined', 'priority_combined', 'regulate_combined',
       'worried_combined']]


# ## Cleaning the Environmental Justice Index

# Variables to keep from EJI:
# * geo codes
# - GEOID: County identifier; a concatenation of current state Federal Information Processing Series (FIPS) code and county FIPS code
# - COUNTY: County names
# - E_TOTPOP: Population estimate, 2014-2018 ACS

# In[8]:


# keep only the first 4 digets of GEOID so it matches with the climate opinion data

eji['GEOID'] = eji['GEOID'].astype(str).str[:4].astype(int)


# In[9]:


#separate the estimate columns with start with E_ and EP_. These columns contain the estimate variables.
eji_est = [col for col in eji.columns if col.startswith('E_') or col.startswith('EP_')]


# In[10]:


#keep only estimate columns from EJI. Include GEOID and COUNTY.

eji_clean = eji[['GEOID','COUNTY','E_TOTPOP','E_OZONE','E_PM','E_DSLPM','E_TOTCR','E_NPL','E_TRI','E_TSD','E_RMP','E_COAL','E_LEAD','E_PARK','E_HOUAGE','E_WLKIND','E_RAIL','E_ROAD','E_AIRPRT','E_IMPWTR','EP_MINRTY','EP_POV200','EP_NOHSDP','EP_UNEMP','EP_RENTER','EP_HOUBDN','EP_UNINSUR','EP_NOINT','EP_AGE65','EP_AGE17','EP_DISABL','EP_LIMENG','EP_MOBILE','EP_GROUPQ','EP_BPHIGH','EP_ASTHMA','EP_CANCER', 'EP_MHLTH','EP_DIABETES']]


# In[11]:


# aggregate census tract data into county-level data by calculating weighted averages 
#for various metrics

#Calculate the weighted average for each theme using E_TOTPOP (population variable) as the weight
def weighted_avg(group, avg_name, weight_name):
    """Calculate weighted average."""
    d = group[avg_name]
    w = group[weight_name]
    return (d * w).sum() / w.sum()

# Group by COUNTY and apply weighted average or sum where appropriate
eji_clean_grouped = eji_clean.groupby('COUNTY').apply(
    lambda x: pd.Series({
        'E_TOTPOP': x['E_TOTPOP'].sum(),
        'E_OZONE': weighted_avg(x, 'E_OZONE', 'E_TOTPOP'),
        'E_PM': weighted_avg(x, 'E_PM', 'E_TOTPOP'),
        'E_DSLPM': weighted_avg(x, 'E_DSLPM', 'E_TOTPOP'),
        'E_TOTCR': weighted_avg(x, 'E_TOTCR', 'E_TOTPOP'),
        'E_NPL': weighted_avg(x, 'E_NPL', 'E_TOTPOP'),
        'E_TRI': weighted_avg(x, 'E_TRI', 'E_TOTPOP'),
        'E_TSD': weighted_avg(x, 'E_TSD', 'E_TOTPOP'),
        'E_RMP': weighted_avg(x, 'E_RMP', 'E_TOTPOP'),
        'E_COAL': weighted_avg(x, 'E_COAL', 'E_TOTPOP'),
        'E_LEAD': weighted_avg(x, 'E_LEAD', 'E_TOTPOP'),
        'E_PARK': weighted_avg(x, 'E_PARK', 'E_TOTPOP'),
        'E_HOUAGE': weighted_avg(x, 'E_HOUAGE', 'E_TOTPOP'),
        'E_WLKIND': weighted_avg(x, 'E_WLKIND', 'E_TOTPOP'),
        'E_RAIL': weighted_avg(x, 'E_RAIL', 'E_TOTPOP'),
        'E_ROAD': weighted_avg(x, 'E_ROAD', 'E_TOTPOP'),
        'E_AIRPRT': weighted_avg(x, 'E_AIRPRT', 'E_TOTPOP'),
        'E_IMPWTR': weighted_avg(x, 'E_IMPWTR', 'E_TOTPOP'),
        'EP_MINRTY': weighted_avg(x, 'EP_MINRTY', 'E_TOTPOP'),
        'EP_POV200': weighted_avg(x, 'EP_POV200', 'E_TOTPOP'),
        'EP_NOHSDP': weighted_avg(x, 'EP_NOHSDP', 'E_TOTPOP'),
        'EP_UNEMP': weighted_avg(x, 'EP_UNEMP', 'E_TOTPOP'),
        'EP_RENTER': weighted_avg(x, 'EP_RENTER', 'E_TOTPOP'),
        'EP_HOUBDN': weighted_avg(x, 'EP_HOUBDN', 'E_TOTPOP'),
        'EP_UNINSUR': weighted_avg(x, 'EP_UNINSUR', 'E_TOTPOP'),
        'EP_NOINT': weighted_avg(x, 'EP_NOINT', 'E_TOTPOP'),
        'EP_AGE65': weighted_avg(x, 'EP_AGE65', 'E_TOTPOP'),
        'EP_AGE17': weighted_avg(x, 'EP_AGE17', 'E_TOTPOP'),
        'EP_DISABL': weighted_avg(x, 'EP_DISABL', 'E_TOTPOP'),
        'EP_LIMENG': weighted_avg(x, 'EP_LIMENG', 'E_TOTPOP'),
        'EP_MOBILE': weighted_avg(x, 'EP_MOBILE', 'E_TOTPOP'),
        'EP_GROUPQ': weighted_avg(x, 'EP_GROUPQ', 'E_TOTPOP'),
        'EP_BPHIGH': weighted_avg(x, 'EP_BPHIGH', 'E_TOTPOP'),
        'EP_ASTHMA': weighted_avg(x, 'EP_ASTHMA', 'E_TOTPOP'),
        'EP_CANCER': weighted_avg(x, 'EP_CANCER', 'E_TOTPOP'),
        'EP_MHLTH': weighted_avg(x, 'EP_MHLTH', 'E_TOTPOP'),
        'EP_DIABETES': weighted_avg(x, 'EP_DIABETES', 'E_TOTPOP')
    })
).reset_index()

# Add the GEOID column, ensuring it matches the county
eji_clean_grouped['GEOID'] = [eji_clean[eji_clean['COUNTY'] == county]['GEOID'].iloc[0] for county in eji_clean_grouped['COUNTY']]

#save back into eji_clean
eji_clean = eji_clean_grouped


# ## Merge Dataset

# In[12]:


#merge eji_clean and co_clean

eji_co_merge = pd.merge(co_clean, eji_clean, how = 'inner', on = 'GEOID')


# In[13]:


# remove only the float columns
num = eji_co_merge.select_dtypes(include='float')


# In[14]:


#scaling the data

scaler = StandardScaler()
num_scaled = scaler.fit_transform(num)


# In[15]:


#add scaled data back into eji_co_merge
eji_co_merge = pd.DataFrame(num_scaled, columns = num.columns)


# In[16]:


eji_co_merge.shape


# ## Analysis

# In[17]:


# Selecting relevant features (independent variables) from the dataset 'eji_co_merge' These features include environmental and demographic indicators.
X = eji_co_merge[['E_OZONE', 'E_PM', 'E_DSLPM','E_TOTCR', 'E_NPL', 'E_TRI', 'E_TSD', 'E_RMP', 'E_COAL', 'E_LEAD','E_PARK', 'E_HOUAGE', 'E_WLKIND', 'E_RAIL', 'E_ROAD', 'E_AIRPRT','E_IMPWTR', 'EP_MINRTY', 'EP_POV200', 'EP_NOHSDP', 'EP_UNEMP','EP_RENTER', 'EP_HOUBDN', 'EP_UNINSUR', 'EP_NOINT', 'EP_AGE65','EP_AGE17', 'EP_DISABL', 'EP_LIMENG', 'EP_MOBILE', 'EP_GROUPQ','EP_BPHIGH', 'EP_ASTHMA', 'EP_CANCER', 'EP_MHLTH', 'EP_DIABETES']]
y = eji_co_merge[['citizens_combined']]

# Splitting the data into training and testing sets with a 80-20 split
# 'random_state=42' ensures reproducibility of the results
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[18]:


# Initializing a Lasso regression model and setting up a parameter grid for alpha values to test in the Lasso model
# Alpha controls the strength of the regularization (higher alpha -> more regularization)

lasso = Lasso()
parameters = {'alpha': np.arange(0.01, 1, 0.05)}


# In[19]:


# Performing grid search cross-validation to find the best alpha value for the Lasso model
reg = GridSearchCV(lasso, parameters)
reg.fit(X_train, y_train)

# Extracting the best Lasso model with the optimal alpha value
## Using the best parameters
model = reg.best_estimator_


# In[20]:


y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)

print(f'Training MSE: {mse_train}')
print(f'Testing MSE: {mse_test}')
print(f'Training R^2: {r2_train}')
print(f'Testing R^2: {r2_test}')


# In[21]:


column_names = ['Days above O3 standard', 
                'Days above PM2.5 standard', 
                'Ambient concentrations of diesel', 
                'Probability of contracting cancer',
                'Proximity to EPA National Priority List site',
                'Proximity to EPA Toxic Release Inventory site',
                'Proximity to EPA Treatment, Storage,and Disposal site',
                'Proximity to EPA risk management plan site',
                'Proximity to coal mines',
                'Proximity to lead mines',
                'Proximity to green space',
                'Houses built pre-1980',
                'Walkability',
                'Proximity to railroad',
                'Proximity to high-volume road or highway',
                'Proximity to airport',
                'Impacted watershed',
                'Percentage of minority persons',
                'Percentage below 200% poverty',
                'Persons with no high school diploma (age 25+)',
                'Persons unemployed',
                'Persons who rent',
                'Households that make less than 75,000',
                'Persons who are uninsured',
                'Persons without internet',
                'Aged 65 and older',
                'Aged 17 and younger',
                'Population with a disability',
                'Persons (age 5+) who speak English "less than well"',
                'Ammount of mobile homes',
                'People living in group quarters',
                'Persons with high blood pressure',
                'Persons with asthma',
                'Persons with cancer',
                'Persons reporting not good mental health',
                'Persons with diabetes']


# In[22]:


# Creating a DataFrame to display the Lasso coefficients corresponding to each feature
lasso_coefficients = pd.DataFrame({
    'Feature': column_names,
    'Coefficient': model.coef_
})

print(lasso_coefficients)


# In[23]:


# Filter out the features with non-zero coefficients from the Lasso model
non_zero_features = lasso_coefficients[lasso_coefficients.Coefficient != 0]

#create the plot with size, and a line at 0
plt.figure(figsize=(8,8))
plt.barh(y = lasso_coefficients.Feature, width = lasso_coefficients.Coefficient, color = 'green', zorder=3)
plt.axvline(x = 0, color='black', label='axvline - full height', linewidth = 1)

#add horizontal lines for each bar
for feature in non_zero_features.Feature:
    plt.axhline(y=feature, color='gray', linewidth=0.5)

# Set the limits of the x-axis to ensure the plot is scaled automatically based on the coefficient values. The range is extended by 20% on both sides for better visualization
plt.xlim(-1 * max(np.abs(lasso_coefficients.Coefficient)) * 1.2, max(np.abs(lasso_coefficients.Coefficient)) * 1.2)
plt.title(y)


# ## function

# In[24]:


# Defining a function 'split' that takes a target variable 'y' as input

def split(y):
    
    y = eji_co_merge[[y]]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    reg.fit(X_train, y_train)
    
    model = reg.best_estimator_
    
    lasso_coefficients = pd.DataFrame({
    'Feature': column_names,
    'Coefficient': model.coef_})
    
    return lasso_coefficients
    


# # 1. Who should act?
# - **Citizens_combined:** Estimated percentage who think citizens themselves should be doing more or less to address global warming.
# - **Congress_combined:** Estimated percentage who think Congress should be doing more or less to address global warming
# - **Corporations_combined:** Estimated percentage who think corporations and industry should be doing more or less to address global warming
# 
# 
# 
# | Citizens            | Feature | Coefficient |
# | :---------------- | :------: | ----: |
# |         |   Proximity to green space   | 0.188540 |
# |            |   Persons without internet   | -0.258834 |
# 
# | Congress            | Feature | Coefficient |
# | :---------------- | :------: | ----: |
# |         |   Proximity to green space   | 0.285941 |
# |            |   Persons without internet   |  -0.130169 |
# 
# | Corporations            | Feature | Coefficient |
# | :---------------- | :------: | ----: |
# |         |   Proximity to green space   | 0.187994 |
# |            |   Persons without internet   |  -0.313111 |

# ### citizens_combined

# In[25]:


# Defining a function 'split' that takes a target variable 'y' as input
# The function performs the train-test split, fits the Lasso model, and returns the Lasso coefficients

lasso_coefficients = split('citizens_combined')

y = 'citizens_combined'
split(y)


# In[26]:


y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)

print(f'Training MSE: {mse_train}')
print(f'Testing MSE: {mse_test}')
print(f'Training R^2: {r2_train}')
print(f'Testing R^2: {r2_test}')


# In[ ]:


# Filter out the features with non-zero coefficients from the Lasso model
non_zero_features = lasso_coefficients[lasso_coefficients.Coefficient != 0]

#create the plot with size, and a line at 0
plt.figure(figsize=(8,8))
plt.barh(y = lasso_coefficients.Feature, width = lasso_coefficients.Coefficient, color = 'green', zorder=3)
plt.axvline(x = 0, color='black', label='axvline - full height', linewidth = 1)

#add horizontal lines for each bar
for feature in non_zero_features.Feature:
    plt.axhline(y=feature, color='gray', linewidth=0.5)

# Set the limits of the x-axis to ensure the plot is scaled automatically based on the coefficient values. The range is extended by 20% on both sides for better visualization
plt.xlim(-1 * max(np.abs(lasso_coefficients.Coefficient)) * 1.2, max(np.abs(lasso_coefficients.Coefficient)) * 1.2)
plt.title(y)


# ### congress_combined

# In[ ]:


# Defining a function 'split' that takes a target variable 'y' as input
# The function performs the train-test split, fits the Lasso model, and returns the Lasso coefficients

lasso_coefficients = split('congress_combined')

y = 'congress_combined'
split(y)


# In[ ]:


# Filter out the features with non-zero coefficients from the Lasso model
non_zero_features = lasso_coefficients[lasso_coefficients.Coefficient != 0]

#create the plot with size, and a line at 0
plt.figure(figsize=(8,8))
plt.barh(y = lasso_coefficients.Feature, width = lasso_coefficients.Coefficient, color = 'green', zorder=3)
plt.axvline(x = 0, color='black', label='axvline - full height', linewidth = 1)

#add horizontal lines for each bar
for feature in non_zero_features.Feature:
    plt.axhline(y=feature, color='gray', linewidth=0.5)

# Set the limits of the x-axis to ensure the plot is scaled automatically based on the coefficient values. The range is extended by 20% on both sides for better visualization
plt.xlim(-1 * max(np.abs(lasso_coefficients.Coefficient)) * 1.2, max(np.abs(lasso_coefficients.Coefficient)) * 1.2)
plt.title(y)


# ### corporations_combined

# In[ ]:


lasso_coefficients = split('corporations_combined')

y = 'corporations_combined'
split(y)


# In[ ]:


# Filter out the features with non-zero coefficients from the Lasso model
non_zero_features = lasso_coefficients[lasso_coefficients.Coefficient != 0]

#create the plot with size, and a line at 0
plt.figure(figsize=(8,8))
plt.barh(y = lasso_coefficients.Feature, width = lasso_coefficients.Coefficient, color = 'green', zorder=3)
plt.axvline(x = 0, color='black', label='axvline - full height', linewidth = 1)

#add horizontal lines for each bar
for feature in non_zero_features.Feature:
    plt.axhline(y=feature, color='gray', linewidth=0.5)

# Set the limits of the x-axis to ensure the plot is scaled automatically based on the coefficient values. The range is extended by 20% on both sides for better visualization
plt.xlim(-1 * max(np.abs(lasso_coefficients.Coefficient)) * 1.2, max(np.abs(lasso_coefficients.Coefficient)) * 1.2)
plt.title(y)


# # 2. Who is harmed ?
# - **Futuregen_combined:** Estimated percentage who think global warming will harm future generations
# - **harmus_combined:** Estimated percentage who think global warming will harm people in the US
# - **Personal_combined:** Estimated percentage who think global warming will harm them personally a moderate amount/a great deal
# 
# 
# 
# | Future Gen            | Feature | Coefficient |
# | :---------------- | :------: | ----: |
# |         |   Proximity to green space   | 0.265506 |
# |            |   Persons without internet   | -0.153935 |
# 
# | Harm US           | Feature | Coefficient |
# | :---------------- | :------: | ----: |
# |         |   Proximity to green space   | 0.218492 |
# |            |   Persons with high blood pressure   |  -0.133470 |
# 
# | Personal            | Feature | Coefficient |
# | :---------------- | :------: | ----: |
# |         |   Percentage of minority persons   | 0.388241 |
# |            |   Persons with high blood pressure   |  -0.139409 |

# ### futuregen_combined

# In[ ]:


lasso_coefficients = split('futuregen_combined')

y = 'futuregen_combined'
split(y)


# In[ ]:


# Filter out the features with non-zero coefficients from the Lasso model
non_zero_features = lasso_coefficients[lasso_coefficients.Coefficient != 0]

#create the plot with size, and a line at 0
plt.figure(figsize=(8,8))
plt.barh(y = lasso_coefficients.Feature, width = lasso_coefficients.Coefficient, color = 'green', zorder=3)
plt.axvline(x = 0, color='black', label='axvline - full height', linewidth = 1)

#add horizontal lines for each bar
for feature in non_zero_features.Feature:
    plt.axhline(y=feature, color='gray', linewidth=0.5)

# Set the limits of the x-axis to ensure the plot is scaled automatically based on the coefficient values. The range is extended by 20% on both sides for better visualization
plt.xlim(-1 * max(np.abs(lasso_coefficients.Coefficient)) * 1.2, max(np.abs(lasso_coefficients.Coefficient)) * 1.2)
plt.title(y)


# ### harmus_combined

# In[ ]:


lasso_coefficients = split('harmus_combined')

y = 'harmus_combined'
split(y)


# In[ ]:


# Filter out the features with non-zero coefficients from the Lasso model
non_zero_features = lasso_coefficients[lasso_coefficients.Coefficient != 0]

#create the plot with size, and a line at 0
plt.figure(figsize=(8,8))
plt.barh(y = lasso_coefficients.Feature, width = lasso_coefficients.Coefficient, color = 'green', zorder=3)
plt.axvline(x = 0, color='black', label='axvline - full height', linewidth = 1)

#add horizontal lines for each bar
for feature in non_zero_features.Feature:
    plt.axhline(y=feature, color='gray', linewidth=0.5)

# Set the limits of the x-axis to ensure the plot is scaled automatically based on the coefficient values. The range is extended by 20% on both sides for better visualization
plt.xlim(-1 * max(np.abs(lasso_coefficients.Coefficient)) * 1.2, max(np.abs(lasso_coefficients.Coefficient)) * 1.2)
plt.title(y)


# ### personal_combined

# In[ ]:


lasso_coefficients = split('personal_combined')

y = 'personal_combined'
split(y)


# In[ ]:


# Filter out the features with non-zero coefficients from the Lasso model
non_zero_features = lasso_coefficients[lasso_coefficients.Coefficient != 0]

#create the plot with size, and a line at 0
plt.figure(figsize=(8,8))
plt.barh(y = lasso_coefficients.Feature, width = lasso_coefficients.Coefficient, color = 'green', zorder=3)
plt.axvline(x = 0, color='black', label='axvline - full height', linewidth = 1)

#add horizontal lines for each bar
for feature in non_zero_features.Feature:
    plt.axhline(y=feature, color='gray', linewidth=0.5)

# Set the limits of the x-axis to ensure the plot is scaled automatically based on the coefficient values. The range is extended by 20% on both sides for better visualization
plt.xlim(-1 * max(np.abs(lasso_coefficients.Coefficient)) * 1.2, max(np.abs(lasso_coefficients.Coefficient)) * 1.2)
plt.title(y)


# # 3. Voting/ legislation:
# - **Gwvoteimp_combined:** Estimated percentage who say a candidate's views on GW are important to their vote
# - **Priority_combined:** Estimated percentage who say global warming should be a high priority for the next president and Congress
# - **Regulate_combined:** Estimated percentage who somewhat/strongly support regulating CO2 as a pollutant
# 
# 
# 
# | Gwvoteimp            | Feature | Coefficient |
# | :---------------- | :------: | ----: |
# |         |   Proximity to green space   | 0.328951 |
# |            |   Persons without internet   | -0.096976 |
# 
# | Priority           | Feature | Coefficient |
# | :---------------- | :------: | ----: |
# |         |   Proximity to green space   | 0.276327 |
# |            |   Persons with high blood pressure   |  -0.192180 |
# 
# | Regulate            | Feature | Coefficient |
# | :---------------- | :------: | ----: |
# |         |   Proximity to green space   | 0.173203 |
# |            |   Persons without internet   | -0.388985 |
# 

# ### gwvoteimp_combined

# In[ ]:


lasso_coefficients = split('gwvoteimp_combined')

y = 'gwvoteimp_combined'
split(y)


# In[ ]:


# Filter out the features with non-zero coefficients from the Lasso model
non_zero_features = lasso_coefficients[lasso_coefficients.Coefficient != 0]

#create the plot with size, and a line at 0
plt.figure(figsize=(8,8))
plt.barh(y = lasso_coefficients.Feature, width = lasso_coefficients.Coefficient, color = 'green', zorder=3)
plt.axvline(x = 0, color='black', label='axvline - full height', linewidth = 1)

#add horizontal lines for each bar
for feature in non_zero_features.Feature:
    plt.axhline(y=feature, color='gray', linewidth=0.5)

# Set the limits of the x-axis to ensure the plot is scaled automatically based on the coefficient values. The range is extended by 20% on both sides for better visualization
plt.xlim(-1 * max(np.abs(lasso_coefficients.Coefficient)) * 1.2, max(np.abs(lasso_coefficients.Coefficient)) * 1.2)
plt.title(y)


# ### priority_combined

# In[ ]:


lasso_coefficients = split('priority_combined')

y = 'priority_combined'
split(y)


# In[ ]:


# Filter out the features with non-zero coefficients from the Lasso model
non_zero_features = lasso_coefficients[lasso_coefficients.Coefficient != 0]

#create the plot with size, and a line at 0
plt.figure(figsize=(8,8))
plt.barh(y = lasso_coefficients.Feature, width = lasso_coefficients.Coefficient, color = 'green', zorder=3)
plt.axvline(x = 0, color='black', label='axvline - full height', linewidth = 1)

#add horizontal lines for each bar
for feature in non_zero_features.Feature:
    plt.axhline(y=feature, color='gray', linewidth=0.5)

# Set the limits of the x-axis to ensure the plot is scaled automatically based on the coefficient values. The range is extended by 20% on both sides for better visualization
plt.xlim(-1 * max(np.abs(lasso_coefficients.Coefficient)) * 1.2, max(np.abs(lasso_coefficients.Coefficient)) * 1.2)
plt.title(y)


# ### regulate_combined

# In[ ]:


lasso_coefficients = split('regulate_combined')

y = 'regulate_combined'
split(y)


# In[ ]:


# Filter out the features with non-zero coefficients from the Lasso model
non_zero_features = lasso_coefficients[lasso_coefficients.Coefficient != 0]

#create the plot with size, and a line at 0
plt.figure(figsize=(8,8))
plt.barh(y = lasso_coefficients.Feature, width = lasso_coefficients.Coefficient, color = 'green', zorder=3)
plt.axvline(x = 0, color='black', label='axvline - full height', linewidth = 1)

#add horizontal lines for each bar
for feature in non_zero_features.Feature:
    plt.axhline(y=feature, color='gray', linewidth=0.5)

# Set the limits of the x-axis to ensure the plot is scaled automatically based on the coefficient values. The range is extended by 20% on both sides for better visualization
plt.xlim(-1 * max(np.abs(lasso_coefficients.Coefficient)) * 1.2, max(np.abs(lasso_coefficients.Coefficient)) * 1.2)
plt.title(y)


# # 4. Climate change discussion:
# - **Discuss_combined:** Estimated percentage who discuss global warming with friends and family
# - **Mediaweekly_combined:** Estimated percentage who hear about global warming in the media
# - **Human_combined:** Estimated percentage who think that global warming is caused mostly by human activities
# 
# 
# 
# | Discuss            | Feature | Coefficient |
# | :---------------- | :------: | ----: |
# |         |   Proximity to green space   | 0.094367 |
# |            |   Persons with not good mental health   | -0.297540 |
# 
# | Mediaweekly           | Feature | Coefficient |
# | :---------------- | :------: | ----: |
# |         |   Aged 65 and older   | 0.481906 |
# |            |   Persons with diabetes   |  -0.235376 |
# 
# | Human            | Feature | Coefficient |
# | :---------------- | :------: | ----: |
# |         |   Proximity to green space   | 0.314892 |
# |            |   Persons with asthma   | -0.098549 |
# 

# ### discuss_combined

# In[ ]:


lasso_coefficients = split('discuss_combined')

y = 'discuss_combined'
split(y)


# In[ ]:


# Filter out the features with non-zero coefficients from the Lasso model
non_zero_features = lasso_coefficients[lasso_coefficients.Coefficient != 0]

#create the plot with size, and a line at 0
plt.figure(figsize=(8,8))
plt.barh(y = lasso_coefficients.Feature, width = lasso_coefficients.Coefficient, color = 'green', zorder=3)
plt.axvline(x = 0, color='black', label='axvline - full height', linewidth = 1)

#add horizontal lines for each bar
for feature in non_zero_features.Feature:
    plt.axhline(y=feature, color='gray', linewidth=0.5)

# Set the limits of the x-axis to ensure the plot is scaled automatically based on the coefficient values. The range is extended by 20% on both sides for better visualization
plt.xlim(-1 * max(np.abs(lasso_coefficients.Coefficient)) * 1.2, max(np.abs(lasso_coefficients.Coefficient)) * 1.2)
plt.title(y)


# ### mediaweekly_combined

# In[ ]:


lasso_coefficients = split('mediaweekly_combined')

y = 'mediaweekly_combined'
split(y)


# In[ ]:


# Filter out the features with non-zero coefficients from the Lasso model
non_zero_features = lasso_coefficients[lasso_coefficients.Coefficient != 0]

#create the plot with size, and a line at 0
plt.figure(figsize=(8,8))
plt.barh(y = lasso_coefficients.Feature, width = lasso_coefficients.Coefficient, color = 'green', zorder=3)
plt.axvline(x = 0, color='black', label='axvline - full height', linewidth = 1)

#add horizontal lines for each bar
for feature in non_zero_features.Feature:
    plt.axhline(y=feature, color='gray', linewidth=0.5)

# Set the limits of the x-axis to ensure the plot is scaled automatically based on the coefficient values. The range is extended by 20% on both sides for better visualization
plt.xlim(-1 * max(np.abs(lasso_coefficients.Coefficient)) * 1.2, max(np.abs(lasso_coefficients.Coefficient)) * 1.2)
plt.title(y)


# ### human_combined

# In[ ]:


lasso_coefficients = split('human_combined')

y = 'human_combined'
split(y)


# In[ ]:


# Filter out the features with non-zero coefficients from the Lasso model
non_zero_features = lasso_coefficients[lasso_coefficients.Coefficient != 0]

#create the plot with size, and a line at 0
plt.figure(figsize=(8,8))
plt.barh(y = lasso_coefficients.Feature, width = lasso_coefficients.Coefficient, color = 'green', zorder=3)
plt.axvline(x = 0, color='black', label='axvline - full height', linewidth = 1)

#add horizontal lines for each bar
for feature in non_zero_features.Feature:
    plt.axhline(y=feature, color='gray', linewidth=0.5)

# Set the limits of the x-axis to ensure the plot is scaled automatically based on the coefficient values. The range is extended by 20% on both sides for better visualization
plt.xlim(-1 * max(np.abs(lasso_coefficients.Coefficient)) * 1.2, max(np.abs(lasso_coefficients.Coefficient)) * 1.2)
plt.title(y)


# # 5. Climate change concern:
# - **Exp_combined:** Estimated percentage who responded that they have personally experienced the effects of global warming
# - **Happening_combined:** Estimated percentage who think that global warming is happening
# - **Worried_combined:** Estimated percentage who are not very/not at all worried about global warming
# 
# 
# 
# 
# | Exp            | Feature | Coefficient |
# | :---------------- | :------: | ----: |
# |         |   Proximity to green space   | 0.241683|
# |            |   Persons without internet   | -0.192961 |
# 
# | Happening           | Feature | Coefficient |
# | :---------------- | :------: | ----: |
# |         |   Proximity to green space   | 0.259983 |
# |            |   Persons without internet   |  -0.220575 |
# 
# | Worried            | Feature | Coefficient |
# | :---------------- | :------: | ----: |
# |         |   Proximity to green space   | 0.278782 |
# |            |   Persons with high blood pressure   | -0.129625 |

# ### exp_combined

# In[ ]:


lasso_coefficients = split('exp_combined')

y = 'exp_combined'
split(y)


# In[ ]:


# Filter out the features with non-zero coefficients from the Lasso model
non_zero_features = lasso_coefficients[lasso_coefficients.Coefficient != 0]

#create the plot with size, and a line at 0
plt.figure(figsize=(8,8))
plt.barh(y = lasso_coefficients.Feature, width = lasso_coefficients.Coefficient, color = 'green', zorder=3)
plt.axvline(x = 0, color='black', label='axvline - full height', linewidth = 1)

#add horizontal lines for each bar
for feature in non_zero_features.Feature:
    plt.axhline(y=feature, color='gray', linewidth=0.5)

# Set the limits of the x-axis to ensure the plot is scaled automatically based on the coefficient values. The range is extended by 20% on both sides for better visualization
plt.xlim(-1 * max(np.abs(lasso_coefficients.Coefficient)) * 1.2, max(np.abs(lasso_coefficients.Coefficient)) * 1.2)
plt.title(y)


# ### happening_combined

# In[ ]:


lasso_coefficients = split('happening_combined')

y = 'happening_combined'
split(y)


# In[ ]:


# Filter out the features with non-zero coefficients from the Lasso model
non_zero_features = lasso_coefficients[lasso_coefficients.Coefficient != 0]

#create the plot with size, and a line at 0
plt.figure(figsize=(8,8))
plt.barh(y = lasso_coefficients.Feature, width = lasso_coefficients.Coefficient, color = 'green', zorder=3)
plt.axvline(x = 0, color='black', label='axvline - full height', linewidth = 1)

#add horizontal lines for each bar
for feature in non_zero_features.Feature:
    plt.axhline(y=feature, color='gray', linewidth=0.5)

# Set the limits of the x-axis to ensure the plot is scaled automatically based on the coefficient values. The range is extended by 20% on both sides for better visualization
plt.xlim(-1 * max(np.abs(lasso_coefficients.Coefficient)) * 1.2, max(np.abs(lasso_coefficients.Coefficient)) * 1.2)
plt.title(y)


# ### worried_combined

# In[ ]:


lasso_coefficients = split('worried_combined')

y = 'worried_combined'
split(y)


# In[ ]:


# Filter out the features with non-zero coefficients from the Lasso model
non_zero_features = lasso_coefficients[lasso_coefficients.Coefficient != 0]

#create the plot with size, and a line at 0
plt.figure(figsize=(8,8))
plt.barh(y = lasso_coefficients.Feature, width = lasso_coefficients.Coefficient, color = 'green', zorder=3)
plt.axvline(x = 0, color='black', label='axvline - full height', linewidth = 1)

#add horizontal lines for each bar
for feature in non_zero_features.Feature:
    plt.axhline(y=feature, color='gray', linewidth=0.5)

# Set the limits of the x-axis to ensure the plot is scaled automatically based on the coefficient values. The range is extended by 20% on both sides for better visualization
plt.xlim(-1 * max(np.abs(lasso_coefficients.Coefficient)) * 1.2, max(np.abs(lasso_coefficients.Coefficient)) * 1.2)
plt.title(y)


# # produce all plots at once

# In[ ]:


# this function performs the split function and the plot function to produce all plots. I kept each plot separate above to make commentary easier.

# def split_and_plot(y):
#     """
#     This function splits the data into training and testing sets, fits a Lasso regression model,
#     extracts the non-zero coefficients, and then creates a horizontal bar plot of these coefficients.

#     Parameters:
#     y (str): The name of the dependent variable to be used in the regression.

#     The plot includes:
#     - A bar for each feature with a non-zero coefficient.
#     - A vertical line at x = 0 to help distinguish positive and negative coefficients.
#     - Horizontal lines for each bar to visually separate the features.

#     The x-axis limits are automatically scaled based on the maximum absolute value of the coefficients,
#     extended by 20% on both sides for better visualization.
#     """
        
#     # Filter the dependent variable
#     y_var = eji_co_merge[[y]]
    
#     # Split the data into training and testing sets
#     X_train, X_test, y_train, y_test = train_test_split(X, y_var, test_size=0.2, random_state=42)
    
#     # Fit the Lasso regression model
#     reg.fit(X_train, y_train)
    
#     # Extract the best model after GridSearchCV
#     model = reg.best_estimator_
    
#     # Create a DataFrame with features and their corresponding coefficients
#     lasso_coefficients = pd.DataFrame({
#         'Feature': column_names,
#         'Coefficient': model.coef_
#     })
    
#     # Filter out features with non-zero coefficients
#     non_zero_features = lasso_coefficients[lasso_coefficients.Coefficient != 0]
    
#     # Create the plot
#     plt.figure(figsize=(8,8))
#     plt.barh(y = lasso_coefficients.Feature, width = lasso_coefficients.Coefficient, color = 'green', zorder=3)
#     plt.axvline(x = 0, color='black', label='axvline - full height', linewidth = 1)

#     # Add horizontal lines for each bar
#     for feature in non_zero_features.Feature:
#         plt.axhline(y=feature, color='gray', linewidth=0.5)

#     # Set x-axis limits for better visualization
#     plt.xlim(-1 * max(np.abs(lasso_coefficients.Coefficient)) * 1.2, max(np.abs(lasso_coefficients.Coefficient)) * 1.2)
#     plt.title(y)


# # Loop through each variable that contains '_combined' in its name and plot
# for y in co_clean.filter(like='_combined').columns:
#     split_and_plot(y)


# # For saving all plots as PNG

# In[ ]:


## only for saving the plots! commented out for now

# # Loop through each variable that contains '_combined' in its name and save the plot
# for y in co_clean.filter(like='_combined').columns:
#     print(f"Processing: {y}")  # Debugging statement
    
#     # Generate the plot
#     split_and_plot(y)
    
#     # Save the plot
#     filename = f'{y}_coefficients.png'
#     print(f"Saving plot to: {filename}")  # Debugging statement
#     plt.savefig(filename, bbox_inches='tight')  # Save the plot with a tight bounding box
#     plt.close()  # Close the plot to free up memory
#     print(f"Plot saved and closed for {y}")

