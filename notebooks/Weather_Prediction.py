#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import seaborn              as sns
import plotly.express       as px
import plotly.graph_objects as go
import plotly.figure_factory as ff


# In[3]:


import holoviews as hv
from holoviews import opts
hv.extension('bokeh')

import warnings
warnings.filterwarnings("ignore")


# In[127]:


df = pd.read_csv("/Users/deepaksharma/Downloads/weatherHistory.csv")


# In[128]:


df.head()


# In[129]:


df.shape


# In[130]:


df['Precip Type'].value_counts()


# In[131]:


# replace the missing values with rain variable
df['Precip Type'].fillna("rain", inplace = True)


# In[132]:


df.isnull().sum()


# In[133]:


df.drop(["Daily Summary"], axis=1, inplace=True)


# In[134]:


df.columns


# In[135]:


#Changing Formatted Date from String to Datetime
df['Formatted Date'] = pd.to_datetime(df['Formatted Date'],utc=True)
df['Formatted Date'][0]


# In[136]:


df = df.sort_values(by='Formatted Date')


# In[137]:


# this DataFrame will use Formatted Date as an index just for visualization objective
data = df.copy()
#data['Formatted Date'] = data['Formatted Date'].strftime("%d/%m/%Y")
#print('Date String:', data['Formatted Date'][0])
# Set 'Formatted Date' as index and resample Date
resampled_df = data.set_index('Formatted Date')
resampled_days = resampled_df.resample('D').mean()
resampled_df = resampled_df.resample('M').mean()


# In[138]:


len(resampled_df),len(df),len(resampled_days)


# In[139]:


print("The new shape of dataset is:",df.shape)


# In[17]:


df.head()


# In[147]:


a = df["Summary"].unique()


# In[154]:


a


# In[155]:


len(a)


# In[18]:


df.nunique()


# In[40]:


df.dtypes


# In[47]:


df = pd.get_dummies(data, columns = ['Precip Type'])


# In[49]:


from sklearn import preprocessing
  
# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()
  
# Encode labels in column 'species'.
df['Summary']= label_encoder.fit_transform(df['Summary']).astype('float64')
  
df['Summary'].unique()


# In[50]:


df.head()


# In[51]:


df = df.drop(["Formatted Date"], axis = 1)


# In[52]:


df.dtypes


# In[53]:


df.columns


# In[54]:


df = df.drop(["Wind Speed (km/h)", "Wind Bearing (degrees)","Visibility (km)","Pressure (millibars)", "Loud Cover"], axis = 1)


# In[55]:


df.corr()


# In[56]:


df.columns


# In[57]:


df.shape


# In[58]:


X = df.drop(['Summary'],axis=1)
Y = df['Summary']


# In[59]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=42)


# In[60]:


get_ipython().system('pip install xgboost')


# In[61]:


from xgboost import XGBClassifier


# In[62]:


xgb_model = XGBClassifier(random_state = 0 ,use_label_encoder=False)
xgb_model.fit(X_train, Y_train)

print("Feature Importances : ", xgb_model.feature_importances_)


# In[63]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[64]:


from sklearn.linear_model import LogisticRegression
lm = LogisticRegression()


# In[65]:


lm.fit(X_train,Y_train)


# In[66]:


predictions = lm.predict(X_test)
from sklearn import metrics
y_pred = lm.predict(X_test)
print("Accuracy:",metrics.accuracy_score(Y_test, y_pred))
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, y_pred)))


# In[67]:


print(y_pred)


# In[68]:


np.unique(y_pred)


# In[69]:


from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_pred,Y_test))


# In[70]:


print(confusion_matrix(y_pred,Y_test))


# In[71]:


from sklearn.ensemble import RandomForestClassifier 
clf = RandomForestClassifier(n_estimators = 100)   

clf.fit(X_train, Y_train) 
 
y_pred = clf.predict(X_test) 

from sklearn import metrics 
accuracy_s =  metrics.accuracy_score(Y_test, y_pred)


# In[72]:


print("Accuracy:",metrics.accuracy_score(Y_test, y_pred))
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, y_pred)))


# In[73]:


print(classification_report(y_pred,Y_test))
print(confusion_matrix(y_pred,Y_test))


# In[74]:


print(y_pred)


# In[75]:


np.unique(y_pred)


# In[76]:


from sklearn.svm import SVC
model = SVC(kernel='linear',random_state=42)
model.fit(X_train,Y_train)


# In[77]:


confirm = model.predict(X_test)
model.score(X_test,Y_test)


# In[80]:


import pickle


# In[100]:


filename = 'finalized_model.sav'
pickle.dump(clf, open(filename, 'wb'))


# In[101]:


test_file = pd.read_csv("/Users/deepaksharma/Downloads/CoolTerm Capture 2022-10-18 13-45-56.txt")


# In[102]:


test_file.head()


# In[103]:


test_file.shape


# In[104]:


test_file = test_file.drop(["Temperature (F)","Apparent Temperature (F)"], axis=1)


# In[105]:


test_file.dtypes


# In[107]:


test_file.columns


# In[108]:


test_df = pd.get_dummies(test_file, columns = ['Precip Type'])


# In[110]:


test_df["Precip Type_1"] = 0 # Precip Type_0 correponds to Precip Type_rain, vice versa


# In[112]:


test_df # Precipe Type_1 corresponds to Precip Type_snow which means other conditions/ weather


# In[114]:


scaler.fit(test_df)
test_df = scaler.transform(test_df)


# In[115]:


y_pred = clf.predict(test_df)


# In[116]:


print(y_pred)


# In[161]:


# PRINTING THE PREDICTIONS ON THE REAL WORLD TEST DATA


# In[159]:


a[19]


# In[160]:


a[26]


# In[162]:


# LABEL ENCODING REFERENCE


# In[163]:


"""
'Breezy', 
'Breezy and Dry', 
'Breezy and Foggy',
'Breezy and Mostly Cloudy', 
'Breezy and Overcast',
'Breezy and Partly Cloudy', 
'Clear',
'Dangerously Windy and Partly Cloudy', 
'Drizzle', 
'Dry',
'Dry and Mostly Cloudy', 
'Dry and Partly Cloudy', 
'Foggy',
'Humid and Mostly Cloudy', 
'Humid and Overcast',
'Humid and Partly Cloudy', 
'Light Rain', 
'Mostly Cloudy',
'Overcast', 
'Partly Cloudy', 
'Rain', 
'Windy', 
'Windy and Dry',
'Windy and Foggy', 
'Windy and Mostly Cloudy', 
'Windy and Overcast',
'Windy and Partly Cloudy'

"""


# In[ ]:




