import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn              as sns
import plotly.express       as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import holoviews as hv
from holoviews import opts
hv.extension('bokeh')

import warnings
warnings.filterwarnings("ignore")
import pickle
from sklearn import preprocessing
  
# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()
filename = 'finalized_model.sav'
# load the model from disk
clf = pickle.load(open(filename, 'rb'))
from sklearn.preprocessing import StandardScaler
test_file = pd.read_csv("/Users/deepaksharma/Downloads/CoolTerm Capture 2022-10-18 13-45-56.txt")
scaler = StandardScaler()
scaler.fit(test_file)
test_file = scaler.transform(test_file)
test_file = test_file.drop(["Temperature (F)","Apparent Temperature (F)"], axis=1)
test_df = pd.get_dummies(test_file, columns = ['Precip Type'])
test_df["Precip Type_1"] = 0 # Precip Type_0 correponds to Precip Type_rain, vice versa
test_df # Precipe Type_1 corresponds to Precip Type_snow which means other conditions/ weather
y_pred = clf.predict(test_df)
print(y_pred) # a[19] = 'Partly Cloudy'
# a[26] = ''Windy and Partly Cloudy'
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


