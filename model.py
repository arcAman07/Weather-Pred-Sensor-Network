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
df = pd.read_csv("/Users/deepaksharma/Downloads/weatherHistory.csv")
# replace the missing values with rain variable
df['Precip Type'].fillna("rain", inplace = True)
df.drop(["Daily Summary"], axis=1, inplace=True)
#Changing Formatted Date from String to Datetime
df['Formatted Date'] = pd.to_datetime(df['Formatted Date'],utc=True)
df['Formatted Date'][0]
df = df.sort_values(by='Formatted Date')
# this DataFrame will use Formatted Date as an index just for visualization objective
data = df.copy()
#data['Formatted Date'] = data['Formatted Date'].strftime("%d/%m/%Y")
#print('Date String:', data['Formatted Date'][0])
# Set 'Formatted Date' as index and resample Date
resampled_df = data.set_index('Formatted Date')
resampled_days = resampled_df.resample('D').mean()
resampled_df = resampled_df.resample('M').mean()
df = pd.get_dummies(data, columns = ['Precip Type'])
from sklearn import preprocessing
  
# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()
  
# Encode labels in column 'species'.
df['Summary']= label_encoder.fit_transform(df['Summary']).astype('float64')
df = df.drop(["Formatted Date"], axis = 1)
df = df.drop(["Wind Speed (km/h)", "Wind Bearing (degrees)","Visibility (km)","Pressure (millibars)", "Loud Cover"], axis = 1)
X = df.drop(['Summary'],axis=1)
Y = df['Summary']
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=42)
from xgboost import XGBClassifier
xgb_model = XGBClassifier(random_state = 0 ,use_label_encoder=False)
xgb_model.fit(X_train, Y_train)

print("Feature Importances : ", xgb_model.feature_importances_)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
from sklearn.linear_model import LogisticRegression
lm = LogisticRegression()
lm.fit(X_train,Y_train)
predictions = lm.predict(X_test)
from sklearn import metrics
y_pred = lm.predict(X_test)
print("Accuracy:",metrics.accuracy_score(Y_test, y_pred))
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, y_pred)))
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_pred,Y_test))
print(confusion_matrix(y_pred,Y_test))
from sklearn.ensemble import RandomForestClassifier 
clf = RandomForestClassifier(n_estimators = 100)   

clf.fit(X_train, Y_train) 
 
y_pred = clf.predict(X_test) 

from sklearn import metrics 
accuracy_s =  metrics.accuracy_score(Y_test, y_pred)
print("Accuracy:",metrics.accuracy_score(Y_test, y_pred))
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, y_pred)))
print(classification_report(y_pred,Y_test))
print(confusion_matrix(y_pred,Y_test))
from sklearn.svm import SVC
model = SVC(kernel='linear',random_state=42)
model.fit(X_train,Y_train)
confirm = model.predict(X_test)
model.score(X_test,Y_test)
from sklearn.model_selection import GridSearchCV
  
# defining parameter range
param_grid = {'C': [0.1, 1, 10, 100, 1000], 
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf']} 
  
grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)
  
# fitting the model for grid search
grid.fit(X_train, Y_train)
print("Accuracy:",metrics.accuracy_score(Y_test, y_pred))
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, y_pred)))
