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
