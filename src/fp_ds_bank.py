# Bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (FunctionTransformer, OneHotEncoder, StandardScaler)
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Transform Yeo
from sklearn.preprocessing import PowerTransformer

# Random Forest
from sklearn.ensemble import RandomForestClassifier

# Decission Tree
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

from xgboost import XGBClassifier


from sklearn.metrics import (accuracy_score,
                             classification_report,
                             f1_score,
                             recall_score,
                             confusion_matrix,
                             classification_report)

from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import GridSearchCV

import plotly.express as px

import requests
from io import StringIO

from joblib import dump