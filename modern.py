# SGD Classifier

import pandas as pd
import numpy as np
from datetime import datetime, date, time
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import scale, OneHotEncoder
from sklearn.metrics import log_loss
import scipy as sp


columns = ['C1','banner_pos','site_category','app_category',\
            'device_type','device_conn_type','C14','C15','C16','C17','C18','C19','C20','C21']

encode = [ 'C1','C14','C15','C16','C17','C18','C19','C20','C21' ]

def dayhour(timestr):
    d = datetime.strptime(str(x), "%y%m%d%H")
    return [float(d.weekday()), float(d.hour)]

filereader = pd.read_csv('testtrain.csv',chunksize=10000)

# classifier = SGDClassifier(loss='log')
fh = FeatureHasher(n_features = 2**20, input_type="string")

for chunk in filereader:
    y_train = chunk.as_matrix(columns=['click'])

    chunk = chunk.join(pd.DataFrame([dayhour(x) for x in chunk.hour], columns=["wd", "hr"]))

    chunk.drop(["hour"], axis=1, inplace = True)
    
    X_train = chunk.as_matrix(columns=[["wd", "hr"] + columns])

    print X_train

    Xcat = OneHotEncoder(categorical_features=["wd", "hr"] + columns)
    X = Xcat.fit(X_train)
    print X

    