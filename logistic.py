# Avazu CTR prediction
# SGD Logistic regression + hashing trick.

import pandas as pd
import numpy as np
from datetime import datetime, date, time
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import LabelEncoder
import scipy as sp

cols = ["C1","banner_pos","site_category", "device_type","device_conn_type","C14","C15","C16","C17","C18","C19","C20","C21", "hour"]

def llfun(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll

# add two columns for hour and weekday
def dayhour(timestr):
    d = datetime.strptime(str(x), "%y%m%d%H")
    return [float(d.weekday()), float(d.hour)]

fh = FeatureHasher(n_features = 2**20, input_type="string")

# Train classifier
clf = SGDClassifier(loss="log", n_iter=1)
train = pd.read_csv("train/subtrain.csv", chunksize = 100000, iterator = True, skiprows=100000)
all_classes = np.array([0, 1])
for chunk in train:
    y_train = chunk["click"]
    chunk = chunk[cols]
    chunk = chunk.join(pd.DataFrame([dayhour(x) for x in chunk.hour], columns=["wd", "hr"]))
    chunk.drop(["hour"], axis=1, inplace = True)
    Xcat = fh.transform(np.asarray(chunk.astype(str)))
    clf.partial_fit(Xcat, y_train, classes=all_classes)
    
# Create a submission file
usecols = cols + ["id"]
X_test = pd.read_csv("test/mtest.csv", usecols=usecols)
X_test = X_test.join(pd.DataFrame([dayhour(x) for x in X_test.hour], columns=["wd", "hr"]))
X_test.drop(["hour"], axis=1, inplace = True)

X_enc_test = fh.transform(np.asarray(X_test.astype(str)))

y_act = pd.read_csv("test/mtest.csv", usecols=['click'])
print y_act
y_pred = clf.predict_proba(X_enc_test)[:, 1]
print y_pred
with open('logloss.txt','w') as f:
    f.write(llfun(y_act, y_pred))

with open("submission/submission_sgd.csv", "w") as f:
    f.write("id,click\n")
    for idx, xid in enumerate(X_test.id):
        f.write(str(xid) + "," + "{0:.10f}".format(y_pred[idx]) + "\n")
f.close()