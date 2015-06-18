import csv
import numpy
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction import FeatureHasher

#List to store the values retrieved from the database.
lst = []

#Read from the main train.csv
with open('site_train.csv','r') as f:
    rdr = csv.reader(f)
    i = 0
    for row in rdr:
        lst.append(row)
        i += 1
            #Read 200 rows
        if i == 200:
            break

headings = lst[0]

lst = numpy.array(lst[1:])
click = lst[:,1]

lst_ = lst[2:,2:]

lstp = lst[1,:]

lr = LinearRegression()
fh = FeatureHasher(n_features = 2**20, input_type="string")

lr.fit(fh.transform(lst_),click)
lr.predict(fh.transform(lstp))
