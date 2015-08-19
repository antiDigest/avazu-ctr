import pandas as pd

filereader = pd.read_csv('train.csv',iterator='True',chunksize=100)

for row in filereader:
    row[:60].to_csv('testtrain.csv',header=True,index=0)
    row[60:].to_csv('testtest.csv',header=True,index=0)
    break

i = 1

for row in filereader:
    row[:60].to_csv('testtrain.csv', mode='a',header=False,index=0)
    row[60:].to_csv('testtest.csv', mode='a',header=False,index=0)
    i += 1

    if (i == 5000):
        break
