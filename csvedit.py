import pandas as pd

filereader = pd.read_csv('train.csv',iterator='True',chunksize=100)

# for row in filereader:
#     row[:60].to_csv('subtrain.csv',header=True,index=0)
#     row[60:].to_csv('subtest.csv',header=True,index=0)
#     break

# for row in filereader:
#     row[:60].to_csv('subtrain.csv', mode='a',header=False,index=0)
#     row[60:].to_csv('subtest.csv', mode='a',header=False,index=0)