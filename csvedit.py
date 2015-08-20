import pandas as pd

filereader = pd.read_csv('train.csv',iterator='True',chunksize=1000)

columns = ["C1","banner_pos","site_category","app_category", "device_type","device_conn_type"\
            ,"C14","C15","C16","C17","C18","C19","C20","C21", "hour",'click']

for row in filereader:
    row = row[columns]
    row[:600].to_csv('subtrain.csv',header=True,index=0)
    row[600:].to_csv('subtest.csv',header=True,index=0)
    break

i=1

for row in filereader:
    row = row[columns]
    row[:600].to_csv('subtrain.csv', mode='a',header=False,index=0)
    row[600:].to_csv('subtest.csv', mode='a',header=False,index=0)

    i+=1
    if i>=100:
        break