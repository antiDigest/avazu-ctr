import csv

#List to store the values retrieved from the database.
lst = []
l = []

#Read from the main train.csv
with open('train.csv','rb') as f:
    rdr = csv.reader(f)
    i = 0
    for row in rdr:
            #l = row[0], row[1], row[2], row[4], row[5], row[6], row[7]
        lst.append(row)
        i += 1
        if i > 100000:
            break

#write to a new file subtrain.csv
with open('subtrain.csv','w') as f:
    writer = csv.writer(f)
    writer.writerows(lst)
