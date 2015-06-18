import csv
import math
import numpy

#List to store the values retrieved from the database.
data = []
Xs = []
y = []
e = 2.7182818

def sigmoid(z):
    '''if not isinstance(z,float) and not isinstance(z,int):
        g = numpy.zeros(len(z))    
        #print g
        k=0
        for i in z:
            g[k] = 1/(1+math.pow(e,(-i)))
            k += 1
        #print g
    else:'''
    g = 1/(1+(e**(-z)))
    return g
        #print g

def costFunction(theta):
    J = 0;
    grad = numpy.zeros(len(theta))
    m = len(y)
    k = 0
    print m
    for t in theta:
        #print t
        for i in range(0,m):
            for X in Xs[i]:
                #print "p"
                sumx = y[i] * math.log( sigmoid(X[i]*t) )
                suml = (1-y[i]) * math.log(1- sigmoid(X[i]*t) )
                sumz = X * ( sigmoid(X[i]*t) - y[i])
                print sumx,suml,sumz
            J = J - (1/m)*sumx + (suml)
            grad[k] = (1/m) * (sumz)
            k+=1

    print J
    print grad

def main():
    #Read from the main train.csv

    with open('site_train.csv','rb') as f:
        rdr = csv.reader(f)
        i=0
        for row in rdr:
            if i!=0:
                y.append(row[1])
                data = row[2], row[3], row[4], row[5], row[6]
                Xs.append(data)
                #print data
            i+=1
        #print Xs
    
    #sigmoid(0)
    initial_theta = numpy.zeros((7,))

    costFunction(initial_theta)
    #sigmoid()

if __name__ == '__main__':
    main()