import csv
import math
import numpy

#List to store the values retrieved from the database.
data = []
Xs = []
y = []
e = 2.7182818

def sigmoid(z):
    g=[0,]
    for i in z:
        g += 1/(1+(e**(-i)))
        print g
    return g
    # except Error:
    #     print 'Only integer values allowed in z.'
        
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
    initial_theta = numpy.zeros((7,))

    costFunction(initial_theta)
    #sigmoid()

if __name__ == '__main__':
    main()