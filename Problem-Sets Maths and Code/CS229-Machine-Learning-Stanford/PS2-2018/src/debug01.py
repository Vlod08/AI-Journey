# Important note: you do not have to modify this file for your homework.

import util
import numpy as np
import matplotlib.pyplot as plt


def calc_grad(X, Y, theta):
    """Compute the gradient of the loss with respect to theta."""

    m, n = X.shape
    margins = Y * X.dot(theta)                #computes the functional margin
    probs = 1. / (1 + np.exp(margins))        #compress the values from [-inf; +inf] => [0;1] 
    grad = -(1./m) * (X.T.dot(probs * Y))     
    return grad


def logistic_regression(X, Y,teta_trace):
    """Train a logistic regression model."""
    m, n = X.shape
    theta = np.zeros(n)
    learning_rate = 5

    i = 0
    while True:
        i += 1
        prev_theta = theta
        grad = calc_grad(X, Y, theta)
        theta = theta - learning_rate * grad
        teta_trace = np.append(teta_trace, theta.reshape(1, -1), axis=0)
        if i % 10000 == 0:
            print('Finished %d iterations' % i)
        if np.linalg.norm(prev_theta - theta) < 1e-15 or i>= 10e4:
            print('Converged in %d iterations' % i)
            break
    return teta_trace







def main():
    print('==== Training model on data set A ====')
    tetaa=np.array([0,0,0])
    tetaa=np.reshape(tetaa,(1,3))

    Xa, Ya = util.load_csv('../data/ds1_a.csv', add_intercept=True)
    logistic_regression(Xa, Ya,tetaa)
    plt.figure()
    plt.scatter(Xa[:,1],Xa[:,2],c=Ya)
    plt.savefig('../output/p01_lra.png')

    print('\n==== Training model on data set B ====')
    Xb, Yb = util.load_csv('../data/ds1_b.csv', add_intercept=True)
    tetab = np.array([0,0,0])
    tetab=np.reshape(tetab,(1,3))

    tetab = logistic_regression(Xb, Yb,tetab)
    #plt.figure()
    #plt.scatter(Xb[:,1],Xb[:,2],c=Yb)
    #plt.savefig('../output/p01_lrb.png')
    plt.figure()
    plt.scatter(tetab[:][1],tetab[:][2])
    plt.show()


if __name__ == '__main__':
    main()
