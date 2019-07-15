# based off of scikit-learn svm tutorial
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import math
import Datasets


def plot_scatter(sig,bkg,xvar='x',yvar='y'):
    plt.plot(sig[xvar],sig[yvar], 'o', c='tab:blue', label='sig', alpha=0.5, markeredgecolor='k')
    plt.plot(bkg[xvar],bkg[yvar], 'o', c='tab:orange', label='bkg', alpha=0.5, markeredgecolor='k')
    plt.legend()
    #plt.show()  
    return

def make_meshgrid(x, y, h=.01):
    x_min, x_max = x.min() - 0.1, x.max() + 0.1
    y_min, y_max = y.min() - 0.1, y.max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy
    

def plot_contours(clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = plt.contourf(xx, yy, Z, **params)
    return out

def evaluate_svm(sig,bkg,clf):
    X = np.concatenate( [sig.values,bkg.values] )
    y = np.concatenate( [np.ones(len(sig.index)),np.zeros(len(bkg.index))] )

    clf.fit(X,y)

    X0, X1 = X[:,0], X[:,1]
    xx, yy = make_meshgrid(X0,X1)

    plot_contours(clf, xx, yy, colors=['tab:orange','tab:blue','tab:blue','tab:orange'], alpha=0.8)
    plot_scatter(sig,bkg,'x','y')
    plt.show()
    return


#sig = datasets.gen_chess('s',1000)
#bkg = datasets.gen_chess('b',1000)

#sig = datasets.gen_spiral(a=0.2, s=0, n=1000)
#bkg = datasets.gen_spiral(a=-0.2, s=0.2, n=1000)

#plot_scatter(sig,bkg)
#plt.show()

# default is C=1.0
#clf = svm.SVC(kernel='linear',C=1.0)
#evaluate_svm(sig,bkg,clf)

# default is degree=3, coef0=0.0
#clf = svm.SVC(kernel='poly',degree=4)
#evaluate_svm(sig,bkg,clf)

#clf = svm.SVC(kernel='rbf',C=1.0,gamma=15)
#evaluate_svm(sig,bkg,clf)

# let's do the linear version first
#mu_s = [0.45,0.45]
#mu_b = [-0.45,-0.45]
#sigma_s = [0.2, 0.3]
#sigma_b = [0.2, 0.2]
#
#sig = Datasets.gen_2Dgauss(mu_s,sigma_s,100)
#bkg = Datasets.gen_2Dgauss(mu_b,sigma_b,100)
#
#clf = svm.SVC(kernel='linear')
#
#evaluate_svm(sig,bkg,clf)
#
## Overlaps
#mu_s = [0.2,0.2]
#mu_b = [-0.2,-0.2]
#sigma_s = [0.2, 0.3]
#sigma_b = [0.2, 0.2]
#
#sig = Datasets.gen_2Dgauss(mu_s,sigma_s,100)
#bkg = Datasets.gen_2Dgauss(mu_b,sigma_b,100)
#
#clf = svm.SVC(kernel='linear')
#evaluate_svm(sig,bkg,clf)
#
## Now for something less trivial
#
#sig = Datasets.gen_spiral( a=0.2, s=0, n=1000 )
#bkg = Datasets.gen_spiral( a=-0.2, s=0.2, n=1000 )
#
#clf = svm.SVC(kernel='rbf')
#
#evaluate_svm(sig,bkg,clf)
#
# Chessboard

sig = Datasets.gen_chess('s',1000)
bkg = Datasets.gen_chess('b',1000)

clf = svm.SVC(kernel='rbf',gamma=20)

evaluate_svm(sig,bkg,clf)
