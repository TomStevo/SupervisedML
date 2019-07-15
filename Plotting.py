import math
import matplotlib.pyplot as plt
import numpy as np

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
