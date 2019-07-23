import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
import math
import Datasets
import Plotting

def evaluate_dt(sig,bkg,clf):
    X = np.concatenate( [sig.values,bkg.values] )
    y = np.concatenate( [np.ones(len(sig.index)),np.zeros(len(bkg.index))] )

    clf.fit(X,y)

    X0, X1 = X[:,0], X[:,1]
    xx, yy = Plotting.make_meshgrid(X0,X1)

    Plotting.plot_contours(clf, xx, yy, colors=['tab:orange','tab:blue','tab:blue','tab:orange'], alpha=0.8)
    Plotting.plot_scatter(sig,bkg,'x','y')
    plt.show()
    return

mu_s = [0.45,0.45]
mu_b = [-0.45,-0.45]
sigma_s = [0.2, 0.3]
sigma_b = [0.2, 0.2]

sig_train = Datasets.gen_2Dgauss(mu_s,sigma_s,1000)
sig_test = Datasets.gen_2Dgauss(mu_s,sigma_s,1000)

bkg_train = Datasets.gen_2Dgauss(mu_b,sigma_b,1000)
bkg_test = Datasets.gen_2Dgauss(mu_b,sigma_b,1000)

dt = tree.DecisionTreeClassifier(max_depth=3)
evaluate_dt(sig_train,bkg_train,dt)

tree.plot_tree(dt,feature_names=['x','y'],class_names=['sig','bkg'],filled=True)
plt.show()
