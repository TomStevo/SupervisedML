import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from scipy.stats import ks_2samp
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
import Datasets
import Tools

def errorVsSize(sig,bkg,clf,cv,njobs,train_sizes=np.linspace(.1, 1.0, 20)):
    
    X = np.concatenate( [sig.values,bkg.values] )
    y = np.concatenate( [np.ones(len(sig.index)),np.zeros(len(bkg.index))] )

    X,y = shuffle(X,y)

    train_sizes, train_scores, test_scores = learning_curve(clf, X, y, cv=cv, n_jobs=njobs, train_sizes=train_sizes)
    
    train_errors_mean = np.mean(train_scores, axis=1)
    train_errors_std = np.std(train_scores, axis=1)
    test_errors_mean = np.mean(test_scores, axis=1)
    test_errors_std = np.std(test_scores, axis=1)

    plt.figure()
    plt.xlabel("Training Samples")
    plt.ylabel("1-Error")

    plt.grid()
    plt.fill_between(train_sizes, train_errors_mean - train_errors_std,
                     train_errors_mean + train_errors_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_errors_mean - test_errors_std,
                     test_errors_mean + test_errors_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_errors_mean, 'o-', color="r",
             label="Training Error")
    plt.plot(train_sizes, test_errors_mean, 'o-', color="g",
             label="Test Error")

    plt.legend(loc="best")
    plt.show()
    
    return

#mu_s = [0.2,0.2]
#mu_b = [-0.2,-0.2]
#sigma_s = [0.2, 0.3]
#sigma_b = [0.2, 0.2]
#
#sig = Datasets.gen_2Dgauss(mu_s,sigma_s,20)
#bkg = Datasets.gen_2Dgauss(mu_b,sigma_b,20)
#
#clf = svm.SVC(kernel='rbf',gamma='auto', C=1e6)
#
#Tools.train_mva(clf,sig,bkg)
#
#xx,yy = Tools.evaluate_mva(clf,sig,bkg)
#
#sig = Datasets.gen_2Dgauss(mu_s,sigma_s,100)
#bkg = Datasets.gen_2Dgauss(mu_b,sigma_b,100)
#
#Tools.evaluate_mva(clf,sig,bkg)
#
#exit()
#
sig = Datasets.gen_spiral( a=0.2, s=0, n=1000 )
bkg = Datasets.gen_spiral( a=-0.2, s=0.2, n=1000 )

clf = svm.SVC(kernel='rbf',gamma='auto')

Tools.train_mva(clf,sig,bkg)

Tools.calc_roc(clf,sig,bkg)

#cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
#errorVsSize(sig,bkg,clf,cv,4)
