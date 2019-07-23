import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from scipy.stats import ks_2samp
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
import Datasets

def errorVsSize(sig,bkg,clf,cv,njobs,train_sizes=np.linspace(.1, 1.0, 10)):
    
    X = np.concatenate( [sig.values,bkg.values] )
    y = np.concatenate( [np.ones(len(sig.index)),np.zeros(len(bkg.index))] )

    X,y = shuffle(X,y)

    train_sizes, train_scores, test_scores = learning_curve(clf, X, y, cv=cv, n_jobs=njobs, train_sizes=train_sizes)
    
    train_errors_mean = np.mean(1-train_scores, axis=1)
    train_errors_std = np.std(1-train_scores, axis=1)
    test_errors_mean = np.mean(1-test_scores, axis=1)
    test_errors_std = np.std(1-test_scores, axis=1)

    plt.figure()
    plt.xlabel("Training examples")
    plt.ylabel("Error")

    plt.grid()
    plt.fill_between(train_sizes, train_errors_mean - train_errors_std,
                     train_errors_mean + train_errors_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_errors_mean - test_errors_std,
                     test_errors_mean + test_errors_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_errors_mean, 'o-', color="r",
             label="Training Error")
    plt.plot(train_sizes, test_errors_mean, 'o-', color="g",
             label="Cross-validation Error")

    plt.legend(loc="best")
    plt.show()
    
    return

sig = Datasets.gen_spiral( a=0.2, s=0, n=1000 )
bkg = Datasets.gen_spiral( a=-0.2, s=0.2, n=1000 )  

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
errorVsSize(sig,bkg,clf,cv,4)
