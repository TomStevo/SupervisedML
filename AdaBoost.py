import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn import tree
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
import math
import time
import Datasets

def plot_scatter(sig,bkg,xvar='x',yvar='y'):
    plt.plot(sig[xvar],sig[yvar], 'o', c='tab:blue', label='sig', alpha=0.5, markeredgecolor='k')
    plt.plot(bkg[xvar],bkg[yvar], 'o', c='tab:orange', label='bkg', alpha=0.5, markeredgecolor='k')
    plt.legend()
    #plt.show()  
    return

def make_meshgrid(x, y, h=.01):
    x_h = 0.1*(x.max() - x.min())
    y_h = 0.1*(y.max() - y.min())
    x_min, x_max = x.min() - x_h, x.max() + x_h
    y_min, y_max = y.min() - y_h, y.max() + y_h
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1*x_h),
                         np.arange(y_min, y_max, 0.1*y_h))
    return xx, yy
    

def plot_contours(clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = plt.contourf(xx, yy, Z, **params)
    return out

def gen_spiral(a,s,r,w):
    x = a*(r*math.sin(r)+math.cos(r))+w
    y = a*(math.sin(r)-r*math.cos(r))-s+w
    return x, y

def evaluate_bdt(sig,bkg,clf):
    X = np.concatenate( [sig.values,bkg.values] )
    y = np.concatenate( [np.ones(len(sig.index)),np.zeros(len(bkg.index))] )

    print("Training BDT")
    clf.fit(X,y)

    #return
    print("Making grids")
    X0, X1 = X[:,0], X[:,1]
    xx, yy = make_meshgrid(X0,X1)

    print("Plotting contours")
    plot_contours(clf, xx, yy, colors=['tab:orange','tab:blue','tab:blue','tab:orange'], alpha=0.8)
    #plot_scatter(sig,bkg,"DER_mass_MMC","DER_deltar_tau_lep")
    plot_scatter(sig,bkg,"x","y")
    plt.show()
    return

vspiral = np.vectorize(gen_spiral, excluded=['a','s'])

def gen_chess(cls):
    x = []
    y = []
    while len(x) < 1000:
        r1 = np.random.uniform(0,1)
        r2 = np.random.uniform(0,1)
        if r1<0.25 and (r2<0.25 or (r2>0.5 and r2<0.75)):
            if cls == 's':
                x.append(r1)
                y.append(r2)
        elif r1>0.25 and r1<0.5 and ((r2>0.25 and r2<0.5) or r2>0.75):
            if cls == 's':
                x.append(r1)
                y.append(r2)
        elif r1>0.5 and r1<0.75 and (r2<0.25 or (r2>0.5 and r2<0.75)):
            if cls == 's':
                x.append(r1)
                y.append(r2)
        elif r1>0.75 and ((r2>0.25 and r2<0.5) or r2>0.75):
            if cls == 's':
                x.append(r1)
                y.append(r2)
        else:
            if cls == 'b':
                x.append(r1)
                y.append(r2)
    return x,y

def errorVsTree(sig_train,bkg_train,sig_test,bkg_test,clf):

    X_train = np.concatenate( [sig_train.values,bkg_train.values] )
    y_train = np.concatenate( [np.ones(len(sig_train.index)),np.zeros(len(bkg_train.index))] )
    X_test = np.concatenate( [sig_test.values,bkg_test.values] )
    y_test = np.concatenate( [np.ones(len(sig_test.index)),np.zeros(len(bkg_test.index))] )

    X_train,y_train = shuffle(X_train,y_train)
    X_test,y_test = shuffle(X_test,y_test)

    train_errors = []
    test_errors  = []
    
    for train_predict,test_predict in zip(clf.staged_predict(X_train),clf.staged_predict(X_test)):
        train_errors.append(1. - accuracy_score(train_predict, y_train))
        test_errors.append(1. - accuracy_score(test_predict, y_test))

    n_trees = len(clf)

    plt.plot(range(1, n_trees + 1), train_errors, c='red', label='Train')
    plt.plot(range(1, n_trees + 1), test_errors, c='black', label='Test')
    plt.legend()
    #plt.ylim(0.18, 0.62)
    plt.ylim(0.9*min(min(train_errors),min(test_errors)),1.1*max(max(train_errors),max(test_errors)))
    plt.ylabel('Test Error')
    plt.xlabel('Number of Trees')
    plt.show()

    return

def drop_neg(value):
    return value == -999.0

def get_class(value,cls):
    return value != cls

#df = pd.read_csv("training.csv")
#
#df = df.filter(regex='DER*|Label')
#
##df = df.filter(items=["DER_mass_MMC","DER_deltar_tau_lep","Label"])
#
#for col in df.filter(regex="DER*"):
#    df.drop( df[ np.vectorize(drop_neg)(df[col]) ].index, inplace=True )
#
#sig = df.drop( df[ np.vectorize(get_class,excluded=['cls'])(value=df.Label,cls="s") ].index ).drop('Label',axis=1)
#bkg = df.drop( df[ np.vectorize(get_class,excluded=['cls'])(value=df.Label,cls="b") ].index ).drop('Label',axis=1)
#
##temp_sig = sig.sample(1000)
##temp_sig = temp_sig.filter(regex='DER*')
##temp_bkg = bkg.sample(1000)
##temp_bkg = temp_bkg.filter(regex='DER*')
#
#train_sig = sig.sample(len(sig.index)//2)
#test_sig = sig.drop( train_sig.index, inplace=False)
#
#train_bkg = bkg.sample(len(bkg.index)//2)
#test_bkg = bkg.drop( train_bkg.index, inplace=False)
#
#clf = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=2),
#                         algorithm="SAMME",
#                         n_estimators=200)
#
#evaluate_bdt(train_sig,train_bkg,clf)
#
#errorVsTree(train_sig,train_bkg,test_sig,test_bkg,clf)
#
#exit()

# let's do the linear version first
mu_s = [0.45,0.45]
mu_b = [-0.45,-0.45]
sigma_s = [0.2, 0.3]
sigma_b = [0.2, 0.2]

sig = Datasets.gen_2Dgauss(mu_s,sigma_s,100)
bkg = Datasets.gen_2Dgauss(mu_b,sigma_b,100)

clf = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=4),
                         algorithm="SAMME",
                         n_estimators=200)

evaluate_bdt(sig,bkg,clf)

# Overlaps
mu_s = [0.2,0.2]
mu_b = [-0.2,-0.2]
sigma_s = [0.2, 0.3]
sigma_b = [0.2, 0.2]

sig = Datasets.gen_2Dgauss(mu_s,sigma_s,100)
bkg = Datasets.gen_2Dgauss(mu_b,sigma_b,100)

clf = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=4),
                         algorithm="SAMME",
                         n_estimators=200)

evaluate_bdt(sig,bkg,clf)

# Now for something less trivial

sig = Datasets.gen_spiral( a=0.2, s=0, n=1000 )
bkg = Datasets.gen_spiral( a=-0.2, s=0.2, n=1000 )

clf = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=4),
                         algorithm="SAMME",
                         n_estimators=200)

evaluate_bdt(sig,bkg,clf)

# Chessboard

sig = Datasets.gen_chess('s',1000)
bkg = Datasets.gen_chess('b',1000)

clf = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=4),
                         algorithm="SAMME",
                         n_estimators=200)

evaluate_bdt(sig,bkg,clf)
