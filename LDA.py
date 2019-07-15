import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import Datasets

def get_class(label,cls):
    return label != cls

def cut_neg(val):
    return val < 0

def new_col(var1,var2,alpha1,alpha2):
    return var1*alpha1 + var2*alpha2

def get_sep(mus,mub,sigs,sigb):
    return ((mus-mub)**2)/(sigs**2 + sigb**2)

def plot_var(sig,bkg):
    vals = [sig.values,bkg.values]
    var_min = min( sig.mean()-5*sig.std(), bkg.mean()-5*bkg.std())
    var_max = max( sig.mean()+5*sig.std(), bkg.mean()+5*bkg.std())
    step = (var_max-var_min)/30
    bins = [ x*step + var_min for x in range(31) ]
    plt.hist(vals,bins=bins,stacked=False)
    plt.show()
    return

sigmaA = [0.5, 0.6]
sigmaB = [0.3, 0.2]
muA = [1.0,0.7]
muB = [0.1,-0.2]

sig = Datasets.gen_2Dgauss(muA,sigmaA,10000)
bkg = Datasets.gen_2Dgauss(muB,sigmaB,10000)

plot_var(sig.x,bkg.x)
plot_var(sig.y,bkg.y)

mu_s = np.array([sig.x.mean(),sig.y.mean()])
sigma_s = sig[ ['x','y'] ].cov().values

mu_b = np.array([bkg.x.mean(),bkg.y.mean()])
sigma_b = bkg[ ['x','y'] ].cov().values

mu_sub = mu_s - mu_b
cov_add = np.square(sigma_s) + np.square(sigma_b)

inv_cov = np.linalg.inv(cov_add)
alpha = inv_cov.dot(mu_sub)

sig['LDA'] = np.vectorize(new_col)(sig.x,sig.y,alpha[0],alpha[1])
bkg['LDA'] = np.vectorize(new_col)(bkg.x,bkg.y,alpha[0],alpha[1])

plot_var(sig.LDA,bkg.LDA)
#plt.show()

sep1 = get_sep(sig.x.mean(),bkg.x.mean(),sig.x.std(),bkg.x.std())
sep2 = get_sep(sig.y.mean(),bkg.y.mean(),sig.y.std(),bkg.y.std())

print(sep1)
print(sep2)
print(get_sep(sig.LDA.mean(),bkg.LDA.mean(),sig.LDA.std(),bkg.LDA.std()))
