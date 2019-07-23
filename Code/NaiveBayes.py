import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB, MultinomialNB
import Datasets
import Tools

mu_s = [0.45,0.45]
mu_b = [-0.45,-0.45]
sigma_s = [0.2, 0.3]
sigma_b = [0.2, 0.2]

sig = Datasets.gen_2Dgauss(mu_s,sigma_s,100)
bkg = Datasets.gen_2Dgauss(mu_b,sigma_b,100)

clf = GaussianNB()

Tools.train_mva(clf,sig,bkg)
Tools.evaluate_mva(clf,sig,bkg)
