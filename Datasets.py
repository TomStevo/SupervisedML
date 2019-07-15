import pandas as pd
import numpy as np
import math

def gen_gauss(mu, sigma, n):
    df = pd.DataFrame(np.random.normal(mu,sigma,(n,2)), columns=list('xy'))
    return df

def gen_spiral(a,s,n):
    
    r_rand = np.random.uniform(0,10,n)
    w_rand = np.random.normal(0,0.05,n)
    
    df = pd.DataFrame()
    df['x'], df['y'] = np.vectorize(spiral_func, excluded=['a','s'])( a=a, s=s, r=r_rand, w=w_rand)

    return df

def spiral_func(a,s,r,w):
    x = a*(r*math.sin(r)+math.cos(r))+w
    y = a*(math.sin(r)-r*math.cos(r))-s+w
    return x, y

def gen_chess(cls,n):
    x = []
    y = []
    while len(x) < n:
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

    df = pd.DataFrame()
    df['x'] = x
    df['y'] = y
    return df
