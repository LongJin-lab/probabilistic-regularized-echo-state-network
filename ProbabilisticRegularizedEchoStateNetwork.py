from bayes_opt import BayesianOptimization
from numpy import *
from matplotlib.pyplot import *
import scipy.linalg
import os
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import preprocessing

def black_box_function(resSize,SP,gama,bate):
    data = loadtxt('data/lorenz.txt')
    scaler = preprocessing.MinMaxScaler((-1, 1))
    data = scaler.fit_transform(data)
    trainLen = int(data.shape[0] * 0.5 * 0.7)
    testLen = int(data.shape[0] * 0.5 * 0.3)
    initLen = 200  

    inSize = outSize = data.shape[1]  
    Win = (random.rand(int(resSize), 1 + int(inSize)) - 0.5) * 1 
    W = random.rand(int(resSize), int(resSize)) - 0.5  
    rhoW = max(abs(linalg.eig(W)[0]))  
    W *= SP/ rhoW
    
    X = zeros((trainLen - initLen, 1 + inSize + int(resSize)))  
    Yt = data[initLen + 1:trainLen + 1, :]  
    x = zeros((int(resSize), 1))
    for t in range(trainLen):
        u = data[t]
        u = reshape(u, (inSize, 1))
        x = tanh(dot(Win, vstack((1, u))) + dot(W, x))  
        if t >= initLen:  
            X[t - initLen, :] = vstack((1, u, x))[:, 0].T    
  
    X_T = X.T
    XXT = dot(X, X_T)
    I = eye(XXT.shape[0], XXT.shape[0])
    O = ones((XXT.shape[0], XXT.shape[0]))
    Wout = dot(dot(X_T, linalg.inv(dot(X, X_T) - I * ((trainLen * outSize) / gama) - O / bate)), Yt)    
    Y = zeros((testLen,outSize))
    u = data[trainLen]
    u = reshape(u, (inSize, 1))
    for t in range(testLen):
        x = tanh(dot(Win, vstack((1, u))) + dot(W, x))  
        y = dot(vstack((1, u, x)).T, Wout)  
        Y[t, :] = y[0, :]  
        u = data[trainLen + t + 1]
        u = reshape(u, (inSize, 1))
   
    errorLen = testLen
    return -sqrt(mean_absolute_error(data[trainLen + 1:trainLen + errorLen + 1, :], Y))


pbounds = {'resSize': (100, 1000), 'SP':(0, 1),'gama':(pow(e,-10),pow(e,30)), 'bate':(pow(e,-10),pow(e,30))}
optimizer = BayesianOptimization(
    f=black_box_function,
    pbounds=pbounds,
    # random_state=1,
)
optimizer.maximize(
    init_points=3,
    n_iter=5,
)

SP=round(optimizer.max.get('params').get('SP'),2)
resSize=int(optimizer.max.get('params').get('resSize'))
gama=round(optimizer.max.get('params').get('gama'),2)
bate=round(optimizer.max.get('params').get('bate'),2)

data = loadtxt('data/lorenz.txt')
scaler = preprocessing.MinMaxScaler((-1,1))
data=scaler.fit_transform(data)
trainLen=int(data.shape[0]*0.7)
testLen=data.shape[0]-trainLen-1
initLen = 500 
inSize = outSize = data.shape[1]
Win = (random.rand(int(resSize), 1 + inSize) - 0.5) * 1  
W = random.rand(int(resSize), int(resSize)) - 0.5  
W *= SP / rhoW
X = zeros((trainLen - initLen, 1 + inSize + int(resSize)))  
Yt = data[initLen + 1:trainLen + 1, :] 
x = zeros((int(resSize), 1))
for t in range(trainLen):
    u = data[t]
    u = reshape(u, (inSize, 1))
    x = tanh(dot(Win, vstack((1, u))) + dot(W, x))  
    if t >= initLen: 
        X[t - initLen, :] = vstack((1, u, x))[:, 0].T

X_T = X.T
XXT = dot(X, X_T)
I = eye(XXT.shape[0], XXT.shape[0])
O = ones((XXT.shape[0], XXT.shape[0]))
Wout = dot(dot(X_T, linalg.inv(dot(X, X_T) - I * ((trainLen * outSize) / gama) - O / bate)), Yt)
Y = zeros((testLen,outSize))
u = data[trainLen]
u = reshape(u, (inSize, 1))
for t in range(testLen):
    x = tanh(dot(Win, vstack((1, u))) + dot(W, x))  
    y = dot(vstack((1, u, x)).T, Wout)  
    Y[t, :] = y[0, :]  
    u = data[trainLen + t + 1]
    u = reshape(u, (inSize, 1))    
errorLen = testLen
print("mean_absolute_error:", mean_absolute_error(data[trainLen + 1:trainLen + errorLen + 1,:], Y))
print("mean_squared_error:", mean_squared_error(data[trainLen + 1:trainLen + errorLen + 1,:], Y))
print("rmse:", sqrt(mean_squared_error(data[trainLen + 1:trainLen + errorLen + 1,:], Y)))
print("r2 score:", r2_score(data[trainLen + 1:trainLen + errorLen + 1,:], Y))