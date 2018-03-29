import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context="notebook", style="white", palette=sns.color_palette("RdBu"))

import numpy as np
import pandas as pd
import scipy.io as sio
import recommender as rcmd

movies_mat = sio.loadmat('data\ex8_movies.mat')
Y,R = movies_mat.get('Y'),movies_mat.get('R')
m,u = Y.shape
n = 10
param_mat = sio.loadmat('data\ex8_movieParams.mat')
theta ,X = param_mat.get('Theta'),param_mat.get('X')
users = 4
movies = 5
features = 3
X_sub = X[:movies,:features]
theta_sub = theta[:users,:features]
Y_sub = Y[:movies,:users]
R_sub = R[:movies,:users]
param_sub = rcmd.serialize(X_sub,theta_sub)
rcmd.cost(param_sub, Y_sub, R_sub, features)
param = rcmd.serialize(X,theta)
rcmd.cost(param,Y,R,10)
n_movie, n_user = Y.shape

X_grad, theta_grad = rcmd.deserialize(rcmd.gradient(param, Y, R, 10),
                                      n_movie, n_user, 10)
assert X_grad.shape == X.shape #断言语句
assert theta_grad.shape == theta.shape
rcmd.regularized_cost(param_sub, Y_sub, R_sub, features, l=1.5)
rcmd.regularized_cost(param, Y, R, 10, l=1)  # total regularized cost
n_movie, n_user = Y.shape

X_grad, theta_grad = rcmd.deserialize(rcmd.regularized_gradient(param, Y, R, 10),
                                                                n_movie, n_user, 10)

assert X_grad.shape == X.shape
assert theta_grad.shape == theta.shape
movie_list = []

with open('./data/movie_ids.txt', encoding='latin-1') as f:
    for line in f:
        tokens = line.strip().split(' ')
        movie_list.append(' '.join(tokens[1:]))

movie_list = np.array(movie_list)
ratings = np.zeros(1682)

ratings[0] = 4
ratings[6] = 3
ratings[11] = 5
ratings[53] = 4
ratings[63] = 5
ratings[65] = 3
ratings[68] = 5
ratings[97] = 2
ratings[182] = 4
ratings[225] = 5
ratings[354] = 5
Y, R = movies_mat.get('Y'), movies_mat.get('R')


Y = np.insert(Y, 0, ratings, axis=1)  # now I become user 0
R = np.insert(R, 0, ratings != 0, axis=1)
n_features = 50
n_movie, n_user = Y.shape
l = 10
X = np.random.standard_normal((n_movie, n_features))
theta = np.random.standard_normal((n_user, n_features))

param = rcmd.serialize(X, theta)
Y_norm = Y - Y.mean()
Y_norm.mean()
import scipy.optimize as opt
res = opt.minimize(fun=rcmd.regularized_cost,
                   x0=param,
                   args=(Y_norm, R, n_features, l),
                   method='TNC',
                   jac=rcmd.regularized_gradient)
X_trained, theta_trained = rcmd.deserialize(res.x, n_movie, n_user, n_features)
prediction = X_trained @ theta_trained.T
my_preds = prediction[:, 0] + Y.mean()
idx = np.argsort(my_preds)[::-1]  # Descending order
print(my_preds[idx][:10])
for m in movie_list[idx][:10]:
    print(m)

