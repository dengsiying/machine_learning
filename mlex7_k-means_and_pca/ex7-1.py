import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context="notebook", style="white")

import numpy as np
import pandas as pd
import scipy.io as sio

mat = sio.loadmat('data\ex7data1.mat')
X =  mat.get('X')
data = pd.DataFrame(X,columns=['X1','X2'])
sns.lmplot('X1','X2',data=data,fit_reg=False)
plt.show()


def pca(X):
    # normalize the features
    X = (X - X.mean()) / X.std()

    # compute the covariance matrix
    X = np.matrix(X)
    cov = (X.T * X) / X.shape[0]

    # perform SVD
    U, S, V = np.linalg.svd(cov)

    return U, S, V
U, S, V = pca(X)

def project_data(X, U, k):
    U_reduced = U[:,:k]
    return np.dot(X, U_reduced)
Z = project_data(X, U, 1)


fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 4))

sns.regplot('X1', 'X2',
           data=data,
           fit_reg=False,
           ax=ax1)
ax1.set_title('Original dimension')

sns.rugplot(Z, ax=ax2)
ax2.set_xlabel('Z')
ax2.set_title('Z dimension')
def recover_data(Z, U, k):
    U_reduced = U[:,:k]
    return np.dot(Z, U_reduced.T)

X_recovered = recover_data(Z, U, 1)


#fig, ax = plt.subplots(figsize=(12,8))
#ax.scatter(list(X_recovered[:, 0]), list(X_recovered[:, 1]))
#plt.show()



fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(12, 4))

sns.rugplot(Z, ax=ax1)
ax1.set_title('Z dimension')
ax1.set_xlabel('Z')

sns.regplot('X1', 'X2',
           data=pd.DataFrame(X_recovered, columns=['X1', 'X2']),
           fit_reg=False,
           ax=ax2)
ax2.set_title("2D projection from Z")

sns.regplot('X1', 'X2',
           data=pd.DataFrame(X, columns=['X1', 'X2']),
           fit_reg=False,
           ax=ax3)
ax3.set_title('Original dimension')

#faces = sio.loadmat('data\ex7faces.mat')
#X = faces['X']

def plot_n_image(X, n):
    """ plot first n images
    n has to be a square number
    """
    pic_size = int(np.sqrt(X.shape[1]))
    grid_size = int(np.sqrt(n))

    first_n_images = X[:n, :]

    fig, ax_array = plt.subplots(nrows=grid_size, ncols=grid_size,
                                    sharey=True, sharex=True, figsize=(8, 8))

    for r in range(grid_size):
        for c in range(grid_size):
            ax_array[r, c].imshow(first_n_images[grid_size * r + c].reshape((pic_size, pic_size)))
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))

#face = np.reshape(X[3,:], (32, 32))
#plt.imshow(face)
#plt.show()

#U, S, V = pca(X)
#Z = project_data(X, U, 100)
#X_recovered = recover_data(Z, U, 100)
#face = np.reshape(X_recovered[3,:], (32, 32))
#plt.imshow(face)
#plt.show()
mat = sio.loadmat('./data/ex7faces.mat')
X = np.array([x.reshape((32, 32)).T.reshape(1024) for x in mat.get('X')])
plot_n_image(X, n=64)
plt.show()
from sklearn.decomposition import PCA
sk_pca = PCA(n_components=100)
Z = sk_pca.fit_transform(X)
plot_n_image(Z, 64)
plt.show()
X_recover = sk_pca.inverse_transform(Z)
plot_n_image(X_recover, n=64)
plt.show()