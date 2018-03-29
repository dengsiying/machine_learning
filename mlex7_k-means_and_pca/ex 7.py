import seaborn as sns
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
import  numpy as np
from sklearn.cluster import KMeans

mat = sio.loadmat('data\ex7data1.mat')
data1 = pd.DataFrame(mat.get('X'),columns=['X1','X2'])
sns.set(context='notebook',style='white')
sns.lmplot('X1','X2',data=data1,fit_reg=False)
plt.show()

mat = sio.loadmat('./data/ex7data2.mat')
data2 = pd.DataFrame(mat.get('X'), columns=['X1', 'X2'])
sns.lmplot('X1', 'X2', data=data2, fit_reg=False)
plt.show()

clf = KMeans(n_clusters=3)
y_pred = clf.fit_predict(data2[['X1','X2']])
data2['y_pred'] = y_pred
sns.lmplot('X1', 'X2', data=data2,hue='y_pred', size=8,fit_reg=False)
plt.show()

#fig, ax = plt.subplots(figsize=(12,8))
#ax.scatter(data2['X1'], data2['X2'], s=50, c=data2['y_pred'], cmap='RdBu')
#plt.show()
from skimage import io

pic = io.imread('data/bird_small.png')/255
io.imshow(pic)
plt.show()
# serialize data
data = pic.reshape(128*128, 3)
model = KMeans(n_clusters=16)
model.fit(data)
centroids= model.cluster_centers_#(16, 3)
#print(centroids)
C = model.predict(data)#(16384,)
#print(centroids[C])
compressed_pic = centroids[C].reshape((128,128,3))
fig, ax = plt.subplots(1, 2)
ax[0].imshow(pic)
ax[1].imshow(compressed_pic)
plt.show()

