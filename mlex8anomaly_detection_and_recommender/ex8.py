import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context="notebook", style="white", palette=sns.color_palette("RdBu"))

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import stats
import anomaly
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
mat = sio.loadmat('data\ex8data1.mat')
X = mat.get('X')
#print(X.shape) (307, 2)
Xval,Xtest,yval,ytest = train_test_split(mat.get('Xval'),mat.get('yval').ravel(),test_size=0.5)
sns.regplot('Latency', 'Throughput',
           data=pd.DataFrame(X, columns=['Latency', 'Throughput']),
           fit_reg=False,
           scatter_kws={"s":30,
                        "alpha":0.5})
plt.show()

#estimate multivariate Gaussian parameters  μ  and  σ2
mu = X.mean(axis=0)

cov = np.cov(X.T)

# example of creating 2d grid to calculate probability density
np.dstack(np.mgrid[0:3,0:3])

multi_normal = stats.multivariate_normal(mu,cov)
x,y = np.mgrid[0:30:0.01,0:30:0.01]
pos = np.dstack((x,y))
fig, ax = plt.subplots()
#绘制等高线
ax.contourf(x,y,multi_normal.pdf(pos),camp='Blues')#pdf（x，mean = None，cov = 1）	概率密度函数
# plot original data points
sns.regplot('Latency', 'Throughput',
           data=pd.DataFrame(X, columns=['Latency', 'Throughput']),
           fit_reg=False,
           ax=ax,
           scatter_kws={"s":10,
                        "alpha":0.4})
plt.show()

def select_threshold(X,Xval,yval):
    """use CV data to find the best epsilon
       Returns:
           e: best epsilon with the highest f-score
           f-score: such best f-score
       """
    mu = X.mean(axis=0)
    cov = np.cov(X.T)
    multi_normal = stats.multivariate_normal(mu,cov)
    pval = multi_normal.pdf(Xval)
    epsilon = np.linspace(np.min(pval),np.max(pval),num = 10000)
    fs = []
    for e in epsilon:
        y_pred = (pval<=e).astype('int')
        fs.append(f1_score(yval,y_pred))

    argmax_fs = np.argmax(fs)
    return epsilon[argmax_fs],fs[argmax_fs]

e,fs = select_threshold(X,Xval,yval)
#print('Best epsilon:{}\nBest F-score on validation data:{}'.format(e,fs))
multi_normal, y_pred = anomaly.predict(X, Xval, e, Xtest, ytest)
# construct test DataFrame
data = pd.DataFrame(Xtest,columns=['Latency','Throughput'])
data['y_pred'] = y_pred

# create a grid for graphing
x, y = np.mgrid[0:30:0.01, 0:30:0.01]
pos = np.dstack((x, y))
fig, ax = plt.subplots()
# plot probability density
ax.contourf(x, y, multi_normal.pdf(pos), cmap='Blues')

# plot original Xval points
sns.regplot('Latency', 'Throughput',
            data=data,
            fit_reg=False,
            ax=ax,
            scatter_kws={"s":10,
                         "alpha":0.4})
# mark the predicted anamoly of CV data. We should have a test set for this...
#anamoly_data = data[data['y_pred'].isin([1])]
anamoly_data = data[data['y_pred']==1]
ax.scatter(anamoly_data['Latency'], anamoly_data['Throughput'], marker='x', s=50)
plt.show()

mat = sio.loadmat('./data/ex8data2.mat')
X = mat.get('X')
Xval, Xtest, yval, ytest = train_test_split(mat.get('Xval'),
                                            mat.get('yval').ravel(),
                                            test_size=0.5)
e, fs = anomaly.select_threshold(X, Xval, yval)
print('Best epsilon: {}\nBest F-score on validation data: {}'.format(e, fs))
multi_normal, y_pred = anomaly.predict(X, Xval, e, Xtest, ytest)
print('find {} anamolies'.format(y_pred.sum()))