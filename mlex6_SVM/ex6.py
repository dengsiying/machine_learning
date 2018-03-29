import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat
import sklearn.svm
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
sns.set(style='darkgrid')
raw_data = loadmat('data/ex6data1.mat')
data = pd.DataFrame(raw_data['X'],columns=['X1','X2'])
data['y'] = raw_data['y']
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(data['X1'], data['X2'], s=50, c=data['y'], cmap='RdBu')
ax.set_title('Raw data')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
plt.show()

svc1 = sklearn.svm.LinearSVC(C=1,loss='hinge')
svc1.fit(data[['X1','X2']],data['y'])
svc1.score(data[['X1','X2']],data['y'])#0.980392156863
data['SVM1 Confidence'] = svc1.decision_function(data[['X1','X2']])

fig, ax = plt.subplots(figsize=(8,6))
ax.scatter(data['X1'], data['X2'], s=50, c=data['SVM1 Confidence'], cmap='RdBu')
ax.set_title('SVM (C=1) Decision Confidence')
plt.show()

svc100 = sklearn.svm.LinearSVC(C=100, loss='hinge')
svc100.fit(data[['X1', 'X2']], data['y'])
svc100.score(data[['X1', 'X2']], data['y'])

data['SVM100 Confidence'] = svc100.decision_function(data[['X1', 'X2']])#预测样本的置信度分数。样本的置信度分数是样本与超平面的有符号距离。
fig, ax = plt.subplots(figsize=(8,6))
ax.scatter(data['X1'], data['X2'], s=50, c=data['SVM100 Confidence'], cmap='RdBu')
ax.set_title('SVM (C=100) Decision Confidence')
plt.show()

#print(data)

def gaussian_kernel(x1,x2,sigma):
    return np.exp(-np.sum((x1-x2)**2)/(2*(sigma**2)))

mat = loadmat('data/ex6data2.mat')
#print(mat.keys())#dict_keys(['__header__', '__version__', '__globals__', 'X', 'y'])
data = pd.DataFrame(mat.get('X'),columns=['X1','X2'])
data['y'] = mat.get('y')

sns.set(context='notebook',style='darkgrid')
fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(data['X1'], data['X2'],s=25, c=data['y'], cmap='RdBu')
ax.set_title('Raw data')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
plt.show()

svc = sklearn.svm.SVC(C=100,kernel='rbf',gamma=10,probability=True)
svc.fit(data[['X1','X2']],data['y'])
svc.score(data[['X1','X2']],data['y'])#0.9698725376593279
predict_prob = svc.predict_proba(data[['X1', 'X2']])[:, 0]
data['Probability'] = svc.predict_proba(data[['X1', 'X2']])[:,0]
#
fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(data['X1'], data['X2'], s=30, c=data['Probability'], cmap='RdBu')
plt.show()

mat = loadmat('data\ex6data3.mat')
training = pd.DataFrame(mat.get('X'),columns=['X1','X2'])
training['y'] = mat.get('y')#(211,3)

cv = pd.DataFrame(mat.get('Xval'),columns=['X1','X2'])
cv['y'] = mat.get('yval') #(200,3)

candidate = [0.01,0.03,0.1,0.3,1,0.3,1,3,10,30,100]
combination = [(C,gamma) for C in candidate for gamma in candidate]

search = []
for  C,gamma in combination:
    svc = sklearn.svm.SVC(C=C,gamma=gamma)
    svc.fit(training[['X1','X2']],training['y'])
    search.append(svc.score(cv[['X1','X2']],cv['y']))

best_score = search[np.argmax(search)]
best_param = combination[np.argmax(search)]

#print(best_param,best_score)#(0.3, 100) 0.965
best_svc = sklearn.svm.SVC(C=100,gamma=0.3)
best_svc.fit(training[['X1','X2']],training['y'])
ypred = best_svc.predict(cv[['X1','X2']])

#print(metrics.classification_report(cv['y'],ypred))

mat_tr = loadmat('data\spamTrain.mat')
X,y = mat_tr.get('X'),mat_tr.get('y').ravel()
#print(X.shape,y.shape) #(4000, 1899) (4000,)

mat_test = loadmat('data\spamTest.mat')
test_X ,test_y = mat_test.get('Xtest'),mat_test.get('ytest').ravel()
#((1000, 1899), (1000,))

svc = sklearn.svm.SVC()
svc.fit(X,y)
pred = svc.predict(test_X)
print(metrics.classification_report(test_y,pred))

logit = LogisticRegression()
logit.fit(X,y)
pred = logit.predict(test_X)
print(metrics.classification_report(test_y,pred))