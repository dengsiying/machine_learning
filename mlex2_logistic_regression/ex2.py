import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt

path = 'D:\ml ex2\ex2data1.txt'
data = pd.read_csv(path,header = None,names = ['Exam 1','Exam 2','Admitted'])
#创建两个分数的散点图，并使用颜色编码来可视化，如果样本是正的（被接纳）或负的（未被接纳）
positive = data[data['Admitted'].isin([1])]#选取data['Admitted']列中值为1的行 positive 也是个DATAframe
negative = data[data['Admitted'].isin([0])]#选取data['Admitted']列中值为0的行

fig,ax = plt.subplots(figsize=(12,8))
#scatter()  s=标量 c=颜色 marker=点的形状 一般默认为o 圆圈 x是叉 label是legend图例上显示的名称
ax.scatter(positive['Exam 1'],positive['Exam 2'],s=50,c='b',marker='o',label='Admitted')
ax.scatter(negative['Exam 1'],negative['Exam 2'],s=50,c='r',marker='x',label='Not Admitted')
ax.legend()#默认(自适应方式)
ax.set_xlabel('Exam 1 Score')
ax.set_ylabel('Exam 2 Score')
plt.show()

#sigmoid 函数
def sigmoid(z):
    return  1/(1+np.exp(-z))

nums = np.arange(-10,10,step=1)
fig,ax = plt.subplots(figsize=(12,8))
ax.plot(nums,sigmoid(nums),'r')
plt.show()

#编写代价函数来评估结果。 代价函数：J(θ)
def cost(theta,X,y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y,np.log(sigmoid(X*theta.T)))
    second = np.multiply((1-y),np.log(1-(sigmoid(X*theta.T))))
    return np.sum(first-second)/(len(X))

# add a ones column - this makes the matrix multiplication work out easier
data.insert(0,'Ones',1)
# set X (training data) and y (target variable)
cols = data.shape[1]
X = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]

# convert to numpy arrays and initalize the parameter array theta
X = np.array(X.values)
y = np.array(y.values)
theta = np.zeros(3) #array([ 0.,  0.,  0.])
#theta = np.matrix(np.array([0,0,0]))
#print(X.shape,y.shape,theta.shape)
#让我们计算初始化参数的代价函数(theta为0)。
#print(cost(theta,X,y))

#logistic regression gradient descent(逻辑回归 梯度下降)
def gradient(theta,X,y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)

    error = sigmoid(X * theta.T) - y

    for i in range(parameters):
        term = np.multiply(error, X[:, i])
        grad[i] = np.sum(term) / len(X)

    return grad
#print(gradient(theta,X,y))
#用SciPy's truncated newton（TNC）实现寻找最优参数。
result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(X, y))
print(result)
print(cost(result[0], X, y))
#编写一个函数，用我们所学的参数theta来为数据集X输出预测
def predict(theta,X):
    probability = sigmoid(X*theta.T)
    return [1 if x>=0.5 else 0 for x in probability]
theta_min = np.matrix(result[0])
print(theta_min)
predictions = predict(theta_min,X)
print(predictions)
correct = [1 if ((a==1 and b==1)or (a==0 and b==0)) else 0 for (a,b) in zip(predictions,y)]# zip函数接受任意多个(包括0个和1个)序列作为参数,返回一个tuple列表
#对可迭代函数'iterable'中的每一个元素应用‘function’方法，将结果作为list返回。
accuracy = (sum(map(int, correct)) % len(correct))
print ('accuracy = {0}%'.format(accuracy))

#正则化逻辑回归
path = 'ex2data2.txt'
data2 = pd.read_csv(path, header=None, names=['Test 1', 'Test 2', 'Accepted'])
positive = data2[data2['Accepted'].isin([0])]
negative = data2[data2['Accepted'].isin([1])]

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(positive['Test 1'], positive['Test 2'], s=50, c='b', marker='o', label='Accepted')
ax.scatter(negative['Test 1'], negative['Test 2'], s=50, c='r', marker='x', label='Rejected')
ax.legend()
ax.set_xlabel('Test 1 Score')
ax.set_ylabel('Test 2 Score')
plt.show()
#特征缩放 (不懂)
degree = 5
x1 = data2['Test 1']
x2 = data2['Test 2']
data2.insert(3,'Ones',1)
for i in range(1,degree):
    for j in range(0,i):
        data2['F'+str(i)+str(j)] = np.power(x1, i-j) * np.power(x2, j)

data2.drop('Test 1',axis=1,inplace=True)#删除Test 1 列
data2.drop('Test 2',axis=1,inplace=True)



#regularized cost（正则化代价函数）
def costReg(theta,X,y,learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y,np.log(sigmoid(X*theta.T)))
    second = np.multiply((1-y),np.log(1-sigmoid(X*theta.T)))
    reg = (learningRate/2*len(X))*np.sum(np.power(theta[:,1:theta.shape[1]],2))
    return np.sum(first-second)/(len(X))+reg


def gradientReg(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)

    error = sigmoid(X * theta.T) - y

    for i in range(parameters):
        term = np.multiply(error, X[:, i])

        if (i == 0):
            grad[i] = np.sum(term) / len(X)
        else:
            grad[i] = (np.sum(term) / len(X)) + ((learningRate / len(X)) * theta[:, i])

    return grad

# set X and y (remember from above that we moved the label to column 0)
cols = data2.shape[1]
X2 = data2.iloc[:,1:cols]
y2 = data2.iloc[:,0:1]

# convert to numpy arrays and initalize the parameter array theta
X2 = np.array(X2.values)
y2 = np.array(y2.values)
theta2 = np.zeros(11)
learningRate = 1
print(costReg(theta2, X2, y2, learningRate))
print(gradientReg(theta2, X2, y2, learningRate))
result2 = opt.fmin_tnc(func=costReg, x0=theta2, fprime=gradientReg, args=(X2, y2, learningRate))
theta_min = np.matrix(result2[0])
predictions = predict(theta_min, X2)
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y2)]
accuracy = (sum(map(int, correct)) % len(correct))
print ('accuracy = {0}%'.format(accuracy))
from sklearn import linear_model#调用sklearn的线性回归包
model = linear_model.LogisticRegression(penalty='l2', C=1.0)
model.fit(X2, y2.ravel())
print(model.score(X2, y2))


























