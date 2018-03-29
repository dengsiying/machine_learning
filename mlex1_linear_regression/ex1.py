
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = 'D:\ml ex1\ex1data1.txt'
data = pd.read_csv(path,header=None,names=['Population','Profit'])
data.head()
data.describe()
#散点图（ scatter plot）
data.plot(kind = 'scatter',x = 'Population',y = 'Profit',figsize = (12,8))
plt.show()
#cost函数。cost函数评估我们模型的质量
# 通过（模型参数和实际数据点）计算我们模型对数据点的预测的误差（error）。
# 例如，如果给定城市人口是4而我们预测值是7，我们的误差就是(7-4)^2=3^2=9（假设一个L2约束或“最小二乘”损失函数）。
# 我们对每个X变量的数据点做计算并求和得到cost。
def computeCost(X,y,theta):#代价函数J(θ)
    inner = np.power(((X*theta.T)-y),2)
    return np.sum(inner)/(2*len(X))
#让我们在训练集中添加一列，以便我们可以使用向量化的解决方案来计算代价和梯度。
data.insert(0,'Ones',1)
# set X (training data) and y (target variable)
#矩阵有一个shape属性，是一个(行，列)形式的元组
cols = data.shape[1]#列
#iloc是根据标签所在的位置，从0开始计数。 "，"前面的"："表示选取整列 所有行
# 0:cols-1表示选取第0列到cols-1列，前闭后开，第cols-1列是不在范围之内
X = data.iloc[:,0:cols-1]
#最后一列
y = data.iloc[:,cols-1:cols]
X.head()
y.head()
#代价函数是应该是numpy矩阵，所以我们需要转换X和Y，然后才能使用它们。 我们还需要初始化theta。
X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix(np.array([0,0]))
print(X.shape, theta.shape, y.shape)
print(computeCost(X,y,theta))
#batch gradient decent（批量梯度下降）
def gradientDescent(X,y,theta,alpha,iters):
    temp = np.matrix(np.zeros(theta.shape))#生成相应大小的零矩阵
    parameters = int(theta.ravel().shape[1])#ravel()将多维数组降位一维后  取列数
    cost = np.zeros(iters)
    for i in range(iters):
        error = (X*theta.T)-y

        for j in range(parameters):
            term = np.multiply(error,X[:,j])
            # 更新权值
            temp[0,j] = theta[0,j]-((alpha/len(X))*np.sum(term))
        theta = temp
        cost[i] = computeCost(X,y,theta)

    return theta,cost

alpha = 0.01
iters = 1000
g,cost = gradientDescent(X,y,theta,alpha,iters)
print(g)
print(computeCost(X,y,g))
#linspace 创建等差数列
x = np.linspace(data.Population.min(), data.Population.max(), 100)
f = g[0, 0] + (g[0, 1] * x)
#通过.sublots()命令来创建新的figure对象, 可以通过设置figsize参数 大小设置
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(x, f, 'r', label='Prediction')#'r'红色 'k'黑色
ax.scatter(data.Population, data.Profit, label='Traning Data')
#loc= 参数设置
# 'best'         : 0  默认 (only implemented for axes legends)(自适应方式)
#'upper right'  : 1,
#'upper left'   : 2,
#'lower left'   : 3,
#'lower right'  : 4,
#'right'        : 5,
#'center left'  : 6,
#'center right' : 7,
#'lower center' : 8,
#'upper center' : 9,
#'center'       : 10,
ax.legend(loc=2) #图例位置
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
plt.show()

#由于梯度方程式函数也在每个训练迭代中输出一个代价的向量，所以我们也可以绘制。 请注意，代价总是降低 - 这是凸优化问题的一个例子。
fig , ax  = plt.subplots(figsize=(12,8))
ax.plot(np.arange(iters),cost,'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs Training Epoch')
plt.show()

#multiple variable
path = 'D:\ml ex1\ex1data2.txt'
data2 = pd.read_csv(path,header=None,names=['Size','Bedrooms','Price'])
#特征归一化 mean 平均值 std 标准差
data2 = (data2-data2.mean())/data2.std()
data2.insert(0, 'Ones', 1)
cols = data2.shape[1]
X2 = data2.iloc[:,0:cols-1]
y2 = data2.iloc[:,cols-1:cols]
X2 = np.matrix(X2.values)
y2 = np.matrix(y2.values)
theta2 = np.matrix(np.array([0,0,0]))
g2,cost2 = gradientDescent(X2,y2,theta2,alpha,iters)
print(computeCost(X2,y2,g2))
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(np.arange(iters), cost2, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.show()

#scikit-learn的线性回归函数
from sklearn import linear_model
model = linear_model.LinearRegression()
model.fit(X,y)
x = np.array(X[:, 1].A1)# .A1= Return self as a flattened ndarray Equivalent to np.asarray(x).ravel()
print(X[:, 1].A1)
print(x)
f = model.predict(X).flatten()#flatten() 返回一个折叠成一维的数组

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
plt.show()


#normal equation（正规方程）
def normalEqn(X,y):
    theta = np.linalg.inv(X.T@X)@X.T@y
    return  theta
final_theta2=normalEqn(X, y)
print(final_theta2)