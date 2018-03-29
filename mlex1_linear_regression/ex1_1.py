import pandas as pd
import seaborn as sns
sns.set(context="notebook", style="whitegrid", palette="dark")
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
#读取数据并赋予列名
df = pd.read_csv('ex1data1.txt', names=['population', 'profit'])
#size: scalar, optional size #定义子图的高度 fit_reg 是否画出拟合的直线
sns.lmplot('population', 'profit', df, size=6, fit_reg=False)
plt.show()

def get_X(df):#读取特征
#     """
#     use concat to add intersect feature to avoid side effect
#     not efficient for big dataset though
#     """
#numpy.ones(shape, dtype=None, order='C') 创建 返回值就是一个给定类型和大小的数组 len(df) 行数
    ones = pd.DataFrame({'ones':np.ones(len(df))})#pd.DataFrame返回‘ones’是97行1列的dataframe  len(df) 返回df的行数
    data = pd.concat([ones,df],axis=1)#axis：需要合并链接的轴，0是行，1是列 axis=1 在列的方向上
    return data.iloc[:,:-1].as_matrix() #pandas.DataFrame.as_matrix  返回不是一个Numpy矩阵，而是一个Numpy数组。

def get_y(df):#读取标签
#     '''assume the last column is the target'''
    return np.array(df.iloc[:, -1])#df.iloc[:, -1]是指df的最后一列

def normalize_feature(df):
#     """Applies function along input axis(default 0) of DataFrame."""
#df.apply默认是将函数作用到每一列  axis=1 则为行
    return df.apply(lambda column: (column - column.mean()) / column.std())#特征缩放


def linear_regression(X_data, y_data, alpha, epoch,
                      optimizer=tf.train.GradientDescentOptimizer):  # 这个函数是旧金山的一个大神Lucas Shen写的
    # placeholder for graph input
    X = tf.placeholder(tf.float32, shape=X_data.shape)
    y = tf.placeholder(tf.float32, shape=y_data.shape)

    # construct the graph
    with tf.variable_scope('linear-regression'):
        W = tf.get_variable("weights",
                            (X_data.shape[1], 1),
                            initializer=tf.constant_initializer())  # n*1

        y_pred = tf.matmul(X, W)  # m*n @ n*1 -> m*1

        loss = 1 / (2 * len(X_data)) * tf.matmul((y_pred - y), (y_pred - y), transpose_a=True)  # (m*1).T @ m*1 = 1*1

    opt = optimizer(learning_rate=alpha)
    opt_operation = opt.minimize(loss)

    # run the session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        loss_data = []

        for i in range(epoch):
            _, loss_val, W_val = sess.run([opt_operation, loss, W], feed_dict={X: X_data, y: y_data})
            loss_data.append(loss_val[0, 0])  # because every loss_val is 1*1 ndarray

            if len(loss_data) > 1 and np.abs(
                            loss_data[-1] - loss_data[-2]) < 10 ** -9:  # early break when it's converged
                # print('Converged at epoch {}'.format(i))
                break

    # clear the graph
    tf.reset_default_graph()
    return {'loss': loss_data, 'parameters': W_val}  # just want to return in row vector format
data = pd.read_csv('ex1data1.txt', names=['population', 'profit'])
X = get_X(data)
#print(X.shape, type(X))

y = get_y(data)
#print(y.shape, type(y))
theta = np.zeros(X.shape[1]) # theta [ 0.  0.]   X.shape[1]=2,代表特征数n
def lr_cost(theta, X, y):
#     """
#     X: R(m*n), m 样本数, n 特征数
#     y: R(m)
#     theta : R(n), 线性回归的参数
#     """
    m = X.shape[0]#m为样本数

    inner = X @ theta - y  # R(m*1)，X @ theta等价于X.dot(theta)

    # 1*m @ m*1 = 1*1 in matrix multiplication
    # but you know numpy didn't do transpose in 1d array, so here is just a
    # vector inner product to itselves
    square_sum = inner.T @ inner
    cost = square_sum / (2 * m)

    return cost
lr_cost(theta, X, y)
#偏导数部分
def gradient(theta, X, y):
    m = X.shape[0]

    inner = X.T @ (X @ theta - y)  # (m,n).T @ (m, 1) -> (n, 1)，X @ theta等价于X.dot(theta)

    return inner / m
def batch_gradient_decent(theta, X, y, epoch, alpha=0.01):
#   拟合线性回归，返回参数和代价
#     epoch: 批处理的轮数
#     """
    cost_data = [lr_cost(theta, X, y)]
    _theta = theta.copy()  # 拷贝一份，不和原来的theta混淆

    for _ in range(epoch):
        _theta = _theta - alpha * gradient(_theta, X, y)
        cost_data.append(lr_cost(_theta, X, y))

    return _theta, cost_data
#批量梯度下降函数
epoch = 500
final_theta, cost_data = batch_gradient_decent(theta, X, y, epoch)
ax = sns.tsplot(time=np.arange(epoch+1),data=cost_data )
ax.set_xlabel('epoch')
ax.set_ylabel('cost')
plt.show()

b = final_theta[0] # intercept，Y轴上的截距
m = final_theta[1] # slope，斜率

plt.scatter(data.population, data.profit, label="Training data")
plt.plot(data.population, data.population*m + b, label="Prediction")
plt.legend(loc=2)
plt.show()
#multi-var batch gradient decent（多变量批量梯度下降）
raw_data = pd.read_csv('ex1data2.txt', names=['square', 'bedrooms', 'price'])
data = normalize_feature(raw_data)
X = get_X(data)
y = get_y(data)
alpha = 0.01#学习率
theta = np.zeros(X.shape[1])#X.shape[1]：特征数n
epoch = 500#轮数
final_theta, cost_data = batch_gradient_decent(theta, X, y, epoch, alpha=alpha)
#print(cost_data,len(cost_data))
sns.tsplot(time=np.arange(len(cost_data)), data = cost_data)
plt.xlabel('epoch', fontsize=18)
plt.ylabel('cost', fontsize=18)
plt.show()
#learning rate
#np.logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None)  base意思是取对数的时候log的下标
base = np.logspace(-1, -5, num=4)
#numpy.concatenate((a1,a2,...), axis=0)函数。能够一次完成多个数组的拼接 默认axis=0 在行的方向上拼接
candidate = np.sort(np.concatenate((base, base*3)))
#print(candidate)
epoch=50

fig, ax = plt.subplots(figsize=(16, 9))

for alpha in candidate:
    _, cost_data = batch_gradient_decent(theta, X, y, epoch, alpha=alpha)
    ax.plot(np.arange(epoch+1), cost_data, label=alpha)

ax.set_xlabel('epoch', fontsize=18)
ax.set_ylabel('cost', fontsize=18)
ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ax.set_title('learning rate', fontsize=18)
plt.show()

X_data = get_X(data)
#print(X_data.shape, type(X_data))

y_data = get_y(data).reshape(len(X_data), 1)  # special treatment for tensorflow input data
#print(y_data.shape, type(y_data))
epoch = 2000
alpha = 0.01
optimizer_dict={'GD': tf.train.GradientDescentOptimizer,
                'Adagrad': tf.train.AdagradOptimizer,
                'Adam': tf.train.AdamOptimizer,
                'Ftrl': tf.train.FtrlOptimizer,
                'RMS': tf.train.RMSPropOptimizer
               }
results = []
for name in optimizer_dict:
    res = linear_regression(X_data, y_data, alpha, epoch, optimizer=optimizer_dict[name])
    res['name'] = name
    results.append(res)
print(results)

fig, ax = plt.subplots(figsize=(16, 9))

for res in results:
    loss_data = res['loss']

    #     print('for optimizer {}'.format(res['name']))
    #     print('final parameters\n', res['parameters'])
    #     print('final loss={}\n'.format(loss_data[-1]))
    ax.plot(np.arange(len(loss_data)), loss_data, label=res['name'])

ax.set_xlabel('epoch', fontsize=18)
ax.set_ylabel('cost', fontsize=18)
#bbox_to_anchor：表示legend的位置，前一个表示左右，后一个表示上下。当使用这个参数时。loc将不再起正常的作用，ncol=3表示图例三列显示。 borderaxespad the pad between the axes and legend border
ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()