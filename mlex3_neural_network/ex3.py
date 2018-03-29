import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import matplotlib
#优化和拟合库scipy.optimize
import scipy.optimize as opt
from sklearn.metrics import classification_report#这个包是评价报告

data = sio.loadmat('ex3data1.mat')
print(data)

def load_data(path,transpose=True):
    data = sio.loadmat(path)
    y = data.get('y') # (5000,1)
    y = y.reshape(y.shape[0]) # make it back to column vector
    X = data.get('X')
    if transpose:
        X = np.array([im.reshape((20,20)).T for im in X])
        X = np.array([im.reshape(400) for im in X])

    return X,y

X,y = load_data('D:\ml ex3\ex3data1')
#print(X.shape)#(5000,400)
#print(y.shape)#(5000,)
def plot_an_image(image):
    fig,ax = plt.subplots(figsize=(1,1))
    ax.matshow(image.reshape((20,20)),cmap=matplotlib.cm.binary)
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))


pick_one  = np.random.randint(0,5000)
plot_an_image(X[pick_one,:])
plt.show()
print('this should be {}'.format(y[pick_one]))

def plot_100_image(X):
    """ sample 100 image and show them
       assume the image is square

       X : (5000, 400)
       """
    size = int(np.sqrt(X.shape[1]))
    sample_idx = np.random.choice(np.arange(X.shape[0]),100)
    sample_images = X[sample_idx,:]
    fig,ax_array = plt.subplots(nrows=10,ncols=10,sharey=True,sharex=True,figsize=(8,8))

    for r in range(10):
        for c in range(10):
            ax_array[r,c].matshow(sample_images[10*r+c].reshape((size,size)),cmap=matplotlib.cm.binary)
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))

plot_100_image(X)
plt.show()

raw_X,raw_y = load_data('ex3data1.mat')
#print(raw_X.shape) (5000,400)
#print(raw_y.shape) (5000,)
#准备数据
# add intercept=1 for x0
X = np.insert(raw_X,0,values=np.ones(raw_X.shape[0]),axis=1)
#print(X.shape) #(5000,401)

# y have 10 categories here. 1..10, they represent digit 0 as category 10 because matlab index start at 1
# I'll ditit 0, index 0 again
y_matrix = []
for k in range(1,11):
    y_matrix.append((raw_y==k).astype(int))
print(y_matrix)

# last one is k==10, it's digit 0, bring it to the first position，最后一列k=10，都是0，把最后一列放到第一列
y_matrix = [y_matrix[-1]]+y_matrix[:-1]
y = np.array(y_matrix)
# 扩展 5000*1 到 5000*10
#     比如 y=10 -> [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]: ndarray
#     """
#print(y)
#train 1 model（训练一维模型）
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def cost(theta,X,y):
    return  np.mean(-y*np.log(sigmoid(X@theta))-(1-y)*np.log(1-sigmoid(X@theta)))
def regularized_cost(theta,X,y,l=1):
    theta_j1_to_n = theta[1:]
    regularized_term = (1/2*(len(X)))*np.power(theta_j1_to_n,2).sum()
    return cost(theta,X,y)+regularized_term
def gradient(theta,X,y):
    return (1/len(X))*X.T@(sigmoid(X@theta)-y)
def regularized_gradient(theta,X,y,l=1):
    theta_j1_to_n = theta[1:]
    regularized_theta = (1/(2*len(X)))*theta_j1_to_n
    regularized_term = np.concatenate([np.array([0]),regularized_theta])
    return gradient(theta,X,y)+regularized_term
def logistic_regression(X,y,l=1):
    """generalized logistic regression
        args:
            X: feature matrix, (m, n+1) # with incercept x0=1
            y: target vector, (m, )
            l: lambda constant for regularization

        return: trained parameters
        """
    theta = np.zeros(X.shape[1])
    res = opt.minimize(fun=regularized_cost,x0=theta,args=(X,y,1),method='TNC',jac=regularized_gradient,options={'disp':True})
    final_theta =res.x
    return final_theta
def predict(x,theta):
    prob = sigmoid(x@theta)
    return (prob>=0.5).astype(int)
t0 = logistic_regression(X,y[0])
y_pred = predict(X,t0)
print('Accuracy={}'.format(np.mean(y[0] == y_pred)))
#train k model（训练k维模型）

k_theta = np.array([logistic_regression(X,y[k]) for k in range(10)])
#print(k_theta.shape) #(10,401)

prob_matrix = sigmoid(X@k_theta.T)
#np.set_printoptions(suppress=True)#如果不想省略中间部分，可以通过set_printoptions来强制NumPy打印所有数据。 suppress=True 是否使用科学记数法抑制小浮点值的打印（默认为False）。
#print(prob_matrix)
y_pred = np.argmax(prob_matrix,axis=1)#返回沿轴axis最大值的索引，默认axis=0  在列的向下方向上一行一行比 axis=1代表在行的向右方向上一列一列比
y_answer = raw_y.copy()
y_answer[y_answer==10] = 0
print(classification_report(y_answer,y_pred))

def load_weights(path):
    data  =  sio.loadmat(path)
    return data['Theta1'],data['Theta2']

theta1,theta2 = load_weights('ex3weights.mat')
#print(theta1.shape,theta2.shape) (25, 401) (10, 26)

X,y = load_data('ex3data1.mat',transpose=False)
X = np.insert(X,0,values=np.ones(X.shape[0]),axis=1)
#print(X.shape,y.shape) (5000,401) (5000,)

#feed forward prediction（前馈预测）
a1 = X
z2 = a1@theta1.T  # (5000, 401) @ (25,401).T = (5000, 25)
z2 = np.insert(z2,0,values=np.ones(z2.shape[0]),axis=1)
a2 = sigmoid(z2) #(5000,26)
z3 = a2@theta2.T #(5000,10)
a3 = sigmoid(z3)
#print(a3.shape)
y_pred = np.argmax(a3,axis=1)+1  # numpy is 0 base index, +1 for matlab convention\


print(classification_report(y, y_pred))















