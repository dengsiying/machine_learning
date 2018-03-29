import numpy as np
import scipy.io as sio
import scipy.optimize as opt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid')

def  load_data():
    """for ex5
      d['X'] shape = (12, 1)
      pandas has trouble taking this 2d ndarray to construct a dataframe, so I ravel
      the results
      """
    d = sio.loadmat('ex5data1.mat')
    return  map(np.ravel,[d['X'],d['y'],d['Xval'],d['yval'],d['Xtest'],d['ytest']])
X,y,Xval,yval,Xtest,ytest =load_data()

df = pd.DataFrame({'water_level':X,'flow':y})

sns.lmplot('water_level','flow',data=df,fit_reg=False,size=7)
plt.show()

X, Xval, Xtest = [np.insert(x.reshape(x.shape[0], 1), 0, np.ones(x.shape[0]), axis=1) for x in (X, Xval, Xtest)]
def cost(theta,X,y):
    """
       X: R(m*n), m records, n features
       y: R(m)
       theta : R(n), linear regression parameters
       """
    m = X.shape[0]
    inner = X@theta - y
    square_sum = inner.T@inner
    cost = square_sum/(2*m)
    return  cost
theta = np.ones(X.shape[1])
cost(theta,X,y)

def gradient(theta,X,y):
    m = X.shape[0]
    inner = X.T@(X@theta - y)
    return inner/m

def regularized_gradient(theta,X,y,l=1):
    m = X.shape[0]

    regularized_term = theta.copy()
    regularized_term[0] = 0
    regularized_term = (l/m)*regularized_term

    return gradient(theta,X,y)+regularized_term

def regularized_cost(theta,X,y,l=1):
    m = X.shape[0]
    regularized_term = (l/(2*m))*np.power(theta[1:],2).sum()
    return cost(theta,X,y)+regularized_term

def linear_regression_np(X,y,l=1):
    """linear regression
       args:
           X: feature matrix, (m, n+1) # with incercept x0=1
           y: target vector, (m, )
           l: lambda constant for regularization

       return: trained parameters
       """
    theta = np.ones(X.shape[1])
    res = opt.minimize(fun=regularized_cost,x0=theta,args=(X,y,l),method='TNC',jac=regularized_gradient,options={'disp':False})
    return res

final_theta = linear_regression_np(X,y,l=0).get('x')
b = final_theta[0]
m = final_theta[1]

plt.scatter(X[:,1],y,label='Training data')
plt.plot(X[:,1],X[:,1]*m+b,'r',label='Prediction')
plt.legend(loc=2)
plt.show()

training_cost, cv_cost = [],[]
#use the subset of training set to fit the model
#no regularization when you compute the training cost and CV cost
#remember to use the same subset of training set to compute training cost
m = X.shape[0]
for i in range(1,m+1):
    res  = linear_regression_np(X[:i,:],y[:i],l=0)
    tc = regularized_cost(res.x,X[:i,:],y[:i],l=0)
    cv = regularized_cost(res.x,Xval,yval,l=0)

    training_cost.append(tc)
    cv_cost.append(cv)

plt.plot(np.arange(1,m+1),training_cost,label='training cost')
plt.plot(np.arange(1,m+1),cv_cost,label='cv cost')
plt.legend()
plt.show() #under fitting

def poly_features(x, power,as_ndarray=False):
    data = {'f{}'.format(i):np.power(x,i) for i in range(1,power+1)}
    df = pd.DataFrame(data)
    return df.as_matrix() if as_ndarray else df


def normalize_feature(df):
    return df.apply(lambda  column : (column-column.mean())/column.std())


def prepare_poly_data(*args,power):
    def  prepare(x):
        df = poly_features(x,power=power)
        ndarr = normalize_feature(df).as_matrix()
        return  np.insert(ndarr,0,np.ones(ndarr.shape[0]),axis=1)
    return [prepare(x) for x in args]

X, y, Xval, yval, Xtest, ytest = load_data()


#print(poly_features(X,3))
X_poly,Xval_poly,Xtest_poly = prepare_poly_data(X,Xval,Xtest,power=8)
#print(X_poly[:3,:])

def plot_learning_curve(X,y,Xval,yval,l=0):
    training_cost,cv_cost = [],[]
    m = X.shape[0]

    for i in range(1,m+1):
        # regularization applies here for fitting parameters
        res = linear_regression_np(X[:i,:],y[:i],l=l)
        # remember, when you compute the cost here, you are computing
        # non-regularized cost. Regularization is used to fit parameters only
        tc = cost(res.x,X[:i,:],y[:i])
        cv = cost(res.x,Xval,yval)

        training_cost.append(tc)
        cv_cost.append(cv)

    plt.plot(np.arange(1,m+1),training_cost,label='training cost')
    plt.plot(np.arange(1,m+1),cv_cost,label='cv cost')
    plt.legend()

plot_learning_curve(X_poly,y,Xval_poly,yval,l=0)
plt.show() #过拟合
plot_learning_curve(X_poly,y,Xval_poly,yval,l=1)
plt.show() #减轻过拟合
plot_learning_curve(X_poly,y,Xval_poly,yval,l=100)
plt.show() #太多正则化了. 变成 欠拟合状态

l_candidate = [0,0.001,0.003,0.01,0.03,0.1,0.3,1,3,10]
training_cost,cv_cost = [],[]
for l in l_candidate:
    res = linear_regression_np(X_poly,y,l)
    tc = cost(res.x,X_poly,y)
    cv = cost(res.x,Xval_poly,yval)
    training_cost.append(tc)
    cv_cost.append(cv)

plt.plot(l_candidate,training_cost,label='training')
plt.plot(l_candidate,cv_cost,label='cross validation')
plt.legend()
plt.xlabel('lambda')
plt.ylabel('cost')
plt.show()
#print(l_candidate[np.argmin(cv_cost)]) # return 1
# use test data to compute the cost
for l in l_candidate:
    theta = linear_regression_np(X_poly,y,l).x
    print('test cost(l={})={}'.format(l,cost(theta,Xtest_poly,ytest)))

