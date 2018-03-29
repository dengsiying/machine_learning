import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
data = loadmat('ex3data1.mat')
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost(theta,X,y,learningrate):
    pass