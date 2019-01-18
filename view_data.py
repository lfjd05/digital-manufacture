import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from scipy import sparse
import warnings
import time
import sys
import os
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.metrics import log_loss

train = pd.read_csv('dataset/jinnan_round1_train_20181227.csv', encoding='gb18030')
test = pd.read_csv('dataset/jinnan_round1_testA_20181227.csv', encoding='gb18030')
