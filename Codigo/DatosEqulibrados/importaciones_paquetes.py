# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 18:46:53 2022

@author: VicBoxMS
"""

#Pyspark
import findspark
findspark.init()
import pyspark
from pyspark import SparkContext
#Parche de sklearn
from sklearnex import patch_sklearn, config_context
patch_sklearn(['SVC'])
##Basicos
import numpy as np
import pandas as pd
from numpy import pi
import time
##Sklearn
from sklearn.datasets import load_breast_cancer, load_iris,make_moons, make_circles
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import f1_score , accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel,euclidean_distances
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
#Otros Intalado por defecto en algunos casos
import csv                 #tambien instalado por defecto
import warnings            #tambien instalado por defecto
warnings.filterwarnings("ignore")