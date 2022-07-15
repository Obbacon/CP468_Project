#Authors - Stephen Morris and Lily Dinh

#Imports
import os
import nltk
from nltk.corpus import stopwords
#nltk.download('stopwords')
import pandas as pd 
import seaborn as sea
import sklearn

from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as matp
import re
import numpy as np  

# Import not in use

#import tensorflow as ten
#import keras


#Get the path
"""
for getDirectory, _, files in os.walk('\CP468_Project\anime.csv'):
    for file in files:
        print(os.path.join(getDirectory, file))
"""

#Linear regression model
def linearFunc():
    amountOfRows = 2500
    data = pd.read_csv('anime.csv', delimiter=',', nrows = amountOfRows)
    x_axis = data['Ranked'].values
    y_axis = data['Score'].values
    print(f'Rank: {x_axis}')
    print(f'Score: {y_axis}')

    x_axis = x_axis.reshape(-1, 1)
    y_axis = y_axis.reshape(-1, 1)
    print(x_axis)
    print(y_axis)

    train_x_axis, test_x_axis, train_y_axis, test_y_axis = train_test_split(x_axis, y_axis, train_size=.10, test_size=.2, random_state=90)
    print("Train the data on x_axis {0}".format(train_x_axis.shape))
    print("Train the data on y_axis {0}".format(train_y_axis.shape))
    print("Testing the data on x_axis {0}".format(test_x_axis.shape))
    print("Testing the data on y_axis {0}".format(test_y_axis.shape))
    print()

    #Graphs:


linearFunc()

# Function for making a correlation matrix;
def CreateCorrMatrix(data, getWidth):
    filename = data.dataframeName
    # Drop all NAN values
    #data = data.dropna('columns') 
    data = data[[columns for columns in data if data[columns].nunique() > 1]]
    if data.shape[1] < 2:
        print('No correlation')
        return
    corralation = data.corr()
    matp.figure(num=None, figsize=(getWidth, getWidth), dpi=70, facecolor='w', edgecolor='w')
    corrMat = matp.matshow(corralation, fignum = 1)
    matp.xticks(range(len(corralation.columns)), corralation.columns, rotation=90)
    matp.yticks(range(len(corralation.columns)), corralation.columns)
    matp.title(f'Correlation Matrix for {filename}', fontsize=14)
    matp.colorbar(corrMat)
    matp.show()

# Count of rows and columns
def printInfo():
    amountOfRows = 2500
    data = pd.read_csv('anime.csv', delimiter=',', nrows = amountOfRows)
    getNumRowAndCols = data.shape
    data.dataframeName = 'animes.csv'
    print('Amount of rows and columns equal {0}'.format(getNumRowAndCols))
#printInfo()


#CreateCorrMatrix(data, 10) #Calls matrix function