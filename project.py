#Authors - Stephen Morris and Lily Dinh

#Imports
import os
import pandas as pd 
import seaborn as sea
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as matp
import numpy as np  

#Get the path
"""
for getDirectory, _, files in os.walk('\CP468_Project\anime.csv'):
    for file in files:
        print(os.path.join(getDirectory, file))
"""

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
amountOfRows = 2500
data = pd.read_csv('anime.csv', delimiter=',', nrows = amountOfRows)
getNumRowAndCols = data.shape
data.dataframeName = 'animes.csv'
print('Amount of rows and columns equal {0}'.format(getNumRowAndCols))

CreateCorrMatrix(data, 10)