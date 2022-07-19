#Authors - Stephen Morris and Lily Dinh

#Imports
import os
from turtle import color
from matplotlib import mlab
import nltk
from nltk.corpus import stopwords
#nltk.download('stopwords')
import pandas as pd 
import seaborn as sns
from sklearn import metrics
from sklearn.tree import DecisionTreeRegressor
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import plot_tree
import matplotlib.pyplot as matp
import matplotlib as mpl
import re
import numpy as np  


# Import not in use
#import tensorflow as ten
#import keras


#Get the path
"""
for getDirectory, _, files in os.walk('\CP468_Project\animes.csv'):
    for file in files:
        print(os.path.join(getDirectory, file))
"""

#Linear regression model
def linearFunc():
    amountOfRows = 17000
    data = pd.read_csv('animes.csv', delimiter=',', nrows = amountOfRows)
    data.dropna(inplace=True)
    df = pd.DataFrame(data)
    df.drop(['uid', 'episodes'], axis=1, inplace=True)

    x_axis = df['ranked'].values
    y_axis = df['score'].values
    
    print(f'Rank: {x_axis}')
    print(f'Score: {y_axis}')

    x_axis = x_axis.reshape(-1, 1)
    y_axis = y_axis.reshape(-1, 1)
    print(x_axis)
    print(y_axis)

    train_x_axis, test_x_axis, train_y_axis, test_y_axis = train_test_split(x_axis, y_axis, train_size=.10, test_size=.2, random_state=100)
    print(f'Train the data on x_axis {train_x_axis.shape}')
    print("Train the data on y_axis {0}".format(train_y_axis.shape))
    print("Testing the data on x_axis {0}".format(test_x_axis.shape))
    print("Testing the data on y_axis {0}".format(test_y_axis.shape))
    print()

    #Graphs:
    '''
    matp.rcParams['figure.figsize'] = [18, 12]
    matp.scatter(train_x_axis, train_y_axis)
    #data.plot(kind = "scatter", x=train_x_axis, y=train_y_axis)
    matp.ylabel('Score')
    matp.xlabel('Ranked')
    
    matp.title('Anime Scored/Rank')
    matp.show()
    '''
    linearModels = LinearRegression()
    linearModels.fit(train_x_axis, train_y_axis)
    prediction = linearModels.predict(test_x_axis)
    print(f'Training: {round(linearModels.score(train_x_axis, train_y_axis) * 100,2)}%')
    print(f'Predict: {round(linearModels.score(test_x_axis, test_y_axis) * 100, 2)}%')
    print(metrics.mean_absolute_error(test_y_axis, prediction))
    print(metrics.explained_variance_score(test_y_axis, prediction))

    

    matp.show()

#Logistic Regression model
def logFunc():
    amountOfRows = 17000
    data = pd.read_csv('animes.csv', nrows=17000)
    data.dropna(inplace=True)
    df = pd.DataFrame(data)
    df.drop(['uid', 'episodes', 'synopsis', 'aired', 'title', 'genre', 'img_url', 'link'], axis=1, inplace=True)

    y_axis = df['ranked']
    x_axis = df.drop('ranked', axis=1)

    train_x_axis, test_x_axis, train_y_axis, test_y_axis = train_test_split(x_axis, y_axis, test_size=0.3)
    
    logisticReg = LogisticRegression()
    logisticReg.fit(train_x_axis, train_y_axis)
    prediction = logisticReg.predict(test_x_axis)
    #print(classification_report(test_y_axis, prediction))
    print(confusion_matrix(test_y_axis, prediction))


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

def decisionTree():
    datas = pd.read_csv('animes.csv')
    x_axis = "ranked"
    y_axis = "popularity"
    Training, Target = datas[[x_axis]], datas[y_axis]
    Testing = pd.DataFrame(np.arange(Training[x_axis].min(),
                                   Training[x_axis].max()),
                                   columns=[x_axis])
    
    make_Tree = DecisionTreeRegressor(max_depth=1)
    make_Tree.fit(Training, Target)
    prediction = make_Tree.predict(Testing)
    sns.scatterplot(data=datas, x=x_axis, y=y_axis, color="black", alpha=0.9)
    matp.plot(Testing[x_axis], prediction, label="Decision tree")
    matp.legend()
    _ = matp.title("Decision Tree")
    matp.show

    _, axis = matp.subplots(figsize=(8, 6))
    _ = plot_tree(make_Tree, feature_names=x_axis, ax=axis)

# Count of rows and columns
amountOfRows = 17000
data = pd.read_csv('animes.csv', delimiter=',', nrows = amountOfRows)
df = pd.DataFrame(data)
df.drop(['uid', 'episodes'], axis=1, inplace=True)
getNumRowAndCols = data.shape
df.dataframeName = 'animes.csv'
print('Amount of rows and columns equal {0}'.format(getNumRowAndCols))

decisionTree()
#logFunc()
linearFunc()
#CreateCorrMatrix(df, 8) #Calls matrix function