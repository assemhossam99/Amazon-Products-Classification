import pickle
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from preProcessing import *

# Reading data from the dataset file and pre-processing it
data = pd.read_csv('AmazonProductClassification.csv')
data = preProcess(data)
cols =('manufacturer', 'category1', 'category2', 'seller_name_1', 'seller_name_2', 'seller_name_3')
data = Feature_Encoder(data, cols)

#Set the Input and the output data and split them
X = data.drop('ProductGrade', axis=1)
Y = data['ProductGrade']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)

print('Training Model starts')
startTrain = time.time()
model = SVC(kernel='poly', degree=5)
model.fit(X_train, Y_train)
endTrain = time.time()
print('Training Model ended')
print('Testing Model starts')
startTest = time.time()
accuracy = model.score(X_test, Y_test)
endTest = time.time()
print('Testing Model ended')
print('Model Accuracy: ', accuracy)
print('Model Training Time: ', endTrain - startTrain, ' seconds')
print('Model Testing Time: ', endTest - startTest, ' seconds')

#Save the trained modle
pickle.dump(model, open('SVM.sav', 'wb'))