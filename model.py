import pickle
import numpy as np
import sympy as sy
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

import scipy as sci
import statsmodels.api as sm
import sklearn
import sklearn.preprocessing as pre

from scipy import interpolate
from scipy import stats
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import norm, uniform
from scipy.stats import skew
from scipy.stats import boxcox 


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

from sklearn.datasets import load_digits

from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import roc_auc_score , roc_curve

from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestRegressor

from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC
from sklearn.svm import LinearSVC

from sklearn.multiclass import OneVsOneClassifier , OneVsRestClassifier , OutputCodeClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.neighbors import KNeighborsClassifier

from sklearn.cluster import KMeans
from feature_engine.imputation import CategoricalImputer

"""Read data"""

data = pd.read_csv('train.csv')

data.shape

"""# Data description"""

#data.describe(include='all')

#data.info()

"""# Pre-processing data

finding missing columns
"""

#missing_val_count_by_column = (data.isnull().sum())
#print(missing_val_count_by_column[missing_val_count_by_column > 0])

#data["education"].fillna("unkown").value_counts()/data["education"].count()

#data["previous_year_rating"].fillna("unkown").value_counts()/data["previous_year_rating"].count()

"""filling missing values """

data = CategoricalImputer(variables=['education'],imputation_method='frequent').fit_transform(data)

imputer = SimpleImputer(strategy='median', missing_values=np.nan).fit_transform(data[['previous_year_rating']])
data[['previous_year_rating']] = imputer
#data1

"""Seprating target variable"""

X = data.drop(['is_promoted','employee_id'],axis=1)
y = data['is_promoted']

"""segregate"""

X_num = X.select_dtypes(exclude=['object'])
X_num = X_num.drop('awards_won?',axis=1)
X_obj = X.select_dtypes(exclude=['int64','float64'])

"""Label encoding

"""

le = LabelEncoder()
oe = OrdinalEncoder()

object_cols = ['department','region','gender','recruitment_channel']

data_oe = pd.DataFrame(oe.fit_transform(X_obj[object_cols]))
data_oe.columns = object_cols

#y_le = le.fit_transform(y)
#y_le.columns = ['is_promoted']

"""One hot encoding"""

data_ohe = pd.get_dummies(X['education'], columns = ['education'])

#"""checking distribution"""

#sns.displot(x = data['previous_year_rating'])
#qqplot(X['age'],norm,fit=True,line="45")

"""Transform data acc. to skewness

"""

comp = X_num.select_dtypes(include=['float64','int64']).apply(lambda x : 
                                    [sci.stats.skew(x),
                                     sci.stats.kurtosis(x),
                                     x.mean(),x.median(), stats.iqr(x)]).T

comp.columns = ['skewness','kurtosis','mean','median','iqr']
comp

"""also can use box cox if data is positive"""

def transform(x):
    if round(skew(x),1) > 1 :
        if pd.Series(x<=0).any() :
            x = np.log(x+1.5)
            #print('significantly positive skewed with 0 values')
        else :
            x = np.log(x)
            #print('significantly positive skewed')
    elif round(skew(x),1) > 0.5  :
        x = np.sqrt(x)
        #print('moderately positive skewed')
    elif round(skew(x),1) < -1 :
        x = np.log(2-x)
        #print('significantly negetive skewed')
    elif round(skew(x),1) < -0.5 :
        x = np.sqrt(2-x)
        #print('moderately negetive skewed')
    else :
        None 
        #print('approx symmetric')
    
    return x

X_transform = X_num.apply(lambda x : transform(x))



#plt.figure(figsize=(22,22))
#g= sns.heatmap(data.corr(),annot=True,cmap='viridis',linewidths=.5)

#fig, axes = plt.subplots(2,2,figsize=(16,8))
#fig.suptitle('handling skewed data : previous_year_rating ')

#sns.distplot(X_num.previous_year_rating,ax=axes[0,0]).set(xlabel=None)#hue = 'X_numlist'
#axes[0,0].set_title('unchanged')

#sns.distplot( np.log(X_num.previous_year_rating),ax=axes[0,1]).set(xlabel=None)
#axes[0,1].set_title('log')

#sns.distplot(np.sqrt(X_num.previous_year_rating),ax=axes[1,0]).set(xlabel=None)
#axes[1,0].set_title('square root')

#boc,lmbda=sci.stats.boxcox(X_num.previous_year_rating,lmbda=None)
#sns.distplot( boc,ax=axes[1,1]).set(xlabel=None)
#axes[1,1].set_title('boxcox')

#plt.show()
#print('the skewness for ')
#print('original is ',skew(X_num.previous_year_rating))
#print('log is',skew(np.log(X_num.previous_year_rating)),)
#print('sqrt is',skew(np.sqrt(X_num.previous_year_rating)),)
#boc,lmbda=boxcox(X_num.previous_year_rating,lmbda=None)
#print('boxcox is',skew(boc))

#sns.kdeplot( data=X , x="previous_year_rating", hue="gender", fill=True, common_norm=False,alpha=.5, linewidth=0,)
#sns.kdeplot(data=X, x="previous_year_rating", hue="education")

#sns.catplot(data = data , y = 'age',x='education',hue='is_promoted',kind='box')

#sns.catplot(data = data1 , y = 'length_of_service',x='awards_won?',kind='boxen')

X_final = pd.concat([X_transform,data_oe, X['awards_won?'],data_ohe, ],axis=1)
X_final

X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.33 , random_state=5 , stratify = y )
#print(X_train.shape)
#print(y_train.shape)
#print(X_test.shape)
#print(y_test.shape)




#error_rate=[]
#for i in range(1,100):
#            knn = DecisionTreeClassifier(max_depth=i)
#            model = knn.fit(X_train,y_train)
#            pred_i = knn.predict(X_test)
#            error_rate.append(np.mean(pred_i != y_test))
#plt.figure(figsize=(13,8))
#plt.plot(range(1,100), error_rate, linestyle = 'dotted', marker = 'o',color = 'g')
#plt.xlabel('K value')
#plt.ylabel('Error Rate')
#plt.title('K value Vs Error Rate')
#plt.show()

classifier = DecisionTreeClassifier(max_depth=10).fit(X_train, y_train)
predict = classifier.predict(X_test)
#plot_confusion_matrix(model,X_test,y_test,cmap=plt.cm.Blues,xticks_rotation='vertical')
#print(classification_report(y_test,pred))
#metrics.accuracy_score(y_test, pred)

pickle.dump(classifier, open('model.pkl','wb'))
#load the model and test with a custom input
model = pickle.load( open('model.pkl','rb'))
