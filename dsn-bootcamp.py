
#Create The Classifier

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
#matplotlib inline

#Reading file
#Importing the data
data1 = pd.read_csv('C:/Users/Hp/Desktop/DNS ML Bootcamp/Bootcamp/Train.csv') #Train dim((56000, 52))
data2 = pd.read_csv('C:/Users/Hp/Desktop/DNS ML Bootcamp/Bootcamp/Test.csv') #Test dim((24000, 51))
print("Shape of dataframe is: {}".format(data1.shape))
print("Shape of dataframe is: {}".format(data2.shape))

df1 = data1.copy()# Make a copy of the original sourcefile
df2 = data2.copy() ##Checkpoint (Saving a copy of datasets)
applic_id = df1['Applicant_ID']
applic_id.head()

#EDA
df1.info()
df2.info()
df1.columns.to_series().groupby(df1.dtypes).groups


#plt.figure(figsize=(6, 5))
#Freq = sns.countplot(x = 'no', order = [1, 0], data = df1['default_status'])

plt.title('\nDistribution of Default Status \n', fontsize = 20, weight = 'bold')
plt.xlabel('')
plt.ylabel('')
plt.tick_params(labelsize = 16,
                bottom = False,
                left = False)
Freq.set_xticklabels(labels = ['no', 'yes'])

plt.show(Freq)

df1['default_status'].iplot(kind='hist', xTitle='default_status',
                         yTitle='count', title='default_status Distribution')

# Drop columns not helpful to the model
#hotornot.drop(['SpotifyURI', 'ReleaseDate', 'Popularity'], axis = 1, inplace = True)
df1.drop(['Applicant_ID'], 1, inplace=True)
df2.drop(['Applicant_ID'], 1, inplace=True)
df1.head()
df2.head()


#Pre-processing Pipeline
#In this section, we undertake data pre-processing steps to prepare the datasets for Machine Learning algorithm implementation.

#Encoding
#Machine Learning algorithms can typically only have numerical values as their predictor variables. Hence Label Encoding becomes necessary as they encode categorical labels with numerical values. To avoid introducing feature importance for categorical features with large numbers of unique values, we will use both Lable Encoding and One-Hot Encoding as shown below.

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# Create a label encoder object
le = LabelEncoder()

print(df1.shape)
df1.head()



# Label Encoding will be used for columns with 2 or less unique values
le_count = 0
for col in df1.columns[1:]:
    if df1[col].dtype == 'object':
        if len(list(df1[col].unique())) <= 2:
            le.fit(df1[col])
            df1[col] = le.transform(df1[col])
            le_count += 1
print('{} columns were label encoded.'.format(le_count))


# Label Encoding will be used for columns with 2 or less unique values
le_count = 0
for col in df2.columns[1:]:
    if df2[col].dtype == 'object':
        if len(list(df2[col].unique())) <= 2:
            le.fit(df2[col])
            df2[col] = le.transform(df2[col])
            le_count += 1
print('{} columns were label encoded.'.format(le_count))


# convert rest of categorical variable into dummy
df1 = pd.get_dummies(df1, drop_first=True)

# convert rest of categorical variable into dummy
df2 = pd.get_dummies(df2, drop_first=True)

print(df1.shape)
df1.head()

print(df2.shape)
df2.head()

#Feature Scaling
#Feature Scaling using MinMaxScaler essentially shrinks the range such that the range is now between 0 and n. Machine Learning algorithms perform better when input numerical variables fall within a similar scale. In this case, we are scaling between 0 and 5.

# import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 5))
df_col = list(df1.columns)
df_col.remove('default_status')
for col in df_col:
    df1[col] = df1[col].astype(float)
    df1[[col]] = scaler.fit_transform(df1[[col]])
df1['default_status'] = pd.to_numeric(df1['default_status'], downcast='float')
df1.head()

print('Size of Full Encoded Dataset: {}'. format(df1.shape))


#Data Labelling
y_train = df1[['default_status']] #Target Variable
         
X_train = df1.loc[:, df1.columns != 'default_status']  #Predictor variable

X_test = df2
X_train.head()
#y_test =

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import scikitplot as skplt

# Split into test and train set 75:25
#X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, random_state = 0)

# Scaled Version of the dataset
#sc = MinMaxScaler()
#X_train_s = sc.fit_transform(X_train)
#X_test_s = sc.transform(X_test)

# UDF to show metrics for each model
def show_metrics(y_test, y_pred):
    '''
    Pass y_true and y_pred and print accuracy, precision, recall and f1 score.
    '''
    acc = round(accuracy_score(y_test, y_pred), 3)
    prec = round(precision_score(y_test, y_pred), 3)
    rec = round(recall_score(y_test, y_pred), 3)
    f1 = round(f1_score(y_test, y_pred), 3)
    
    print('Accuracy = {:.1%}'.format(acc))
    print('Precision = {:.1%}'.format(prec))
    print('Recall = {:.1%}'.format(rec))
    print('F1 score = {:.1%}'.format(f1))
    
def metrics_list(y_test, y_pred):
    '''
    Pass y_true and y_pred and return a list of metrics:
    accuracy, precision, recall and f1 score.
    '''
    
    acc = round(accuracy_score(y_test, y_pred), 3)
    prec = round(precision_score(y_test, y_pred), 3)
    rec = round(recall_score(y_test, y_pred), 3)
    f1 = round(f1_score(y_test, y_pred), 3)
    
    metrics = [acc, prec, rec, f1]
    
    return metrics


#Logistic Regression

from sklearn.linear_model import LogisticRegression

LR = LogisticRegression()
LR.fit(X_train, y_train)
LR_y_pred = LR.predict(X_test)

skplt.metrics.plot_confusion_matrix(
    y_test, LR_y_pred,
    normalize = True,
    figsize = (6, 4),
    title = 'Logistic Regression \nConfusion Matrix',
    text_fontsize = 20,
    title_fontsize = 22)
plt.ylim(1.5, -0.5)
#b += 0.5
#t -= 0.5
#plt.ylim(b, t)
plt.show()


show_metrics(y_test, LR_y_pred)

#Support Vector Machine

from sklearn.svm import SVC

SVM = SVC(gamma = 0.2, kernel = 'rbf', random_state = 0)
SVM.fit(X_train, y_train)
SVM.score(X_test, y_test)
SVM_y_pred = SVM.predict(X_test)

skplt.metrics.plot_confusion_matrix(
    y_test, SVM_y_pred,
    normalize = True,
    figsize = (6, 4),
    title = 'Support Vector Machine \nConfusion Matrix',
    text_fontsize = 20,
    title_fontsize = 22)
plt.ylim(1.5, -0.5)
#b += 0.5
#t -= 0.5
#plt.ylim(b, t)
plt.show()


show_metrics(y_test, SVM_y_pred)

#Random Forest
from sklearn.ensemble import RandomForestClassifier
#Random Forest
RF = RandomForestClassifier(random_state = 0)
RF.fit(X_train, y_train)
RF.score(X_test, y_test)
RF_y_pred = RF.predict(X_test)

skplt.metrics.plot_confusion_matrix(
    y_test, RF_y_pred,
    normalize = True,
    figsize = (6, 4),
    title = 'Random Forest \nConfusion Matrix',
    text_fontsize = 20,
    title_fontsize = 22)
plt.ylim(1.5, -0.5)
#b += 0.5
#t -= 0.5
#plt.ylim(b, t)
plt.show()


show_metrics(y_test, RF_y_pred)

Metrics = pd.DataFrame({
    'Model': ['Logistic Regression', 'Support Vector Machine', 'Random Forest'],
    'Accuracy' : [metrics_list(y_test, LR_y_pred)[0],
                  metrics_list(y_test, SVM_y_pred)[0],
                  metrics_list(y_test, RF_y_pred)[0]],             
    'Precision' : [metrics_list(y_test, LR_y_pred)[1],
                   metrics_list(y_test, SVM_y_pred)[1],
                   metrics_list(y_test, RF_y_pred)[1]],
    'Recall' : [metrics_list(y_test, LR_y_pred)[2],
                metrics_list(y_test, SVM_y_pred)[2],
                metrics_list(y_test, RF_y_pred)[2]],
    'F1_score' : [metrics_list(y_test, LR_y_pred)[3],
                  metrics_list(y_test, SVM_y_pred)[3],
                  metrics_list(y_test, RF_y_pred)[3]]})

print(Metrics)

plt.figure(figsize = (10,5))
barWidth = 0.2

bars1 = Metrics.iloc[0, :].tolist()[1:5]
bars2 = Metrics.iloc[1, :].tolist()[1:5]
bars3 = Metrics.iloc[2, :].tolist()[1:5]

r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

plt.bar(r1, bars1, width = barWidth, color = '#c73e39', edgecolor = 'white', label = 'Logistic Regression')
plt.bar(r2, bars2, width = barWidth, color = '#4d5262', edgecolor = 'white', label = 'Support Vector Machine')
plt.bar(r3, bars3, width = barWidth, color = '#5bcb93', edgecolor = 'white', label = 'Random Forest')

plt.title('\nModel Metrics \n', fontsize = 24, weight = 'bold')
plt.xticks([r + barWidth for r in range(len(bars1))],
           ['Accuracy', 'Precision', 'Recall', 'F1 score'])
plt.tick_params(labelsize = 16,
                bottom = False,
                left = False)
 
plt.legend(loc = 'lower right', prop = {'size': 10})
plt.show()
