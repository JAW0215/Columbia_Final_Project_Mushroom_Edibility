from ctypes.wintypes import SIZE
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Ridge



pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

data = pd.read_csv('Mushrooms.csv', delimiter= ';' )

data.replace({'class':{'p':0}}, inplace = True)
data.replace({'class':{'e':1}}, inplace = True)

data.replace({'cap-shape':{'b':0}}, inplace = True)
data.replace({'cap-shape':{'c':1}}, inplace = True)
data.replace({'cap-shape':{'x':2}}, inplace = True)
data.replace({'cap-shape':{'f':3}}, inplace = True)
data.replace({'cap-shape':{'s':4}}, inplace = True)
data.replace({'cap-shape':{'p':5}}, inplace = True)
data.replace({'cap-shape':{'o':6}}, inplace = True)

data.replace({'cap-color':{'n':0}}, inplace = True)
data.replace({'cap-color':{'b':1}}, inplace = True)
data.replace({'cap-color':{'g':2}}, inplace = True)
data.replace({'cap-color':{'r':3}}, inplace = True)
data.replace({'cap-color':{'p':4}}, inplace = True)
data.replace({'cap-color':{'u':5}}, inplace = True)
data.replace({'cap-color':{'e':6}}, inplace = True)
data.replace({'cap-color':{'w':7}}, inplace = True)
data.replace({'cap-color':{'y':8}}, inplace = True)
data.replace({'cap-color':{'l':9}}, inplace = True)
data.replace({'cap-color':{'k':10}}, inplace = True)
data.replace({'cap-color':{'o':11}}, inplace = True)

data.replace({'does-bruise-or-bleed':{'t':0}}, inplace = True)
data.replace({'does-bruise-or-bleed':{'f':1}}, inplace = True)


data.replace({'gill-color':{'n':0}}, inplace = True)
data.replace({'gill-color':{'b':1}}, inplace = True)
data.replace({'gill-color':{'g':2}}, inplace = True)
data.replace({'gill-color':{'r':3}}, inplace = True)
data.replace({'gill-color':{'p':4}}, inplace = True)
data.replace({'gill-color':{'u':5}}, inplace = True)
data.replace({'gill-color':{'e':6}}, inplace = True)
data.replace({'gill-color':{'w':7}}, inplace = True)
data.replace({'gill-color':{'y':8}}, inplace = True)
data.replace({'gill-color':{'l':9}}, inplace = True)
data.replace({'gill-color':{'k':10}}, inplace = True)
data.replace({'gill-color':{'f':11}}, inplace = True)
data.replace({'gill-color':{'o':12}}, inplace = True)

data.replace({'stem-color':{'n':0}}, inplace = True)
data.replace({'stem-color':{'b':1}}, inplace = True)
data.replace({'stem-color':{'g':2}}, inplace = True)
data.replace({'stem-color':{'r':3}}, inplace = True)
data.replace({'stem-color':{'p':4}}, inplace = True)
data.replace({'stem-color':{'u':5}}, inplace = True)
data.replace({'stem-color':{'e':6}}, inplace = True)
data.replace({'stem-color':{'w':7}}, inplace = True)
data.replace({'stem-color':{'y':8}}, inplace = True)
data.replace({'stem-color':{'l':9}}, inplace = True)
data.replace({'stem-color':{'k':10}}, inplace = True)
data.replace({'stem-color':{'f':11}}, inplace = True)
data.replace({'stem-color':{'o':11}}, inplace = True)

data.replace({'has-ring':{'t':0}}, inplace = True)
data.replace({'has-ring':{'f':1}}, inplace = True)

data.replace({'habitat':{'d':0}}, inplace = True)
data.replace({'habitat':{'m':1}}, inplace = True)
data.replace({'habitat':{'g':2}}, inplace = True)
data.replace({'habitat':{'h':3}}, inplace = True)
data.replace({'habitat':{'l':4}}, inplace = True)
data.replace({'habitat':{'p':5}}, inplace = True)
data.replace({'habitat':{'w':6}}, inplace = True)
data.replace({'habitat':{'u':7}}, inplace = True)

data.replace({'season':{'s':0}}, inplace = True)
data.replace({'season':{'w':1}}, inplace = True)
data.replace({'season':{'a':2}}, inplace = True)
data.replace({'season':{'u':3}}, inplace = True)

NAN_list = []
i = 0
for key in data.keys():
    list.append(i)
    i = 0
    
    for num in data.loc[:,key]:
        if (isinstance(num,str)==False):
            if math.isnan(num) ==True:
                i+=1
                
list.append(i)


#print(list, "\n", data.shape)
del list[0]
#print(list, "\n", data.shape, "\n\n", data.keys(), data.loc[:,'has-ring'],data.keys(),list)

#print(list,data.keys())

data_noNAN_Col = data.drop(columns=['cap-surface','gill-attachment','gill-spacing','stem-root','stem-surface','veil-type','veil-color','ring-type','spore-print-color'])

#[0, 0, 0, 14120, 0, 0, 9884, 25063, 0, 0, 0, 51538, 38124, 0, 57892, 53656, 0, 2471, 54715, 0, 0]



#Index(['class', 'cap-diameter', 'cap-shape', 'cap-surface', 'cap-color',
      # 'does-bruise-or-bleed', 'gill-attachment', 'gill-spacing', 'gill-color',
     #  'stem-height', 'stem-width', 'stem-root', 'stem-surface', 'stem-color',
  #     'veil-type', 'veil-color', 'has-ring', 'ring-type', 'spore-print-color',
    #   'habitat', 'season'],

#print(data_noNAN_Col.head(), "\n\n\n", data.head())

data_noNAN_RowCol = data_noNAN_Col.dropna()
#print(data_noNAN_RowCol.keys())

#--------------------------------  p==1          e == 0
X = data_noNAN_RowCol.loc[:,['cap-diameter', 'cap-shape', 'cap-color','does-bruise-or-bleed', 'gill-color', 'stem-height', 'stem-width','stem-color', 'has-ring', 'habitat', 'season']]
y =data_noNAN_RowCol.loc[:, 'class']







#DISPLAY DATA WITH MATPLOTLIB---------------------------------------------------------------------------------------------


#print(X.iloc[:,0], X.iloc[:,1])


#for i in range(2):
    #plt.scatter(X.iloc[:30000+30000*i,9], X.iloc[:30000+30000*i,0], c= y.loc[:29999+30000*i], s = .2)
    #plt.show()




#More Data

#sns.set_palette(sns.color_palette(["#A60707", "#B880DF"]))
#sns.pairplot(data, hue='class')

#plt.show()


#f, ax = plt.subplots(figsize=(8, 5))
#ax.set_xlim([0,30])
#ax.set_ylim([0,30])
#ax = sns.kdeplot(x=data['stem-width'], y=data['stem-height'], cmap="Greens", fill=True)

#sns.set_palette(sns.color_palette(["#F10C0C", "#B880DF"]))
#f, ax = plt.subplots(figsize=(8, 5))
#sns.countplot(x=data['season'], hue=data['class'])

#sns.set_palette(sns.color_palette(["#A60707", "#26C2C1"]))
#f, ax = plt.subplots(figsize=(8, 5))
#sns.violinplot(x='stem-color', y='stem-width', data=data, hue='class')


#plt.show()





#ALGORITHM------------------------------------------------------------------------------------------------------


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


#use decision tree, linear, nueral, knn - test each for best with scaling parameters
#i assume nueral will be best because large data



nears = []
#for i in range(100):
    #knn = KNeighborsClassifier(n_neighbors=i*1+1)
    #knn.fit(X_train, y_train)
    #nears.append(knn.score(X_test, y_test))
    #print("KNN Score: ", knn.score(X_test,y_test))
#print(max(nears))




knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
print("KNN Score: ", knn.score(X_test,y_test))

#KNN Score: 0.9918672561541401 at 1 neighbor



tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)

print("Tree Score: ", tree.score(X_test,y_test))

#Tree Score: 0.9849353201244474

rid = []
for i in range(100):
    ridge = Ridge(alpha=.00000001*i)
    ridge.fit(X_train, y_train)
    rid.append(ridge.score(X_test, y_test))


print("Linear Ridge Score: ", max(rid))

#Linear Ridge Score: 0.0740163557568887


mlp = MLPClassifier(solver='lbfgs', random_state=0, hidden_layer_sizes=[100, 100, 100, 20, 10, 2], max_iter= 1000)
mlp.fit(X_train, y_train)
print("Nueral Network Score: ", mlp.score(X_test, y_test))

#Nueral Network Score: 0.8317777413896621
