from las_py import Laspy
import pandas as pd
import numpy as np
import matplotlib.pyplot as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.markers as mmarkers
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split 
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV  
from sklearn import model_selection, metrics 
from sklearn.metrics import f1_score, precision_score, recall_score
import time
mpl.rcParams['font.sans-serif'] = ['Microsoft Yahei'] 
mpl.rcParams['font.family'] = ['sans-serif']
mpl.rcParams['axes.unicode_minus'] = False




def train(df):

    # Random forest training
    # 1 mudstone data set
    # 2 coarse sandstone data set
    # 3 medium and fine sandstone data set
    # 8 coarse gravel data set
    # 4 medium and fine gravel data set
    # 6 coal rock data Set

    df_d0 = df.drop(['#Depth', 'SP', 'CALI', 'RI', 'RXO'], axis=1)
    df_dnan = df_d0.dropna(axis=0,how='any')

    # Since the number divided by the original logging data is random, in order to avoid the impact of its order on the recognition of the model, the lithology classification is digital--> text conversion and then monothermal code encoding.
    lith_num = df_dnan['R']

    m = {1:'mudstone', 2:'coarse sandstone', 3:'medium and fine sandstone', 4:'medium and fine gravel data set', 8:'coarse gravel', 6:'coal rock'}

    lith_str = pd.DataFrame(list(map(lambda x:m[x],lith_num))) # Change the corresponding digital label to the corresponding lithology.

    # The label matrix is binaryized to become a multi-dimensional 0, 1 matrix.
    oh = preprocessing.LabelBinarizer()
    lith_oh = oh.fit_transform(lith_str)

    
    xdata = df_dnan[['GR', 'RT', 'AC', 'DEN','CNL']]
    ydata = lith_oh



    # ---------------------Use GridSearchCV for reference---------------------
#     start = time.time()
    # -------1.n_estimators, adjust the reference, the number of trees for decision trees-------

    n_estimators = range(1,71,1) # Parameter traversal range
    param_test1 = {'n_estimators':n_estimators}  # Name of parameter
    gsearch1 = GridSearchCV(estimator = RandomForestClassifier(min_samples_split=100,  
                                     min_samples_leaf=20,max_depth=8,max_features='sqrt' ,random_state=10),  
                           param_grid =param_test1,cv=5)  
    gsearch1.fit(xdata,ydata)  
    n_estimators_best = gsearch1.best_params_['n_estimators']


    # -------2.max_depth, maximum depth of decision tree in the best number of weak learner iterations.-------  

    #-------Depth-------
    max_depth = range(3,14,1)

    param_test2 = {'max_depth':max_depth}  
    gsearch2 = GridSearchCV(estimator = RandomForestClassifier(n_estimators= n_estimators_best,  
                                     min_samples_leaf=20,max_features='sqrt' ,random_state=10),  
       param_grid = param_test2, cv=5)  
    gsearch2.fit(xdata,ydata) 
    max_depth_best = gsearch2.best_params_['max_depth']


    # #-------3.min_samples_leaf and min_samples_split, adjust the reference------- 
    min_samples_split = range(40,150,5)
    min_samples_leaf = range(5,100,5)


    param_test3 = {'min_samples_split':min_samples_split}
    gsearch3 = GridSearchCV(estimator = RandomForestClassifier(n_estimators= n_estimators_best,max_depth=max_depth_best,  
                                     min_samples_leaf=10,max_features='sqrt', random_state=10),  
       param_grid = param_test3, cv=5)  
    gsearch3.fit(xdata,ydata)
    min_samples_split_best = gsearch3.best_params_['min_samples_split']

    param_test4 = {'min_samples_leaf':min_samples_leaf}
    gsearch4 = GridSearchCV(estimator = RandomForestClassifier(n_estimators= n_estimators_best,max_depth=max_depth_best,  
                                     min_samples_split=80,max_features='sqrt', random_state=10),  
       param_grid = param_test4, cv=5)  
    gsearch4.fit(xdata,ydata)
    min_samples_leaf_best = gsearch4.best_params_['min_samples_leaf']


    #-------4.max_features, adjust the reference------- 

    max_features = range(1,6,1)
    param_test5 = {'max_features':max_features}
    gsearch5= GridSearchCV(estimator = RandomForestClassifier(n_estimators= n_estimators_best,max_depth=max_depth_best,  
                                     min_samples_split=min_samples_split_best,min_samples_leaf=min_samples_leaf_best, random_state=10),  
       param_grid = param_test5, cv=5)  
    gsearch5.fit(xdata,ydata)
    max_features_best = gsearch5.best_params_['max_features']

#     print(n_estimators_best,max_depth_best,min_samples_split_best,min_samples_leaf_best,max_features_best)


    rfc = RandomForestClassifier(n_estimators= n_estimators_best,max_depth=max_depth_best,
                                 min_samples_split=min_samples_split_best,min_samples_leaf=min_samples_leaf_best,
                                 max_features=max_features_best,random_state=10,class_weight='balanced',oob_score=True)
    
    joblib.dump(rfc, ('D:/rfc.m'))

def rfr(df):

    

    df_d0 = df.drop(['#Depth', 'SP', 'CALI', 'RI', 'RXO'], axis=1)
    df_dnan = df_d0.dropna(axis=0,how='any')
    df_data = df_dnan.drop(['R'], axis=1)
    # Oilfield lithology code 1: 'mudstone', 2: 'coarse sandstone', 3: 'medium and fine sandstone', 4: 'medium and fine gravel', 8: 'coarse gravel', 6: 'coal rock'
    rfc = joblib.load(('D:/rfc.m')) # Read the rfc model
    rfc.n_jobs = -1
    # Set the forecast data
    xdata = df_data
    y_pred = rfc.predict(xdata)


    # Because the predicted result is a single thermal code, it needs to be decoded into a lithological text and then converted into the initial lithological number (matched with the oilfield data)
    # Set decoding rules
    oh = preprocessing.LabelBinarizer()
    lith_str = np.array([['mudstone'], ['coarse sandstone'], ['medium and fine sandstone'], ['medium and fine gravel data set'], ['coarse gravel'], ['coal rock']])
    oh.fit_transform(lith_str)
    lith_str = oh.inverse_transform(y_pred) # Unique hot code--> string

    m_inv = {'mudstone':1, 'coarse sandstone':2, 'medium and fine sandstone':3, 'medium and fine gravel data set':4, 'coarse gravel':8, 'coal rock':6}
    lith_num = pd.DataFrame(list(map(lambda x:m_inv[x],lith_str))) # Change the corresponding lithological label to the corresponding number.
    return lith_num, df_dnan

df_train = pd.read_excel('example.xls')
df_test = pd.read_excel('example-test.xls')
train(df_train)
lith_num, df_dnan = rfr(df_test)