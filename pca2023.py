from las_py import Laspy
import pandas as pd
import numpy as np
import matplotlib.pyplot as mpl
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib

mpl.rcParams['font.sans-serif'] = ['Microsoft Yahei'] 
mpl.rcParams['font.family'] = ['sans-serif']
mpl.rcParams['axes.unicode_minus'] = False

# Training set reading and data cleaning
df_train = pd.read_excel('example.xls')

df_d0 = df_train.drop(['#Depth', 'SP', 'CALI', 'RI', 'RXO'], axis=1)
df_dnan = df_d0.dropna(axis=0,how='any')
df_data = df_dnan.drop(['R'], axis=1)


# Training set normalization
features0 = StandardScaler().fit_transform(df_data)
# Principal component analysis
pca = PCA(n_components=3, whiten=True)
features_pca0 = pca.fit_transform(features0)
print('Main component：', pca.components_[0:3])


# Read the test set
df_test = pd.read_excel('example-test.xls')
df_d0 = df_test.drop(['#Depth', 'SP', 'CALI', 'RI', 'RXO'], axis=1)
df_dnan = df_d0.dropna(axis=0,how='any')
df_data = df_dnan.drop(['R'], axis=1)


# Training set normalization
features = StandardScaler().fit_transform(df_data)


# Principal component conversion
features_pca = pd.DataFrame(pca.fit_transform(features),columns=['F1','F2','F3'])


df_pcardata = np.column_stack((features_pca, df_dnan['R']))
# print(df_pcardata.shape)
df_pcar = pd.DataFrame(df_pcardata, columns=['F1', 'F2', 'F3', 'R'])

df_0 = df_pcar.drop(df_pcar[df_pcar['R']!=1].index) # mudstone data set
df_1 = df_pcar.drop(df_pcar[df_pcar['R']!=2].index) # coarse sandstone data set
df_2 = df_pcar.drop(df_pcar[df_pcar['R']!=3].index) # medium and fine sandstone data set
df_3 = df_pcar.drop(df_pcar[df_pcar['R']!=8].index) # coarse gravel data set
df_4 = df_pcar.drop(df_pcar[df_pcar['R']!=4].index) # medium and fine gravel data set
df_5 = df_pcar.drop(df_pcar[df_pcar['R']!=6].index) # coal rock data Set


fig0 = plt.figure(figsize=(15, 10), dpi=150)
ax0 = fig0.add_subplot(111)
fig1 = plt.figure(figsize=(15, 10), dpi=150)
ax1 = fig1.add_subplot(111)
fig2 = plt.figure(figsize=(15, 10), dpi=150)
ax2 = fig2.add_subplot(111)
ax0.scatter(df_1['F1'], df_1['F2'], c='b', marker='x', alpha=0.8, s=70, label='coarse sandstone')
ax0.scatter(df_2['F1'], df_2['F2'], c='aqua', marker='x', alpha=0.8, s=70, label='medium and fine sandstone')
ax0.scatter(df_3['F1'], df_3['F2'], c='purple', marker='+', alpha=0.8, s=70, label='coarse gravel')
ax0.scatter(df_4['F1'], df_4['F2'], c='orange', marker='+', alpha=0.8, s=70, label='medium and fine gravel')


ax1.scatter(df_1['F1'], df_1['F3'], c='b', marker='x', alpha=0.8, s=70, label='coarse sandstone')
ax1.scatter(df_2['F1'], df_2['F3'], c='aqua', marker='x', alpha=0.8, s=70, label='medium and fine sandstone')
ax1.scatter(df_3['F1'], df_3['F3'], c='purple', marker='+', alpha=0.8, s=70, label='coarse gravel')
ax1.scatter(df_4['F1'], df_4['F3'], c='orange', marker='+', alpha=0.8, s=70, label='medium and fine gravel')

ax2.scatter(df_1['F2'], df_1['F3'], c='b', marker='x', alpha=0.8, s=70, label='coarse sandstone')
ax2.scatter(df_2['F2'], df_2['F3'], c='aqua', marker='x', alpha=0.8, s=70, label='medium and fine sandstone')
ax2.scatter(df_3['F2'], df_3['F3'], c='purple', marker='+', alpha=0.8, s=70, label='coarse gravel')
ax2.scatter(df_4['F2'], df_4['F3'], c='orange', marker='+', alpha=0.8, s=70, label='medium and fine gravel')
# Set the label
xylabel = np.array([['F1', 'F2'], ['F1', 'F3'], ['F2', 'F3']])
mj = 'major'
mn = 'minor'
bo = 'both'
for i in range(3):    
    exec('ax%d.tick_params(labelsize=20)'%i)
    exec('ax%d.set_xlabel(xylabel[%d][0], fontsize=20)'%(i, i))
    exec('ax%d.set_ylabel(xylabel[%d][1], fontsize=20)'%(i, i))
    exec('ax%d.legend(loc=0,ncol=1,fontsize=20)'%i)   
    
    exec('ax%d.xaxis.set_minor_locator(plt.MultipleLocator(0.1))'%i)
    exec('ax%d.yaxis.set_minor_locator(plt.MultipleLocator(0.1))'%i)
    exec('ax%d.grid(which=mj, axis=bo,linewidth=0.75)'%i)
    exec('ax%d.grid(which=mn, axis=bo,linewidth=0.25)'%i)

plt.show()