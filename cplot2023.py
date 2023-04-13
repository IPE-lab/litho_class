from las_py import Laspy
import pandas as pd
import numpy as np
import matplotlib.pyplot as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.markers as mmarkers
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

mpl.rcParams['font.sans-serif'] = ['Microsoft Yahei'] 
mpl.rcParams['font.family'] = ['sans-serif']
mpl.rcParams['axes.unicode_minus'] = False



df = pd.read_excel('example.xls')

fig0 = plt.figure(figsize=(15, 10), dpi=150)
ax0 = fig0.add_subplot(111)
fig1 = plt.figure(figsize=(15, 10), dpi=150)
ax1 = fig1.add_subplot(111)
fig2 = plt.figure(figsize=(15, 10), dpi=150)
ax2 = fig2.add_subplot(111)
fig3 = plt.figure(figsize=(15, 10), dpi=150)
ax3 = fig3.add_subplot(111)
fig4 = plt.figure(figsize=(15, 10), dpi=150)
ax4 = fig4.add_subplot(111)
fig5 = plt.figure(figsize=(15, 10), dpi=150)
ax5 = fig5.add_subplot(111)
# Confidence ellipse drawing
def confidence_ellipse(x, y, ax, n_std=2.3, facecolor='none', **kwargs): 
    if len(x) != len(y):
        raise ValueError("x and y must be the same size")
    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Find the characteristic root of the equation
    # Two-dimensional data
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)
    # Calculate the standard deviation
    # Calculate variance
    # Standard deviation
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)
    # Calculate y standard deviation
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)
    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

df_0 = df.drop(df[df['R']!=1].index) # mudstone data set
df_1 = df.drop(df[df['R']!=2].index) # coarse sandstone data set
df_2 = df.drop(df[df['R']!=3].index) # medium and fine sandstone data set
df_3 = df.drop(df[df['R']!=8].index) # coarse gravel data set
df_4 = df.drop(df[df['R']!=4].index) # medium and fine gravel data set
df_5 = df.drop(df[df['R']!=6].index) # coal rock data Set


# ax0 DEN&Z Crossplot

ax0.scatter(df_0['DEN'], df_0['AC'], c='r', marker='4', alpha=0.8, s=70, label='mudstone')
ax0.scatter(df_1['DEN'], df_1['AC'], c='b', marker='x', alpha=0.8, s=70, label='coarse sandstone')
ax0.scatter(df_2['DEN'], df_2['AC'], c='aqua', marker='x', alpha=0.8, s=70, label='medium and fine sandstone')
ax0.scatter(df_3['DEN'], df_3['AC'], c='purple', marker='+', alpha=0.8, s=70, label='coarse gravel')
ax0.scatter(df_4['DEN'], df_4['AC'], c='orange', marker='+', alpha=0.8, s=70, label='medium and fine gravel')
ax0.scatter(df_5['DEN'], df_5['AC'], c='black', marker=5, alpha=0.8, s=70, label='coal rock')

# ax1 CNL&GR Crossplot
ax1.scatter(df_0['CNL'], df_0['GR'], c='r', marker='4', alpha=0.8, s=70, label='mudstone')
ax1.scatter(df_1['CNL'], df_1['GR'], c='b', marker='x', alpha=0.8, s=70, label='coarse sandstone')
ax1.scatter(df_2['CNL'], df_2['GR'], c='aqua', marker='x', alpha=0.8, s=70, label='medium and fine sandstone')
ax1.scatter(df_3['CNL'], df_3['GR'], c='purple', marker='+', alpha=0.8, s=70, label='coarse gravel')
ax1.scatter(df_4['CNL'], df_4['GR'], c='orange', marker='+', alpha=0.8, s=70, label='medium and fine gravel')
ax1.scatter(df_5['CNL'], df_5['GR'], c='black', marker=5, alpha=0.8, s=70, label='coal rock')


# ax2 SP&RT Crossplot

ax2.scatter(df_0['SP'], df_0['RT'], c='r', marker='4', alpha=0.8, s=70, label='mudstone')
ax2.scatter(df_1['SP'], df_1['RT'], c='b', marker='x', alpha=0.8, s=70, label='coarse sandstone')
ax2.scatter(df_2['SP'], df_2['RT'], c='aqua', marker='x', alpha=0.8, s=70, label='medium and fine sandstone')
ax2.scatter(df_3['SP'], df_3['RT'], c='purple', marker='+', alpha=0.8, s=70, label='coarse gravel')
ax2.scatter(df_4['SP'], df_4['RT'], c='orange', marker='+', alpha=0.8, s=70, label='medium and fine gravel')
ax2.scatter(df_5['SP'], df_5['RT'], c='black', marker=5, alpha=0.8, s=70, label='coal rock')


# ax3 CALI&RI Crossplot
# df_0CALI = df_0.drop(df_0[df_0['CALI']<=4].index)
ax3.scatter(df_0['CALI'], df_0['RI'], c='r', marker='4', alpha=0.8, s=70, label='mudstone')
ax3.scatter(df_1['CALI'], df_1['RI'], c='b', marker='x', alpha=0.8, s=70, label='coarse sandstone')
ax3.scatter(df_2['CALI'], df_2['RI'], c='aqua', marker='x', alpha=0.8, s=70, label='medium and fine sandstone')
ax3.scatter(df_3['CALI'], df_3['RI'], c='purple', marker='+', alpha=0.8, s=70, label='coarse gravel')
ax3.scatter(df_4['CALI'], df_4['RI'], c='orange', marker='+', alpha=0.8, s=70, label='medium and fine gravel')
ax3.scatter(df_5['CALI'], df_5['RI'], c='black', marker=5, alpha=0.8, s=70, label='coal rock')
                          
# ax4 SP&RXO Crossplot

ax4.scatter(df_0['SP'], df_0['RXO'], c='r', marker='4', alpha=0.8, s=70, label='mudstone')
ax4.scatter(df_1['SP'], df_1['RXO'], c='b', marker='x', alpha=0.8, s=70, label='coarse sandstone')
ax4.scatter(df_2['SP'], df_2['RXO'], c='aqua', marker='x', alpha=0.8, s=70, label='medium and fine sandstone')
ax4.scatter(df_3['SP'], df_3['RXO'], c='purple', marker='+', alpha=0.8, s=70, label='coarse gravel')
ax4.scatter(df_4['SP'], df_4['RXO'], c='orange', marker='+', alpha=0.8, s=70, label='medium and fine gravel')
ax4.scatter(df_5['SP'], df_5['RXO'], c='black', marker=5, alpha=0.8, s=70, label='coal rock')

# ax5 CNL&RT Crossplot
ax5.scatter(df_0['CNL'], df_0['RT'], c='r', marker='4', alpha=0.8, s=70, label='mudstone')
ax5.scatter(df_1['CNL'], df_1['RT'], c='b', marker='x', alpha=0.8, s=70, label='coarse sandstone')
ax5.scatter(df_2['CNL'], df_2['RT'], c='aqua', marker='x', alpha=0.8, s=70, label='medium and fine sandstone')
ax5.scatter(df_3['CNL'], df_3['RT'], c='purple', marker='+', alpha=0.8, s=70, label='coarse gravel')
ax5.scatter(df_4['CNL'], df_4['RT'], c='orange', marker='+', alpha=0.8, s=70, label='medium and fine gravel')
ax5.scatter(df_5['CNL'], df_5['RT'], c='black', marker=5, alpha=0.8, s=70, label='coal rock')


# confidence_ellipse(df_0['CNL'], df_0['RT'], ax5, edgecolor='red')
# confidence_ellipse(df_1['CNL'], df_1['RT'], ax5, edgecolor='b')
# confidence_ellipse(df_2['CNL'], df_2['RT'], ax5, edgecolor='aqua')
# confidence_ellipse(df_3['CNL'], df_3['RT'], ax5, edgecolor='purple')
# confidence_ellipse(df_4['CNL'], df_4['RT'], ax5, edgecolor='orange')
# confidence_ellipse(df_5['CNL'], df_5['RT'], ax5, edgecolor='black')

# Set the label
xylabel = np.array([[r'DEN/(g·cm$^-$$^3$)', r'AC/(μs·ft$^-$$^1$)'], [r'RT/%', r'GR/API'], 
                     [r'SP/mV', r'RT/(Ω·m)'], [r'CALI/in', r'RI/(Ω·m)'],
                     [r'SP/mV', r'RXO/(Ω·m)'], [r'RT/%', r'RT/(Ω·m)']])
mj = 'major'
mn = 'minor'
bo = 'both'
for i in range(6):    
    exec('ax%d.tick_params(labelsize=20)'%i)
    exec('ax%d.set_xlabel(xylabel[%d][0], fontsize=20)'%(i, i))
    exec('ax%d.set_ylabel(xylabel[%d][1], fontsize=20)'%(i, i))
    exec('ax%d.legend(loc=0,ncol=1,fontsize=20)'%i) 
    exec('ax%d.xaxis.set_minor_locator(plt.MultipleLocator(1))'%i)
    exec('ax%d.yaxis.set_minor_locator(plt.MultipleLocator(1))'%i)
    exec('ax%d.grid(which=mj, axis=bo,linewidth=0.75)'%i)
    exec('ax%d.grid(which=mn, axis=bo,linewidth=0.25)'%i)

plt.show()
