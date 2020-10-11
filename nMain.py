import os
import glob
import time
import seaborn
import LogMaster1 as LM
import matplotlib.pyplot as mat
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


from collections import Counter
import more_itertools


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, confusion_matrix
import scipy.stats as stats
#from statsmodels.graphics.gofplots import qqplot


scaler = MinMaxScaler()
seaborn.set_style('ticks')
seaborn.set_context('paper', font_scale=0.9)
csfont = {'fontname':'Monospace'}


def Custom_colors(value_array):
    colors = []

    for item in value_array:
        if item == 1:
            colors.append('g')
        else:
            colors.append('r')

    return colors


def Ponderm1(label_item):

    label_item = label_item.tolist()
    count = label_item.groupby(-1).count()
    return count

def Ponder1(label_item):

    label_item = label_item.tolist()
    count = label_item.groupby(1).count()
    return count


def Rolling_window_Count(iterable, size):
    iters = tee(iterable, size)
    for i in range(1, size):
        for each in iters[i:]:
            next(each, None)
    return zip(*iters)


##################################################################################################

# DEVELOPERS ONLY. only file_list has a "general" importance. Variance_Analysis is a purely speculative function. So, decomment the following two lines
# only to see some descriptive volatility plots and generate your dataframe.
#======================================#
file_list = glob.glob('storm*.zip')#('2019*.zip')## #
print(file_list)                                  #
Around_the_clock = LM.Variance_Analysis(file_list)#
#======================================#
##################################################################################################

Time_series = pd.read_csv(os.getcwd()+'/variation_db', sep=',')
#Time_series['RollWin'] = np.array(ClockTime_list)

#print(Time_series)

##################################################################################################

#============================================================#
#Anomaly detection training with IsolationForests and/or OCSVM
#============================================================#

anomaly_detector = OneClassSVM(kernel = 'rbf', nu = 0.2, gamma='auto')#IsolationForest(n_estimators = 100, behaviour = 'new', bootstrap = True, contamination = .2)   #crosscheck the .12#
for item_name in file_list:
    detecting_on = np.array(Time_series[str(item_name)])
    detecting_on = scaler.fit_transform(detecting_on.reshape(-1, 1))
    pUnderDetection = np.array(Time_series[str(item_name)])
    #print('ESTIMATING ON' + item_name, estimated)
    detection = anomaly_detector.fit_predict(detecting_on.reshape(-1,1))

    #====06/10/2020=====#
    #aggiunto salvataggio documentale della predizione dei singoli punti
    '''il formato dell'output è un file di testo a due colonne con:
                Prima colonna   :  Valore del punto
                Seconda colonna :  Classificazione OCSVM
    '''

    outputData = {'Volatility': np.array(pUnderDetection), 'Prediction': np.array(detection)}
    data = pd.DataFrame(outputData)
    data.to_csv(r'outputfe.txt', header=['volatility','prediction'], index=None, sep=';', mode='a')

    c_palette = Custom_colors(detection)
    r_patch = mpatches.Patch(color ='red', label='OCSVM = Anomaly')
    g_patch = mpatches.Patch(color='g', label= 'OCSVM = Regular')
    o_patch = mpatches.Patch(color='orange', label= 'Vol_gradient')

#   pct_change = (np.diff(detecting_on, axis = 0)/detecting_on[:-1])*100

    #Time_series['perc.change.'] = Time_series[str(item_name)].pct_change()
    #stat test for pct_change distribution.
    #alpha = 0.05
    #stat, p = shapiro(pct_change)
    #print('Statistics=%.3f, p=%.3f' % (stat, p))
    #if p > alpha:
    #    print('Sample looks Gaussian (fail to reject H0)')
    #    mean_change = np.mean(pct_change)
    #    std_change = np.std(pct_change)
    #else:
    #    print('Sample does not look Gaussian (reject H0)')



    fig, (ax,ay) = mat.subplots(2, figsize = (20,15), sharex=False, sharey=False)
    #ax.plot(range(0, len(complete_frame.volatility)), complete_frame.volatility, color = 'k', alpha = 0.6)
    ax.scatter(detecting_on, np.ones(len(detecting_on)),  color = c_palette, s = 50)
    ax.set_xlabel('Volatility (u.a.)')
    ax.legend(handles = [r_patch, g_patch], loc = 'upper right', fancybox=True, framealpha=1, shadow=True, borderpad=1)

    ay.plot(detecting_on, linewidth = 2, color = 'indigo')
    ay.scatter(range(0, len(detecting_on)), detecting_on, color = c_palette, s = 20)
    ay.set_xlabel('TW index')
    ay.set_ylabel('Volatility (u.a.)')
    ay.legend(handles = [r_patch, g_patch], loc = 'upper right', fancybox=True, framealpha=1, shadow=True, borderpad=1)
    fig.suptitle('Anomaly detection on' + ' ' + str(item_name), **csfont)


    #az.plot(pct_change, color = 'navy', linewidth = 2, label = 'Percentage shift', alpha = 0.6)
    #az.plot(np.gradient(Time_series['perc.change.']), color = 'gold', linewidth = 3, label = '%shift gradient')
    #threat_level = []
    #for num in Time_series['perc.change.']:
    #    if (mean_change-std_change <= num <= mean_change+std_change):
    #        threat_level.append('blue')
    #    elif (mean_change-2*std_change <= num <= mean_change-std_change) or (mean_change+std_change <= num <= mean_change+2*std_change):
    #        threat_level.append('orange')
    #    else:
    #        threat_level.append('crimson')

    #az.set_xlabel('TW index')
    #az.set_ylabel('% change')
    #b_patch = mpatches.Patch(color ='blue', label='one-sigma shift')
    #o_patch = mpatches.Patch(color='orange', label= 'one-two-sigma shift')
    #cr_patch = mpatches.Patch(color='crimson', label= '>two-sigma shift')
    #az.legend(handles = [b_patch, o_patch, cr_patch], loc = 'upper right', fancybox=True, framealpha=1, shadow=True, borderpad=1)
    #seaborn.despine(fig, offset = 20)
    mat.show()

    figu,axk = mat.subplots(1, figsize = (18,6))
    axk.plot(detecting_on, linewidth = 3, color = 'indigo')
    #axk.plot(np.gradient(detecting_on.T), linewidth = 1, color = 'orange', alpha = 0.5)
    axk.scatter(range(0, len(detecting_on)), detecting_on, color = c_palette, s = 40)
    axk.set_xticks(range(0,283,6))
    axk.set_xticklabels(Around_the_clock, rotation = 45, va = 'top', ha = 'right')
    axk.set_xlabel('Time')
    axk.set_ylabel('Volatility (u.a.)')
    axk.legend(handles = [r_patch, g_patch], loc = 'upper right', fancybox=True, framealpha=1, shadow=True, borderpad=1)
    mat.show()

#############################################################################################

#============================================#
#Investigation of anomalous time windows
#============================================#

DATASET = LM.Plug(os.getcwd())
print(DATASET)
DATASET = LM.Time_Analysis(DATASET)

DATASET = LM.Janitor(DATASET)
DATASET = DATASET.tolist()
tfidf_datamatrix, bag_o_words, tFile = LM.TFIDF(DATASET)

'''il formato dell'output è un file di testo a due colonne con:
            Prima colonna   :  Keyword
            Seconda colonna :  TFIDF Score
'''

#outputText = {'keyword': np.array(bag_o_words), 'Score': np.ravel(docColumn)}
#tFile = pd.DataFrame(outputText)
tFile.to_csv(r'outputTFIDFfe.txt', header=['Word'], index = bag_o_words, sep=';', mode='w')
#print(outputText)
print(bag_o_words)
