import os
import glob
import time
import numpy as np
import pandas as pd
import regex as re
import matplotlib.pyplot as mat
import matplotlib.cm as cm
import seaborn



from collections import Counter
from functools import cmp_to_key

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
from collections import Counter
#from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
#from sklearn.manifold import TSNE
#from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.preprocessing import MinMaxScaler


seaborn.set_style('whitegrid')
seaborn.set_style('ticks')
seaborn.set_context('paper', font_scale=0.7)
csfont = {'fontname':'Monospace'}

#=================================================================================================================#
# TECHNICAL ASPECTS:
# Code was compiled on a HP Spectre late 2018 (Intel Core i5-8265U, 8 Gb 2133 MHZ Ram) using the WSL and pure Win10
# and crosstested on an Asus K501UX late 2015 (Intel Core i7-6500U, 12 Gb Ram) pure Ubuntu 19.04

# WSL STANDARDS (I/O operations slowed down by the fake-kernel safety pipeline):
# If compiled under a "turboclock regime" the execution requires from 15 to 25 minutes (without TSNE plot)
# If compiled under "standardclock" regime, the elapsed exec. time approximately doubles.

# UBUNTU STANDARDS (and ideally any UNIX-like system)
# Basically the execution requires the turboclock regime time expense.

# WINDOWS STANDARDS HAVE NOT (as of 24/07/2019) BEEN CHECKED

# The entire routine is (right now) CPU-intensive.
#=================================================================================================================#


def Plug(INPUT_PATH):
    '''
    file managing function.
    Takes care of folder and logfile management. Apart from the creation of new folders, which supposes
    the "unix bracket", every other path variable is automatically detected and set up.
    Does some basic data cleansing, including splitting the datetime column into two more intuitive columns: calendar_date + wallclock_date
    for easing the analysis.

    ARGUMENTS:
    INPUT_PATH: Path to the working directory.

    RETURNS:
    DATA: the loaded file converted to the pandas.dataframe type.
    FILENAME: obvious

    UP-TO-DO:
    - right now, function is ideally fully deployed.
    '''
    time_format = '%I:%M:%S %p'
    #if os.path.exists(INPUT_PATH + '/output/'):
        #print(str(os.getcwd()) + '/output/' + 'already existent: moving on')
    #else:
        #OUTPUT_FOLDER_PATH = os.getcwd() + '/output/'

    print('AVAILABLE FILES in' + INPUT_PATH)
    # filetype is just an example MUST BE CHANGED if required.
    print(glob.glob('*.zip'))

    bool_coin = False
    filename = input('insert file name:')

    while (bool_coin != True):
        if (filename in glob.glob('*.zip')):
            bool_coin = True
            INPUT_FILE = INPUT_PATH + '/' + filename
            print('Currently loading:', INPUT_FILE)
        else:
            filename = input("wrong filename, re-type:" + ' ')

    tic = time.time()
    # specify compression only if a zip file is going to be opened.
    data = pd.read_csv(INPUT_FILE, sep=',', compression='zip')
    data.dropna(inplace = True)
    data.datetime = pd.to_datetime(data.datetime)
    data.datetime = data['datetime'].dt.round('5min')
    #print(data.datetime) #only a debugging call
    data.datetime = data.datetime.dt.strftime('%Y-%m-%d %r')
    #print(data.datetime) #again, debugging call
    new_column = data["datetime"].str.split(" ", n=1, expand=True)
    data["calendar_date"] = new_column[0]
    data["wallclock_date"] = new_column[1]

    data.drop(columns=["datetime"], inplace=True)
    #fdata = pd.DataFrame({'Log': data['message'].values}, index = [i for i in data.wallclock_date])
    fdata = pd.DataFrame(list(zip(data['message'].values, [i for i in data.wallclock_date])), columns = ['mess', 'time'])

    toc = time.time()
    print('loaded' + ' ' + str(len(data.index)) + ' ' +
          'lines successfully in' + ' ' + str(toc-tic) + 's')



    return fdata

##################################################################


def Janitor(log_item):
    '''
    Log-Cleaning function. cleans whatever is not necessary, via regex substitutions.

    ARGUMENTS:
    log_item: the log file you want to analyze.\

    RETURNS:
    log_item: cleansed and usable log file.

    UP-TO-DO:
    -
    '''

    print("cleaning the log file...")

    USER = '<.*>'
    REQUEST = r'((\w|\d){8})-((\w|\d){4})-((\w|\d){4})-((\w|\d){4})-((\w|\d){12})'
    TOKEN = r'\[token:.*\]'
    SURL = r'srm:.+?(?=]| |$)'
    PATH = r'/.+?(?=]| |$)'
    IPV4 = r'(ffff:(\d){1,3}.(\d){1,3}.(\d){1,3}.(\d){1,3})|(((\S){1,4}:){3,4}:(\S){1,4}:(\S){1,4})|(((\S){1,4}:){4,}(\S){1,4})'
    MAIL = r'(\S)+\@\S+\.\S+'
    SANUM = r"(?<=')\d+(?=')"
    regexes = np.array([USER, REQUEST, TOKEN, SURL, PATH, IPV4, MAIL, SANUM])
    str_replace = np.array(['USER', 'REQUEST', 'TOKEN',
                   'SURL', 'PATH', 'IPV4', 'MAIL', ''])

    tic = time.time()
    for i in range(0, 8, 1):
        log_item.replace(regexes[i], str_replace[i], regex=True, inplace=True)

    log_item = log_item.str.replace('\d+', '')
    log_item = log_item.str.lower()
    #log_item.replace(',','', inplace = True)
    toc = time.time()

    print("Regex filtering completed")
    print("Elapsed time:", str(toc-tic) + ' ' + 's')

    return log_item

####################################################################################


def TFIDF(messages_list):
    '''
    function that calculates the TF-IDF sparse matrix for word importance in document.

    ARGUMENTS:
    messages_list: requires the input file from which TFIDF will be extracted

    RETURNS:
    sparse_mat: the TFIDF sparse matrix
    keyword_list: selected word features purged from common english connectors and other uninteresting words.

    UP-TO-DO:
    -
    '''
    Vectorizer = TfidfVectorizer(
        max_df=0.7, max_features=50, ngram_range = (2,3), stop_words='english', use_idf = True)

    tic = time.time()
    sparse_mat = Vectorizer.fit_transform(messages_list)#.todense()
    tfidf_df = pd.DataFrame(sparse_mat[0].T.todense(), index = Vectorizer.get_feature_names(), columns = ["Word-Score"])
    tfidf_df = tfidf_df.sort_values('Word-Score', ascending = False)
    #sparse_mat.transpose()
    toc = time.time()

    print("Elapsed Time for TFIDF extraction:", str(toc-tic) + ' ' + 's')

    keyword_list = list(Vectorizer.get_feature_names())
    #print(keyword_list)

    common_words = ['and', 'are', 'be', 'by', 'for',
                    'from', 'ip', 'some', 'is', 'no', 'of', 'on', 'than', 'the',
                    'all', 'already', 'because', 'as', 'at', 'when', 'were', 'with', 'to', 'not', 'dn', 'user', 'useruser',
                    'update', 'true', 'ago', 'alias', 'availability', 'available', 'avoid', 'client', 'com', 'consider', 'created',
                    'does', 'equal', 'file', 'files', 'got', 'group', 'info', 'information', 'milliseconds', 'seconds', 'value', 'values',
                    'use', 'using', 'upadate', 'updated', 'tape', 'task', 'surl', 'summary', 'status', 'server', 'set', 'space', 'result', 'roll',
                    'rolling', 'chunk', 'chunks', 'requests', 'token' ]

    for key in common_words:
        if key in keyword_list:
            keyword_list.remove(key)
        elif key in str(range(90000)):
            keyword_list.remove
        else:
            continue


    print(tfidf_df)
    return sparse_mat, keyword_list, tfidf_df

####################################################################################

# just a support function i split apart from Time_Analysis to make code easier to maintain.
def compare_as_time(time_str1, time_str2):
    time_format = '%I:%M:%S %p'  # match hours, minutes and AM/PM
    # parse time strings to time objects
    time1 = time.strptime(time_str1, time_format)
    time2 = time.strptime(time_str2, time_format)

    # return comparison, sort expects -1, 1 or 0 to determine order
    if time1 < time2:
        return -1
    elif time1 > time2:
        return 1
    else:
        return 0


def Time_Analysis(DATASET):
    '''
    function that performs the bar plot of wallclock_time calls of storm service.
    depending on future, this may add some time-series actions on data.

    ARGUMENTS:
    DATASET: Prepared DataFrame
    START_TIC: Start of the An. Time window. Format required: '%I:%M:%S %p'
    END_TIC: End of the An. Time window. Format required: '%I:%M:%S %p'

    RETURNS:
    Selected messages in "raw" format. All words, all useless items.

    '''

    time_format = '%I:%M:%S %p'  # match hours, minutes and AM/PM

    #time_calls = [i for i in DATASET.wallclock_date]
    ##logs = DATASET.message.values
    #
    #unique_time_calls = list(set(time_calls))
    #unique_time_calls = sorted(
    #    unique_time_calls, key=cmp_to_key(compare_as_time))
    #print(unique_time_calls)

    #time_calls_count = Counter(time_calls)
    #time_occ = []

    start_tic = '05:00:00 AM'#'12:15:00 PM'
    end_tic = '06:00:00 AM'
    DATASET.time = pd.to_datetime(DATASET.time)
    mask = (DATASET.time > start_tic) & (DATASET.time <= end_tic)

    N_DATASET = DATASET.loc[mask]#[DATASET.loc[DATASET.wallclock_date>=start_tic]:DATASET.loc[DATASET.wallclock_date<=end_tic]]
    print(N_DATASET)
    #print(N_DATASET)                                                  # START WORKING FROM HERE
    #for hour in unique_time_calls:
        #time_occ.append(time_calls_count[hour])

    #zipped_data = zip(np.array(LIST_OF_TIME), np.array(LIST_OF_LOGS))
    #for h,c in zipped_data:
        #print ('hour: %s, occurr: %s' %(h,c))
    #occurrence_series = pd.Series(time_occ)

    #fig, ax = mat.subplots(1, figsize=(16, 8))
    #occurrence_series.rolling(6).mean().plot(
        #ax=ax, color='rebeccapurple', ls='dotted', lw=8)
    #occurrence_series.rolling(6).std().plot(ax=ax, color='darkmagenta', ls='dotted', lw=6, label = 'Rolling std.dev.')

    #ax.bar(unique_time_calls, time_occ, color='forestgreen', alpha=0.8)
    #
    #ax.set_xlabel('Wallclock_time')
    #ax.set_ylabel('Committed Logs')
    #ax.set_title(str(PLOT_TITLE))
    #mat.xticks([x for x in range(0, len(unique_time_calls), 6)], [unique_time_calls[y]
    #                                                              for y in range(0, len(unique_time_calls), 6)], rotation='vertical')
    #mat.legend()
    #seaborn.despine(fig)
    #mat.show()

    return N_DATASET.mess

    #####################################################################################

def Variance_Analysis(LIST_OF_FILES):
    '''
    Developer Function that shows, in an automated way, the Log Activity volatility around the clock (12 AM - 11:59 PM)

    ARGUMENTS:
    LIST_OF FILES: the result of glob() as string list.

    RETURNS:
    Volatility plot
    variations: pandas dataframe containing (one for each column) the volatility time-series

    UP-TO_DO:
    This function, as of 1/07/19, is considered sane.
    '''

    LIST_OF_MED = []
    variations = pd.DataFrame()
    #fig, ax = mat.subplots(1, figsize = (20,13))
    fig1, ax1 = mat.subplots(1, figsize = (20,13))
    time_format = '%I:%M:%S %p'  # match hours, minutes and AM/PM

    for filename in LIST_OF_FILES:
        fig, ax = mat.subplots(1, figsize = (20,13))
        #INPUT = os.getcwd() + '/' + str(filename)
        data = pd.read_csv(str(filename), sep=',', compression='zip')
        #print(data)
        data.dropna(inplace = True)
        print(str(filename) + ' ' + 'loaded')
        data.datetime = pd.to_datetime(data.datetime)
        data.datetime = data.datetime.dt.round('5min')
        # print(data.datetime) only a debugging call
        data.datetime = data.datetime.dt.strftime('%Y-%m-%d %r')
        # print(data.datetime) #again, debugging call
        new_column = data["datetime"].str.split(" ", n=1, expand=True)
        data["calendar_date"] = new_column[0]
        data["wallclock_date"] = new_column[1]
        data.drop(columns=["datetime"], inplace=True)


        time_calls = [i for i in data.wallclock_date]
        unique_time_calls = list(set(time_calls))
        unique_time_calls = sorted(
            unique_time_calls, key=cmp_to_key(compare_as_time))
        #print(unique_time_calls)
        time_calls_count = Counter(time_calls)
        time_occ = []
        for hour in unique_time_calls:
            time_occ.append(time_calls_count[hour])

        occurrence_series = pd.Series(time_occ)

        # this generates the log occurrence bar-plots

        figu, axu = mat.subplots(1, figsize = (15,6))
        occurrence_series.plot.bar(ax = axu, color = 'g', label = str(filename))
        mat.xticks(range(0, len(unique_time_calls),6))
        axu.set_xticklabels([unique_time_calls[y] for y in range(0, len(unique_time_calls),6)], rotation = 45, va = 'top', ha = 'right')
        axu.set_title('Log count on' + str(filename), **csfont)
        axu.set_xlabel('Time')
        axu.set_ylabel('Log count')
        seaborn.despine(figu)
        mat.show()
        #
        #figur, axxo = mat.subplots(1, figsize = (15,6))
        ##occurrence_series.hist(ax = axxo, color = 'g', label = str(filename))
        #seaborn.distplot(occurrence_series, kde = True, norm_hist = True, ax = axxo, color = 'g')
        #seaborn.despine(figur, offset = 20)
        #axxo.set_xlabel('Log Counts')
        #axxo.set_title('DistPlot of' + ' ' + str(filename))
        #mat.show()

        # this generates the volatility plots

        med_list = occurrence_series.rolling(6).std()
        variations[str(filename)] = med_list
        mat.xticks(range(0, len(unique_time_calls), 6))
        occurrence_series.rolling(6).std().plot(ax = ax, alpha = 0.9, lw = 2, label = str(filename))
        ax.set_xticklabels([unique_time_calls[y] for y in range(0, len(unique_time_calls), 6)], rotation = 45)
        ax.set_title('Volatility plot of' + str(filename), **csfont)
        ax.set_xlabel('Time')
        ax.set_ylabel('Rolling StD')
        leg = mat.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1)
        leg_lines = leg.get_lines()
        leg_texts = leg.get_texts()
        #bulk-set the properties of all lines and texts
        mat.setp(leg_lines, linewidth=4)
        seaborn.despine(fig)
        #mat.show()

        #LIST_OF_MED.append(med_list.mean())
    red_square = dict(markerfacecolor='r', marker='s')
    variations.boxplot(grid = False, ax = ax1, flierprops = red_square)
    #mat.show()

    #ax1.set_xlabel('Log_Files')
    #ax1.set_ylabel('Rolling StD of log activity')
    ax1.set_xticklabels(LIST_OF_FILES, fontsize = 8, rotation = 45, va = 'top', ha = 'right')
    #roll_var = pd.Series(LIST_OF_ROLLVARIANCES)

    #mat.axhline(y=np.mean(LIST_OF_MED), color='k', linestyle='--')
    #mat.axhline(y=np.mean(LIST_OF_MED) + 3*np.std(LIST_OF_MED), color = 'k',  lw = 2, linestyle = 'dashdot', alpha = 0.8)
    #mat.axhline(y=np.mean(LIST_OF_MED) - 3*np.std(LIST_OF_MED), color = 'k', lw = 2, linestyle = 'dashdot', alpha = 0.8)
    #mat.xticks([x for x in range(0, len(unique_time_calls), 6)])
    #ax.set_xticklabels([unique_time_calls[y] for y in range(0, len(unique_time_calls), 6)], rotation = 45)
    #ax.set_xlabel('Time')
    #ax.set_ylabel('Rolling StD')

    #leg = mat.legend(frameon = False)
    #leg_lines = leg.get_lines()
    #leg_texts = leg.get_texts()
    # bulk-set the properties of all lines and texts
    #mat.setp(leg_lines, linewidth=4)
    #seaborn.despine(fig1)
    mat.show()

    Timings = [unique_time_calls[y] for y in range(0, len(unique_time_calls), 6)]
    #variations['ClockTime'] = Timings
    variations.dropna(inplace = True)

    variations.to_csv('variation_db', sep = ',')

    return Timings
################################################################################
################################################################################
