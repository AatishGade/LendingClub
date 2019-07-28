# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 14:37:49 2019

@author: aatis
"""


#################################################################### Sumarry Statsitics ################################

dffun.describe()

################################################################### Skewness ##############################################


dfsk=pd.DataFrame()

dfsk = pd.DataFrame(dffun.skew())
dfsk.columns = ['Skew']
dfsk["col"] = dfsk.index

dfsk=dfsk.reset_index()

dfsk=dfsk.drop(['index'],axis=1)

dfhigsk = dfsk.loc[dfsk['Skew'] > 3]

dflowsk = dfsk.loc[dfsk['Skew'] < -3]

dflist = dfhigsk['col'].tolist()

df_skew = dffun[dflist]

dflg = np.log1p(df_skew)

dflguse =pd.DataFrame()

dflguse = pd.DataFrame(dflg.skew())

dflguse.columns = ['Skew']

dflguse["col"] = dflguse.index


############################################################# SKEWNESS PLOT ###############################################



###################################################################After Log###############################################
vissk = dffun_[['loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'int_rate', 'installment','annual_inc', 'dti','inq_last_6mths','open_acc','pub_rec','revol_bal','revol_util','total_acc','tot_cur_bal','total_rev_hi_lim','return_per','total_pay_f','log_pub_rec']]
split = 18

data=vissk.iloc[:,:split]

cols=data.columns

import seaborn as sns
import matplotlib.pyplot as plt

n_cols = 2
n_rows = 6

for i in range(n_rows):
    fg, ax = plt.subplots(nrows=1,ncols=n_cols, figsize=(12,8))
    for j in range(n_cols):
        sns.violinplot(y=cols[i*n_cols+j], data=data,ax=ax[j])
        
#################################################################Pior before log###############################################


data=dffun_[['pub_rec']]

size=1

cols=data.columns

n_cols = 2
n_rows = 6

for i in range(n_rows):
    fg, ax = plt.subplots(nrows=1,ncols=n_cols, figsize=(12,8))
    for j in range(n_cols):
        sns.violinplot(y=cols[i*n_cols+j], data=data,ax=ax[j])


##############################################log
        
data=dffun_[['log_pub_rec','pub_rec']]

size=1

cols=data.columns

n_cols = 2
n_rows = 6

for i in range(n_rows):
    fg, ax = plt.subplots(nrows=1,ncols=n_cols, figsize=(12,8))
    for j in range(n_cols):
        sns.violinplot(y=cols[i*n_cols+j], data=data,ax=ax[j])
        

dflguse=dflguse.reset_index()

dflguse = dflguse.drop(['index'],axis=1)

dflgkeep = dflguse[(dflguse['Skew'] < 3) & (dflguse['Skew'] > -3)]

df_log_cols = dflgkeep['col'].tolist()

df_log_list = dflg[df_log_cols]

dflogvar = df_log_list.add_prefix('log_')

dflgkeep['col'] = 'log_' + dflgkeep['col']

dffun_ = pd.concat([dffun,dflogvar],axis=1, sort = False)

df_last = dffun_.head()
frm = [dflgkeep,dflowsk]

df_skew_final = pd.concat(frm, sort = False)