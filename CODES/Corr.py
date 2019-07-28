# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 14:39:12 2019

@author: aatis
"""

############################################################## CORRR ##############################################

vissk = dffun_[['loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'int_rate',
                'installment','annual_inc', 'dti','inq_last_6mths','open_acc',
                'pub_rec','revol_bal','revol_util','total_acc','tot_cur_bal',
                'total_rev_hi_lim','return_per','total_pay_f','log_pub_rec']]

size = 15
data = vissk

data_corr = data.corr()

cols=data.columns

# Set the threshold to select only highly correlated attributes
threshold = 0.5

# List of pairs along with correlation above threshold
corr_list = []


#Search for the highly correlated pairs
for i in range(0,size): #for 'size' features
    for j in range(i+1,size): #avoid repetition
        if (data_corr.iloc[i,j] >= threshold and data_corr.iloc[i,j] < 1) or (data_corr.iloc[i,j] < 0 and data_corr.iloc[i,j] <= -threshold):
            corr_list.append([data_corr.iloc[i,j],i,j]) #store correlation and columns index
            
            #Sort to show higher ones first            
s_corr_list = sorted(corr_list,key=lambda x: -abs(x[0]))

#Print correlations and column names
for v,i,j in s_corr_list:
    print ("%s and %s = %.2f" % (cols[i],cols[j],v))
    

for v,i,j in s_corr_list:
    sns.pairplot(data, size=6, x_vars=cols[i],y_vars=cols[j])
plt.show()


