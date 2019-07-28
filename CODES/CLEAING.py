# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 14:04:59 2019

@author: aatis
"""

#importing libraries
import pandas as pd
import numpy as np
import os


#path
z = r'C:\Users\aatis\Desktop\Applied Analytics\Summer\Project\loan.csv'
x = r'C:\Users\aatis\Desktop\Applied Analytics\Summer\Project\loan.update.csv'


#reading files
old_data = pd.read_csv(z,error_bad_lines = False,low_memory = False)

dtype= old_data.dtypes

new_data = pd.read_csv(x,error_bad_lines = False,low_memory=False)

dtype= new_data.dtypes


#slicing the dataset 
a = new_data.loc[:200000]
frames = [old_data,a]

# Merging the two files
df = pd.concat(frames, sort = False)


################################################################DATA CLEANING ##############################################

#Removing rows for ID 
type(df)
df = df.dropna(axis=0, subset=['id'])
#Removing rows for grade = G and loan status = "Does not Meet the credit policy. Status: Fully paid & charged off"

df = df[(df['grade'] != "G") & (df['loan_status'] != 'Does not meet the credit policy. Status:Charged Off') & (df['loan_status'] != 'Does not meet the credit policy. Status:Fully Paid')]


# Percentage of null values
Null_percentage = df.isnull().sum()/len(df)*100



#selecting the columns which are less than 10% null value

df_clean= df.loc[:, df.isnull().mean()<0.1]
df_clean = df_clean[(df_clean['loan_status'] != "Issued")]

#Remove people earning more than 250,000$

df_clean = df_clean[(df_clean['annual_inc'] < 250000)&(df_clean['annual_inc'] != 0)]

df_head1 = df_clean.head()

#Revol high credit dis

df_clean = df_clean[(df_clean['total_rev_hi_lim'] < 250000)]

#Total Cur Bal under 600k only

df_clean= df_clean[df_clean['tot_cur_bal'] < 600000]

#Remove Revol balance 

df_clean= df_clean[df_clean['revol_bal'] < 85000]

#dffun['delinq_2yrs'] less than 3

df_clean = df_clean[df_clean['delinq_2yrs'] <3]

#KEEP ONLY RENT,MORTGAG,OWN

df_clean = df_clean[(df_clean['home_ownership']=='RENT')|(df_clean['home_ownership']=='MORTGAGE')|(df_clean['home_ownership'] == 'OWN')]

#Revol_Uti

df_clean = df_clean[df_clean['revol_util'] < 100]

# Open_accounts drop over 30.35

df_clean = df_clean[df_clean['open_acc'] < 30.35]


#Making grade A Late_Fee Average for loan_status = "Fully paid"

fullyPaid_A = df_clean[(df_clean['grade'] == 'A') & (df_clean['loan_status'] == "Fully Paid")]
fullyPaid_A_LateFee = fullyPaid_A[(fullyPaid_A['total_rec_late_fee'] > 0)]
fullPaid_A_Mean = fullyPaid_A_LateFee['total_rec_late_fee'].mean()


#Making grade B Late_Fee Average for loan_status = "Fully paid"


fullyPaid_B = df_clean[(df_clean['grade'] == 'B') & (df_clean['loan_status'] == "Fully Paid")]
fullyPaid_B_LateFee = fullyPaid_B[(fullyPaid_B['total_rec_late_fee'] > 0)]
fullPaid_B_Mean = fullyPaid_B_LateFee['total_rec_late_fee'].mean()



#Making grade C Late_Fee Average for loan_status = "Fully paid"


fullyPaid_C = df_clean[(df_clean['grade'] == 'C') & (df_clean['loan_status'] == "Fully Paid")]
fullyPaid_C_LateFee = fullyPaid_C[(fullyPaid_C['total_rec_late_fee'] > 0)]
fullPaid_C_Mean = fullyPaid_C_LateFee['total_rec_late_fee'].mean()



#Making grade D Late_Fee Average for loan_status = "Fully paid"


fullyPaid_D = df_clean[(df_clean['grade'] == 'D') & (df_clean['loan_status'] == "Fully Paid")]
fullyPaid_D_LateFee = fullyPaid_D[(fullyPaid_D['total_rec_late_fee'] > 0)]
fullPaid_D_Mean = fullyPaid_D_LateFee['total_rec_late_fee'].mean()

#Making grade E Late_Fee Average for loan_status = "Fully paid"

fullyPaid_E= df_clean[(df_clean['grade'] == 'E') & (df_clean['loan_status'] == "Fully Paid")]
fullyPaid_E_LateFee = fullyPaid_E[(fullyPaid_E['total_rec_late_fee'] > 0)]
fullPaid_E_Mean = fullyPaid_E_LateFee['total_rec_late_fee'].mean()


#Making grade F Late_Fee Average for loan_status = "Fully paid"

fullyPaid_F = df_clean[(df_clean['grade'] == 'F') & (df_clean['loan_status'] == "Fully Paid")]
fullyPaid_F_LateFee = fullyPaid_F[(fullyPaid_F['total_rec_late_fee'] > 0)]
fullPaid_F_Mean = fullyPaid_F_LateFee['total_rec_late_fee'].mean()


#Making Grade A late_fee average for loan_status active
Active_A = df_clean[(df_clean['grade'] == 'A') & (df_clean['loan_status'] != "Fully Paid") & (df_clean['loan_status'] !="Charged Off") & (df_clean['loan_status'] != "Default")]
Active_A_LateFee = Active_A[(Active_A['total_rec_late_fee'] > 0)]
Active_A_Mean = Active_A_LateFee['total_rec_late_fee'].mean()

#Making Grade B late_fee average for loan_status active
Active_B = df_clean[(df_clean['grade'] == 'B') & (df_clean['loan_status'] != "Fully Paid") & (df_clean['loan_status'] !="Charged Off") & (df_clean['loan_status'] != "Default")]
Active_B_LateFee = Active_B[(Active_B['total_rec_late_fee'] > 0)]
Active_B_Mean = Active_B_LateFee['total_rec_late_fee'].mean()


#Making Grade C late_fee average for loan_status active
Active_C = df_clean[(df_clean['grade'] == 'C') & (df_clean['loan_status'] != "Fully Paid") & (df_clean['loan_status'] !="Charged Off") & (df_clean['loan_status'] != "Default")]
Active_C_LateFee = Active_C[(Active_C['total_rec_late_fee'] > 0)]
Active_C_Mean = Active_C_LateFee['total_rec_late_fee'].mean()

#Making Grade D late_fee average for loan_status active
Active_D = df_clean[(df_clean['grade'] == 'D') & (df_clean['loan_status'] != "Fully Paid") & (df_clean['loan_status'] !="Charged Off") & (df_clean['loan_status'] != "Default")]
Active_D_LateFee = Active_D[(Active_D['total_rec_late_fee'] > 0)]
Active_D_Mean = Active_D_LateFee['total_rec_late_fee'].mean()

#Making Grade E late_fee average for loan_status active
Active_E = df_clean[(df_clean['grade'] == 'E') & (df_clean['loan_status'] != "Fully Paid") & (df_clean['loan_status'] !="Charged Off") & (df_clean['loan_status'] != "Default")]
Active_E_LateFee = Active_E[(Active_E['total_rec_late_fee'] > 0)]
Active_E_Mean = Active_E_LateFee['total_rec_late_fee'].mean()

#Making Grade E late_fee average for loan_status active
Active_F = df_clean[(df_clean['grade'] == 'F') & (df_clean['loan_status'] != "Fully Paid") & (df_clean['loan_status'] !="Charged Off") & (df_clean['loan_status'] != "Default")]
Active_F_LateFee = Active_F[(Active_F['total_rec_late_fee'] > 0)]
Active_F_Mean = Active_F_LateFee['total_rec_late_fee'].mean()


#Function for making new column Active_loan_average_late_fee
def change_active (c):
    if (c['grade'] == 'A') & (c['loan_status'] != "Fully Paid" ) & (c['loan_status'] != 'Charged Off') & (c['loan_status'] != "Default"):
        return Active_A_Mean
    elif (c['grade'] == 'B') & (c['loan_status'] != "Fully Paid" ) & (c['loan_status'] != 'Charged Off') & (c['loan_status'] != "Default"):
        return Active_B_Mean
    elif (c['grade'] == 'C') & (c['loan_status'] != "Fully Paid" ) & (c['loan_status'] != 'Charged Off') & (c['loan_status'] != "Default"):
        return Active_C_Mean
    elif (c['grade'] == 'D') & (c['loan_status'] != "Fully Paid" ) & (c['loan_status'] != 'Charged Off') & (c['loan_status'] != "Default"):
        return Active_D_Mean
    elif (c['grade'] == 'E') & (c['loan_status'] != "Fully Paid" ) & (c['loan_status'] != 'Charged Off') & (c['loan_status'] != "Default"):
        return Active_E_Mean
    elif (c['grade'] == 'F') & (c['loan_status'] != "Fully Paid" ) & (c['loan_status'] != 'Charged Off') & (c['loan_status'] != "Default"):
        return Active_F_Mean
    else:
        return 0    

# Applying the function
df_clean['Active_loan_Average_late_fee']=df_clean.apply(change_active, axis = 1)
df_clean['Active_loan_Average_late_fee'].astype(float)

#Function for making new column Fully_Paid_Loan_Average_late_fee
def change_fullypaid (x):
    if (x['grade'] == 'A') & (x['loan_status'] == 'Fully Paid'):
        return fullPaid_A_Mean
    elif (x['grade'] == 'B') & (x['loan_status'] == 'Fully Paid'):
        return fullPaid_B_Mean
    elif (x['grade'] == 'C') & (x['loan_status'] == 'Fully Paid'):
        return fullPaid_C_Mean
    elif (x['grade'] == 'D') & (x['loan_status'] == 'Fully Paid'):
        return fullPaid_D_Mean
    elif (x['grade'] == 'E') & (x['loan_status'] == 'Fully Paid'):
        return fullPaid_E_Mean
    elif (x['grade'] == 'F') & (x['loan_status'] == 'Fully Paid'):
        return fullPaid_F_Mean
    else:
        return 0 

# Applying the function
df_clean['FullyPaid_loan_Average_late_fee'] = df_clean.apply(change_fullypaid, axis =1)
df_clean['FullyPaid_loan_Average_late_fee'].astype(float)


#Creating late Pyament modifier

grade_A_pm = df_clean[(df_clean['grade']=='A')]
lpm_A = (sum(grade_A_pm['total_rec_late_fee'])/ sum(grade_A_pm['funded_amnt']))*100

grade_B_pm = df_clean[(df_clean['grade']=='B')]
lpm_B = (sum(grade_B_pm['total_rec_late_fee'])/ sum(grade_B_pm['funded_amnt']))*100

grade_C_pm = df_clean[(df_clean['grade']=='C')]
lpm_C = (sum(grade_C_pm['total_rec_late_fee'])/ sum(grade_C_pm['funded_amnt']))*100

grade_D_pm = df_clean[(df_clean['grade']=='D')]
lpm_D = (sum(grade_D_pm['total_rec_late_fee'])/ sum(grade_D_pm['funded_amnt']))*100

grade_E_pm = df_clean[(df_clean['grade']=='E')]
lpm_E = (sum(grade_E_pm['total_rec_late_fee'])/ sum(grade_E_pm['funded_amnt']))*100

grade_F_pm = df_clean[(df_clean['grade']=='F')]
lpm_F = (sum(grade_F_pm['total_rec_late_fee'])/ sum(grade_F_pm['funded_amnt']))*100


#Function for making new column Late Payment Modifier
def paymentModifier (i):
    if i['grade'] == 'A':
        return lpm_A
    elif i['grade'] == 'B':
        return lpm_B
    elif i['grade'] == 'C':
        return lpm_C
    elif i['grade'] == 'D':
        return lpm_D
    elif i['grade'] == 'E':
        return lpm_E
    elif i['grade'] == 'F':
        return lpm_F


#Applying the function
df_clean['Late_Payment_Modifier'] = df_clean.apply(paymentModifier,axis = 1)
df_clean['Late_Payment_Modifier'].astype(float)

# Creating issue_date column from issue
dffun = pd.DataFrame(df_clean)

dffun["issue_date"] = pd.to_datetime(dffun["issue_d"]).dt.date

#Regular Expression for Term
dffun["num_month"] = dffun.term.str.extract('(\d+)').astype(str)
dffun['num_month'] = dffun.num_month.astype(int)


#Month
def mth(xa):
    if xa['num_month'] == 36:
        val = 1
    else:
        val=  0
    return val

# Binary for Month
dffun['num_month'] = dffun.apply(mth,axis = 1)

# To find how much money lending club funded
dffun['lc_fun_amt'] = dffun["funded_amnt"] - dffun["funded_amnt_inv"]

def f(row):
    if row['lc_fun_amt'] > 0:
        val = 1
    else:
        val = 0
    return val

# Binary for the lending funded
dffun["lc_fun"] = dffun.apply(f,axis=1)


# Employee length
dffun['emp_length'] = dffun['emp_length'].replace(np.nan, '10+ Years', regex=True)

dffun["yr_emp_int"] = dffun.emp_length.str.extract('(\d+)').astype(int)


def f(row):
    if row['yr_emp_int'] > 5:
        val = 1
    else:
        val = 0
    return val

dffun['emp_more_fiv'] = dffun.apply(f,axis=1)


#State
states = {
        'AK': 'O',
        'AL': 'S',
        'AR': 'S',
        'AS': 'O',
        'AZ': 'W',
        'CA': 'W',
        'CO': 'W',
        'CT': 'N',
        'DC': 'N',
        'DE': 'N',
        'FL': 'S',
        'GA': 'S',
        'GU': 'O',
        'HI': 'O',
        'IA': 'M',
        'ID': 'W',
        'IL': 'M',
        'IN': 'M',
        'KS': 'M',
        'KY': 'S',
        'LA': 'S',
        'MA': 'N',
        'MD': 'N',
        'ME': 'N',
        'MI': 'W',
        'MN': 'M',
        'MO': 'M',
        'MP': 'O',
        'MS': 'S',
        'MT': 'W',
        'NA': 'O',
        'NC': 'S',
        'ND': 'M',
        'NE': 'W',
        'NH': 'N',
        'NJ': 'N',
        'NM': 'W',
        'NV': 'W',
        'NY': 'N',
        'OH': 'M',
        'OK': 'S',
        'OR': 'W',
        'PA': 'N',
        'PR': 'O',
        'RI': 'N',
        'SC': 'S',
        'SD': 'M',
        'TN': 'S',
        'TX': 'S',
        'UT': 'W',
        'VA': 'S',
        'VI': 'O',
        'VT': 'N',
        'WA': 'W',
        'WI': 'M',
        'WV': 'S',
        'WY': 'W'
}

dffun['addr_state']= dffun['addr_state'].map(states)



#Pay to income


dffun['per_month'] = dffun['annual_inc']/12
dffun['pay_to_income'] = dffun['installment']/dffun['per_month']        


# Total amount paid for active loan
def total (c):
    if (c['loan_status'] != "Fully Paid" ) & (c['loan_status'] != 'Charged Off') & (c['loan_status'] != "Default"):
        return c['installment'] * c['num_month']
    else:
        return 0


dffun["total_payment_asu"] = dffun.apply(total, axis =1)


#late_pay for active loan
def late (c):
    if c['total_payment_asu'] != 0:
        return c['funded_amnt'] * c['Late_Payment_Modifier']
    else:
        return 0

dffun['late_pay'] = dffun.apply(late, axis = 1)

#total_final _pay 
def total_final (c):
    if c['late_pay'] != 0:
        return c['total_payment_asu'] + c['late_pay']
    else: 
        return 0
    

dffun['total_pay_f'] = dffun.apply(total_final, axis = 1)

# return_per

def return_per(c):
    if c['total_pay_f'] != 0:
        return c['total_pay_f'] / c['funded_amnt']
    else:
        return c['total_pymnt'] / c['funded_amnt']

dffun['return_per'] = dffun.apply(return_per, axis = 1)

################################################################Target Variable 15%#########################################
def f(row):
    if row['return_per'] > 1.15:
        val = 1
    else:
        val = 0
    return val

dffun["over_15"] = dffun.apply(f,axis=1)









