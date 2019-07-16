import pandas as pd
import os
import numpy as np

# Selectin path & reading the file
z = r'C:\Users\aatis\Desktop\Applied Analytics\Summer\Project\loan.csv'
x = r'C:\Users\aatis\Desktop\Applied Analytics\Summer\Project\loan.update.csv'

old_data= pd.read_csv(z,low_memory= False)
new_data = pd.read_csv(x, low_memory = False)

# Making seprate Df for grade G
G = new_data['grade'] == 'G'
grade_g_2M = new_data[G]

G1 = old_data['grade'] == 'G'
grade_g_88T = old_data[G1]

p = list(grade_g_88T['loan_status'].values)

# Grade_G with loan status Does not meet the credit policy. Status: Charged off & Fully Paid
DN_fullpaid = grade_g_88T['loan_status'] == 'Does not meet the credit policy. Status:Fully Paid'
g_DN_fullpaid = grade_g_88T[DN_fullpaid]

DN_chargedoff = grade_g_88T['loan_status'] == 'Does not meet the credit policy. Status:Charged Off'
g_DN_chargedoff= grade_g_88T[DN_chargedoff]


doNotMeet = grade_g_88T['loan_status'].str.startswith('Does not')
grade_g_DonotMeet = grade_g_88T[doNotMeet]
# Making seprate Df for description
old_data['desc'].isnull().sum()/len(old_data)*100
old_data_desc = old_data.dropna(axis = 0, subset = ['desc'])

#slicing the dataset 
a = new_data.loc[:200000]
frames = [old_data,a]

# Merging the two files
df = pd.concat(frames, sort = False)


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
col_name = list(df_clean.columns.values)

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
# Making a seprate Dict for Grade Late_fee average for loan_status = "Fully paid"

full_Mean =  {'A': fullPaid_A_Mean, 'B': fullPaid_B_Mean, 'C': fullPaid_C_Mean, 'D': fullPaid_D_Mean , 'E': fullPaid_E_Mean , 'F': fullPaid_F_Mean}


zz = df_clean['loan_status']


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


dffun = pd.DataFrame(df_clean)

# Creating issue_date column from issue
dffun["issue_date"] = pd.to_datetime(dffun["issue_d"]).dt.date

dffun['term'] = dffun['term'].str.strip()


# function for months
def f(row):
    if row['term'] == "36 months":
        val = 36
    else: 
        val = 60
    return val

dffun["month"] = dffun.apply(f,axis=1)

# creating future month column
dffun["future_month"] = dffun.apply(lambda x: x['issue_date'] + pd.offsets.DateOffset(months=x['month']), 1)

dffun["future_month"] = pd.to_datetime(dffun["future_month"]).dt.date


#creating paid_date column

dffun["paid_date"] = dffun['future_month']

dffun.loc[df['loan_status']=='Fully Paid','paid_date'] = dffun['last_pymnt_d']

dffun["paid_date"]=pd.to_datetime(dffun["paid_date"]).dt.date

#dffun["paid_date"] = dffun.apply(lambda x: x['paid_date'] + pd.offsets.DateOffset(months=x['month']), 1)

hed=dffun.head()

counttest = dffun.loan_status.value_counts()


dffun['lc_fun_amt'] = dffun["funded_amnt"] - dffun["funded_amnt_inv"]

def f(row):
    if row['lc_fun_amt'] > 0:
        val = 1
    else:
        val = 0
    return val


dffun["lc_fun"] = dffun.apply(f,axis=1)

dfhead = dffun.head(100)

#Fully Funneded

dffun["full_fun_amt"] = dffun['loan_amnt'] - dffun['funded_amnt']

def f(row):
    if row['full_fun_amt'] > 0:
        val = 1
    else:
        val = 0
    return val


dffun["full_amt"] = dffun.apply(f,axis=1)

dfhead = dffun.head(100)

#Rate of Return

dffun["return_per"] = dffun['total_pymnt'] / dffun['funded_amnt'] 


def f(row):
    if row['return_per'] > 1.05:
        val = 1
    else:
        val = 0
    return val


dffun["over_five"] = dffun.apply(f,axis=1)

# Total amount paid

dffun["total_payment_asu"] = dffun['installment'] * dffun['month'] 

dffun["late_pay"] =dffun['funded_amnt'] * dffun['Late_Payment_Modifier']

dffun["total_pay_f"] =  dffun['total_payment_asu'] + dffun['late_pay']
dfhead = dffun.head(100)

###########################################

a = dffun.groupby('grade')['int_rate'].mean()

#############################################

dfss = dffun.describe()

dfsk=dffun.skew()
dflg = np.log1p(dffun["tot_coll_amt"])
dflg.skew()



writer = pd.ExcelWriter('df.xlsx', engine='xlsxwriter')
dfhead.to_excel(writer, sheet_name='A',index=False)
writer.save()
