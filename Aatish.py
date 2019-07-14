import pandas as pd
import os
import numpy as np
from string import digits
# Selectin path & reading the file
z = r'C:\Users\aatis\Desktop\Applied Analytics\Summer\Project\loan.csv'
x = r'C:\Users\aatis\Desktop\Applied Analytics\Summer\Project\loan.update.csv'

old_data= pd.read_csv(z)
new_data = pd.read_csv(x)

# Making seprate Df for grade G
G = new_data['grade'] == 'G'
grade_g_2M = new_data[G]

G1 = old_data['grade'] == 'G'
grade_g_88T = old_data[G1]

p = list(grade_g_88T['loan_status'].values)

# Grade_G with loan status Does not meet the credit policy. Status: Charged off & Fully Paid
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

# Percentage of null values
Null_percentage = df.isnull().sum()/len(df)*100

#selecting the columns which are less than 10% null value

df_clean= df.loc[:, df.isnull().mean()<0.1]
col_name = list(df_clean.columns.values)

#Making grade A Late_Fee Average for loan_status = "Fully paid"

A = df_clean['grade'] == 'A'
grade_A = df_clean[A] 
FP_A = grade_A['loan_status'] == 'Fully Paid'
A_fully_Paid = grade_A[FP_A]
LF_A= A_fully_Paid['total_rec_late_fee'] > 0
A_FullyPaid_LateFee = A_fully_Paid[LF_A]
A_Mean = A_FullyPaid_LateFee['total_rec_late_fee'].mean()


#Making grade B Late_Fee Average for loan_status = "Fully paid"


B = df_clean['grade'] == 'B'
grade_B = df_clean[B] 
FP_B = grade_B['loan_status'] == 'Fully Paid'
B_fully_Paid = grade_B[FP_B]
LF_B= B_fully_Paid['total_rec_late_fee'] > 0
B_FullyPaid_LateFee = B_fully_Paid[LF_B]
B_Mean = B_FullyPaid_LateFee['total_rec_late_fee'].mean()



#Making grade C Late_Fee Average for loan_status = "Fully paid"

C = df_clean['grade'] == 'C'
grade_C = df_clean[C] 
FP_C = grade_C['loan_status'] == 'Fully Paid'
C_fully_Paid = grade_C[FP_C]
LF_C= C_fully_Paid['total_rec_late_fee'] > 0
C_FullyPaid_LateFee = C_fully_Paid[LF_C]
C_Mean = C_FullyPaid_LateFee['total_rec_late_fee'].mean()


#Making grade D Late_Fee Average for loan_status = "Fully paid"


D = df_clean['grade'] == 'D'
grade_D = df_clean[D] 
FP_D = grade_D['loan_status'] == 'Fully Paid'
D_fully_Paid = grade_D[FP_D]
LF_D= D_fully_Paid['total_rec_late_fee'] > 0
D_FullyPaid_LateFee = D_fully_Paid[LF_D]
D_Mean = D_FullyPaid_LateFee['total_rec_late_fee'].mean()

#Making grade E Late_Fee Average for loan_status = "Fully paid"

E = df_clean['grade'] == 'E'
grade_E = df_clean[E] 
FP_E = grade_E['loan_status'] == 'Fully Paid'
E_fully_Paid = grade_E[FP_E]
LF_E= E_fully_Paid['total_rec_late_fee'] > 0
E_FullyPaid_LateFee = E_fully_Paid[LF_E]
E_Mean =E_FullyPaid_LateFee['total_rec_late_fee'].mean()


#Making grade F Late_Fee Average for loan_status = "Fully paid"


F = df_clean['grade'] == 'F'
grade_F = df_clean[F] 
FP_F = grade_F['loan_status'] == 'Fully Paid'
F_fully_Paid = grade_F[FP_F]
LF_F= F_fully_Paid['total_rec_late_fee'] > 0
F_FullyPaid_LateFee = F_fully_Paid[LF_F]
F_Mean=F_FullyPaid_LateFee['total_rec_late_fee'].mean()

# Making a seprate Dict for Grade Late_fee average for loan_status = "Fully paid"

Mean =  {'A': A_Mean, 'B': B_Mean, 'C': C_Mean, 'D': D_Mean , 'E': E_Mean , 'F': F_Mean}


zz = df_clean['loan_status']