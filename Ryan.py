import pandas as pd
import numpy as np
import os

path="/Users/tugtrog/Desktop/Python"

os.chdir(path)

df = pd.read_csv('loan_old.csv',error_bad_lines = False,low_memory = False)

df = df.drop(columns=['id','member_id'])

dtype= df.dtypes
#Drop rows with Issued and G
dfdrow = df[(df['loan_status'] != "Issued") & (df['grade'] != "G")]
#Drop Columns with more that 10% null values
dfdcolumn = dfdrow.loc[:, dfdrow.isnull().mean() < .10]
#Member_Id check to see if in DF more than once
#df12 = dfdcolumn.member_id.value_counts().reset_index(name="count").query("count > 1")["index"]
#Funded by lending club
dffun = pd.DataFrame(dfdcolumn)

dffun['lc_fun_amt'] = dfdcolumn["funded_amnt"] - dfdcolumn["funded_amnt_inv"]

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

dffun["return_per"] = dffun['funded_amnt'] / dffun['total_pymnt_inv']


def f(row):
    if row['return_per'] > 1.05:
        val = 1
    else:
        val = 0
    return val


dffun["over_five"] = dffun.apply(f,axis=1)

