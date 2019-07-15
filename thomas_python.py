# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 16:51:20 2019

@author: ejt20
"""

#importing libraries 
import pandas as pd
import os
import numpy as np

path="D:\Final semester\Advanced Applied Analytics\Project"


os.chdir(path)

df = pd.read_csv('loan_old.csv',error_bad_lines = False,low_memory = False)

df = df.drop(columns=['id','member_id'])

#filtering grades
df_grades_A=df[(df['grade']=='A')]
ratio_A = sum (df_grades_A['total_rec_late_fee'])/ sum (df_grades_A['funded_amnt'])

df_grades_B=df[(df['grade']=='B')]
ratio_B = sum (df_grades_B['total_rec_late_fee'])/ sum (df_grades_B['funded_amnt'])

df_grades_C=df[(df['grade']=='C')]
ratio_C = sum (df_grades_C['total_rec_late_fee'])/ sum (df_grades_C['funded_amnt'])

df_grades_D=df[(df['grade']=='D')]
ratio_D = sum (df_grades_D['total_rec_late_fee'])/ sum (df_grades_D['funded_amnt'])

df_grades_E=df[(df['grade']=='E')]
ratio_E = sum (df_grades_E['total_rec_late_fee'])/ sum (df_grades_E['funded_amnt'])

df_grades_F=df[(df['grade']=='F')]
ratio_F = sum (df_grades_F['total_rec_late_fee'])/ sum (df_grades_F['funded_amnt'])

df_grades_G=df[(df['grade']=='G')]
ratio_G = sum (df_grades_G['total_rec_late_fee'])/ sum (df_grades_G['funded_amnt'])

#Adding a column named factors which is ratio of cooresponding grades
df.loc[df.grade == 'A', 'factors']= ratio_A
df.loc[df.grade == 'B', 'factors']= ratio_B
df.loc[df.grade == 'C', 'factors']= ratio_C
df.loc[df.grade == 'D', 'factors']= ratio_D
df.loc[df.grade == 'E', 'factors']= ratio_E
df.loc[df.grade == 'F', 'factors']= ratio_F
df.loc[df.grade == 'G', 'factors']= ratio_G

#USERDEFINED function.
'''
def ar(c):
    if c['grade'] == 'A':
        c['factorss']=ratio_A

  
df.apply(ar,axis=1)
d1  = df.head(100)
'''