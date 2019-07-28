# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 14:37:04 2019

@author: aatis
"""

################################################################# T-TEST ###################################################

#T-Test 
import scipy.stats as stats
homevsloan = dffun[(dffun['home_ownership'] == 'OWN')] 
own= homevsloan['loan_amnt']
homevsloan1 = dffun[(dffun['home_ownership'] == 'MORTGAGE')]
mortage =homevsloan1['loan_amnt']
homevsloan2 = dffun[(dffun['home_ownership'] == 'RENT')]

own = homevsloan.describe()
mortage = homevsloan1.describe()
rent = homevsloan2.describe()
len(homevsloan1)



# h0 = own> = mortage h1 = mortage > own


print(stats.ttest_ind(own,mortage, equal_var = False)) # p < .001 reject h0 , ppl who have mortage tend to get more loan amount

################################################################### ANOVA #################################################

# Libraries for Anova
from statsmodels.stats.multicomp import (pairwise_tukeyhsd,
                                         MultiComparison)
import statsmodels.api as sm
from statsmodels.formula.api import ols
import scipy.stats as stats
import seaborn as sns
aa = dffun[['grade','return_per']]
aa.columns = ['grade', 'value']


# BOX PLOT
sns.boxplot(x ="grade" ,y ="value" ,  data = aa, palette = "Set3")

# Anova two way 
model = ols('value ~ C(grade)', data = aa).fit()
anova_table = sm.stats.anova_lm(model, typ =1 )
anova_table

#Tukey HSD

m_comp = pairwise_tukeyhsd(endog = aa['value'], groups = aa['grade'] , alpha =0.05)
print(m_comp)

