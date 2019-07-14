# importing libraries 
import pandas as pd
import os
import numpy as np
# Selectin path & reading the file
z = r'C:\Users\aatis\Desktop\Applied Analytics\Summer\Project\loan.csv'
df = pd.read_csv(os.path.join(os.path.dirname(__file__),z))

#selecting the columns which are less than 10% null values
df_clean= df.loc[:, df.isnull().mean()<0.1]
print(list(df_clean.columns.values))