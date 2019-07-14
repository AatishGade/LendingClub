import pandas as pd
import os
import numpy as np
z = r'C:\Users\aatis\Desktop\Applied Analytics\Summer\Project\loan.csv'
df = pd.read_csv(os.path.join(os.path.dirname(__file__),z))