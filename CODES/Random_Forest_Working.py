# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 14:43:11 2019

@author: aatis
"""

######################################################### MODELING ####################################################



############################################ FINAL DATASET ################################
df_final = dffun_[['loan_amnt','dti','int_rate','annual_inc','inq_last_6mths','open_acc','log_pub_rec','revol_bal',
                   'revol_util','tot_cur_bal','lc_fun_amt','delinq_2yrs','total_pay_f','acc_now_delinq','out_prncp_inv','num_month',
                   'emp_more_fiv','lc_fun','grade','sub_grade', 'home_ownership','verification_status','purpose','addr_state','return_per','installment','over_15'
                   ]]

# X Variable

x = df_final[['loan_amnt','dti','annual_inc', 'inq_last_6mths','open_acc','log_pub_rec',#'revol_bal',
'revol_util',#'tot_cur_bal',
'delinq_2yrs','acc_now_delinq',
#'num_month',
 'pay_to_income','emp_more_fiv','lc_fun' , 'grade'
 
#'home_ownership','verification_status','addr_state'
]]


# Y VARIABLE

Y = df_final[['over_15']]



# DUMMY VARIABLE IF GRADE IS ADDED INTO X 

xc = pd.get_dummies(x, columns = ['grade'])


# Splitting the dataset

from sklearn.model_selection import train_test_split
x_train, x_test ,Y_train ,Y_test = train_test_split(xc,Y, test_size= 0.30, random_state = 0) 


#################################################### RANDOM FOREST OVERFITTING ################################################

# Fitting Random Forest classifier to the training set
from sklearn.ensemble import RandomForestClassifier
classifier_r = RandomForestClassifier(n_estimators = 100 ,  random_state =0, n_jobs = -1, oob_score = True, bootstrap = True )
classifier_r.fit(x_train,Y_train)

#SCORE

print('R^2 Training Score: {:.4f} \nOOB Score: {:.4f} \nR^2 Test Score: {:.4f}'.format(classifier_r.score(x_train, Y_train), 
                                                                                             classifier_r.oob_score_,
                                                                                             classifier_r.score(x_test, Y_test)))




################################################## RANDOM FOREST NO-OVERFITTING #############################################

# Fitting Random Forest classifier to the training set
from sklearn.ensemble import RandomForestClassifier
classifier_r1 = RandomForestClassifier(n_estimators = 300 , max_depth = 10,  random_state =0, n_jobs = -1, oob_score = True, bootstrap = True )
classifier_r1.fit(x_train,Y_train)


#SCORE
print('R^2 Training Score: {:.4f} \nOOB Score: {:.4f} \nR^2 Test Score: {:.4f}'.format(classifier_r1.score(x_train, Y_train), 
                                                                                             classifier_r1.oob_score_,
                                                                                             classifier_r1.score(x_test, Y_test)))



#SCORE FOR CROSS VALIDATION
from sklearn.model_selection import cross_val_score
score_RF = cross_val_score(classifier_r1, xc, Y, cv =5, scoring = 'accuracy') # CHANGE X variable accordingly
print("Accuracy: %0.2f (+/- %0.2f)" % (score_RF.mean(), score_RF.std() *2))
df_final_list = list(xc.columns)

# CONFUSION MATRIX PLOT

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
y_pred_cl = classifier_r1.predict(x_test)
mat = confusion_matrix(Y_test, y_pred_cl)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label');


######################################################## TREEE PLOT USE WITH MAX_DEPTH =3 for Better Picture #############

col_list = list(xc.columns) ############ CHANGE X VARIABLE ACCORDINGLY

# Import tools needed for visualization
from sklearn.tree import export_graphviz
import pydot
# Pull out one tree from the forest
tree = classifier_r1.estimators_[5]
# Import tools needed for visualization
from sklearn.tree import export_graphviz
import pydot
# Pull out one tree from the forest
tree = classifier_r1.estimators_[5]
# Export the image to a dot file
export_graphviz(tree, out_file = 'tree.dot', feature_names = col_list, rounded = True, precision = 1)
# Use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file('tree.dot')
# Write graph to a png file
graph.write_png('tree.png')


################################################################ IMPORTANCE OF VARIABLE #################################

# Get numerical feature importances
importances = list(classifier_r1.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(x, round(importance, 2)) for x, importance in zip(col_list, importances)]  ### CHANGE X VARIABLE ACCORDIGNLY
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];



################################################################# PLOTING VARIABLE IMPORTANCE #######################

# Import matplotlib for plotting 
import matplotlib.pyplot as plt
%matplotlib inline
# Set the style
plt.style.use('fivethirtyeight')
# list of x locations for plotting
x_values = list(range(len(importances)))
# Make a bar chart
plt.bar(x_values, importances, orientation = 'vertical')
# Tick labels for x axis
plt.xticks(x_values, df_final_list, rotation='vertical')
# Axis labels and title
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances');


################################################################ PREDICTING FOR 2 ROWS ###################################


from treeinterpreter import treeinterpreter as ti

selected_rows = [31, 85]
selected_df = x_train.iloc[selected_rows,:].values
prediction, bias, contributions = ti.predict(regressor_r, selected_df)

aa = np.array(Y_train)

for i in range(len(selected_rows)):
    print("Row", selected_rows[i])
    print("Prediction:", prediction[i][0])                   , print( 'Actual Value:', aa[selected_rows[i]])
    print("Bias (trainset mean)", bias[i])
    print("Feature contributions:")
    for c, feature in sorted(zip(contributions[i], 
                                 x_train.columns), 
                             key=lambda x: np.logical_not(x[0]).any()):
        print(feature, round(c[i],2))
    print("-"*20)








