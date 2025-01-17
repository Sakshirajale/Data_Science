# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 08:30:38 2024

@author: ACER
"""
#EDA
#DATA pre-Processing

import pandas as pd
#let us import dataset
df=pd.read_csv("C:/5-Data_prep/ethnic diversity.csv.xls")
df
#let us check data types of coloumns
df.dtypes
#salaries data type is float, let us convert into int
#df1=df.Salaries.astype(int)
df.Salaries=df.Salaries.astype(int)
df.Salaries
df.dtypes
#now the data type of Salaries is int
#Similarly age data type must be float
#Presently it is int
df.age=df.age.astype(float)
df.age
df.dtypes
########################
df_new=pd.read_csv("C:/5-Data_prep/education.csv.xls")
df_new
duplicate=df_new.duplicated()
#output of this function is single column
#if there is duplicate records output-true
#if there is no duplicate records output-false
#Series will be created
duplicate
sum(duplicate)
#output will be 0
##################3
#now let us import another dataset
df_new1=pd.read_csv("C:/5-Data_prep/mtcars.csv.xls")
duplicate1=df_new1.duplicated()
duplicate1
sum(duplicate1)
#There are 3 duplicate records
#row 17 is duplicate of row 2 like wise you can 3 duplicated records
#there is function called drop_duplicated()
#Which will drop
duplicate2=df_new2.duplicated()
duplicate2
sum(duplicate2)
###################################
#Outliers treatement
import pandas as pd
import seaborn as sns
df=pd.read_csv("C:/5-Data_prep/ethnic diversity.csv.xls")
df
#Now let us find outliers in Salaries
sns.boxplot(df.Salaries)
#There are outliers
#Let us check outliers in age column
sns.boxplot(df.age)
#There are no outliers
#Let us calculate IQR
IQR=df.Salaries.quantile(0.75)-df.Salaries.quantile(0.25)
IQR
#Have observed IQR in variable explorer
#no, because IQR is in capital letters
#treated as constant
#but if we will try as I,Iqr or iqr then it is
#Trimming
#########################3
##01/08/2024
##########################################################
iqr=df.Salaries.quantile(0.75)-df.Salaries.quantile(0.25)
#have observed IQR in variable explorer
#no,because IQR is in variable letter
#treated asconstant
iqr
#################################################
#but 
lower_limit=df.Salaries.quantile(0.25)-1.5*IQR
lower_limit
##-19446.9675

upper_limit=df.Salaries.quantile(0.75)+1.5*IQR
upper_limit
##93992.8125


##Trimming
import numpy as np
outliers_df=np.where(df.Salaries>upper_limit,True,
                     np.where(df.Salaries<lower_limit,True,False))
 #you can check outliers_df col in variable explorer
df_trimmed=df.loc[~outliers_df]
df.shape
# (310, 13)
df_trimmed.shape
#(306, 13)
sns.boxplot(df_trimmed.Salaries)
#############################################################
#Replacement Technique
#Drawback of trimming tech is we are losing the data
df=pd.read_csv("C:/5-Data_prep/ethnic diversity.csv.xls")
df.describe()

#record no 23 has got outliers
#map all the outlier value to  upper limt
df_replaced=pd.DataFrame(np.where(df.Salaries>upper_limit,
    upper_limit,np.where(df.Salaries<lower_limit,lower_limit,df.Salaries)))
df_replaced
#if the values are greater than upper limit
#map it  to upper limit ,and less than lower limimt
#map it ot lower limit ,if it is within the range
#then keep as it is
sns.boxplot(df_replaced[0])
sns.boxplot(df_replaced.Salaries)  ##giving error

########################################################
'''01-08-2024'''
#Winsorizer
from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',
                  tail='both',
                  fold=1.5,
                  variables=['Salaries']
                  )
#copy Winsorizer and paste in Help tab of 
#top right window ,study the method
df_t=winsor.fit_transform(df[['Salaries']])
sns.boxplot(df['Salaries'])
sns.boxplot(df_t['Salaries'])
########################################################
#Zero variance and near zero variance
#if there is no variance in the feature,then ML
#will not get any 
import pandas as pd
df=pd.read_csv("C:/5-Data_prep/ethnic diversity.csv.xls")
df
df.var()
#Here EmpID and ZIP is nomianal data
#Salary has 4.441953e+08 is  444195200
#not close to 0
#similarly age 8.571358e
#Both the fratures having conside
df.columns
df.var()==0 
#none of them are equal to zerp
df.var(axis=0)==0
###########################################
import pandas as pd
import numpy as np
df=pd.read_csv("C:/5-Data_prep/modified ethnic.csv.xls")
#check for null values
df.isna().sum()
'''
Position            43
State               35
Sex                 34
MaritalDesc         29
CitizenDesc         27
EmploymentStatus    32
Department          18
Salaries            32
age                 35
Race                25
dtype: int64
'''
#create an imputer that creates NaN values
#mean and median is used for numeric data
#mode is used for discrete dada(position,sex,MaritaDes)
from sklearn.impute import SimpleImputer
mean_imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
#check the dataframe
df['Salaries']=pd.DataFrame(mean_imputer.fit_transform(df[['Salaries']]))
df['Salaries'].isna().sum()
#0
#median imputer
median_imputer=SimpleImputer(missing_values=np.nan,strategy='median')
df['age']=pd.DataFrame(median_imputer.fit_transform(df[['age']]))
df['age'].isna().sum()
#0
################################3
#mode imputer
mode_imputer=SimpleImputer(missing_values=np.nan,strategy='most_frequent')
df['Sex']=pd.DataFrame(mode_imputer.fit_transform(df[['Sex']]))
df['Sex'].isna().sum()
#0
df['MaritalDesc']=pd.DataFrame(mode_imputer.fit_transform(df[['MaritalDesc']]))
df['MaritalDesc'].isna().sum()
#0
##############################
pip install imbalanced-learn scikit-learn
import numpy as np
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE

#Step 1:Generate an imbalanced dataset
x,y=make_classification(n_samples=1000, n_features=20, n_informative=2,n_redundant=10,n_clusters_per_class=1,weights=[0.99],flip_y=0,random_state=1)

'''
Parameters
    n_sample=20;
    The total no. of samples(data points) to generate.
    Here, 100 samples will be created.
    
    n_features=20;
    The total no. of features in the dataset.
    Each sample will have 20 features.
    
    n_informative=2;
    The no. of informative features.
    These feature are useful for predicting the target variable.
    
    n_redundant=10;
    The no. of redundant feature
    These feature generated as linear feature combination.
'''
#show the original class distrubution
print("Original class distrubution:",np.bincount(y))

#Step2:Apply SMOTE to balance the dataset
smote=SMOTE(random_state=42)
x_res,y_res=smote.fit_resample(x,y)
#show the new class distrubution after applying SMOTE
print("Resampled class distrubution:",np.bincount(y_res))
#################################
########### 5/8/2024 ################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#generate the sample dataset
np.random.seed(0)
data=np.random.exponential(scale=2.0,size=1000)
df=pd.DataFrame(data,columns=['Value'])
df
#perform log transformation
df['logvalue']=np.log(df['Value'])

#plot log transformation

df['LogValue']=np.log(df['Value'])

#plot the origimal and log transformed data
fig,axes=plt.subplots(1,2,figsize=(12,6))
 #original data
 
axes[0].hist(df['Value'],bins=30,color='blue',alpha=0.7)
axes[0].set_title('Original Data')
axes[0].set_xlabel('Value')
axes[0].set_ylabel('Frequency')

#log transformed data
axes[1].hist(df['LogValue'],bins=30,color='green',alpha=0.7)
axes[1].set_title('Log Transformed Data')
axes[1].set_xlabel('log(Value)')
axes[1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()
###################################
###06-08-2024###
'''
Feature->unsupervised
feature+label/target->supervised
unsupervised->clustering
'''
############09-08-2024##########
import pandas as pd
import matplotlib.pyplot as plt
#Now import file from data set and create a dataframe
univ1=pd.read_excel("C:/7-Clustering/University_Clustering.xlsx")
univ1
a=univ1.describe()
a
#We have one column "state" which really not useful we will drop it
univ=univ1.drop(["State"],axis=1)
univ
#We know that there is scale difference among the columns,
#which we have to remove
#either by using normalization or standardization
#whenever there is mixed data apply normalization
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x
#Now apply this normalization function to univ dataframe
#for all the rows and columns from 1 until end
#since 0th column has university name hence skipprd
df_norm=norm_func(univ.iloc[:,1:])
df_norm
#you can check the df_norm dataframe which is scaled
#between values from 0 to 1
#you can apply describe function to new datafFrame
b=df_norm.describe()
b
#Before you apply clustering,you need to plot dendrograme first
#Now to create dendogram, we need to measure distance, we have to import linkage
from scipy.cluster.hierachy import linkage
import scipy.cluster.hierachy as sch
#linkage function gives us hierachical or agloramative clustering
#ref the help for linkage
z=linkage
###########################



























