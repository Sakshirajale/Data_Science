"""
Created on Tue Apr  2 08:28:04 2024

@author: ACER
Python for Datascience
-Pandas
-Numpy
-NLP
Panads:-Series,columns,second series
"""
#A series is used to model one dimentional data,
#Similar to a list in python.
#The series object also has a few more bits of data,
#including an # index and a name(string).
import pandas as pd
songs2=pd.Series([145,142,38,13],name='counts')
#It is easy to inspect the index of a series (or data for string)
songs2.index
songs2
'''o/p:-
0    145
1    142
2     38
3     13
Name: counts, dtype: int64
'''
#The index can be string based as well,
#in which case pandas indicates
#that the datatype for the index is object (not string)
songs3=pd.Series([145,142,38,13],name='counts',index=['Paul','John','George','Rings'])
songs3.index
songs3
songs3.iloc[3]
#The NAN value
#numeric column will become NAN.
import pandas as pd
f1=pd.read_csv('age.csv.xls')
f1
df=pd.read_excel('c:/1-python/Bahaman.xlsx')
df
#None, NAN, nan, and null are synonyms
#The series object behaves similaryly to a Numpy array.
import numpy as np
numpy_ser=np.array([145,142,38,13])
songs3[1]
#142
numpy_ser[1]
numpy_ser[0]
#The both have methods in common
songs3.mean()
numpy_ser.mean()
#########################################
#The pandas series data stucture provides
#support for the basic CRUD
#operatins-create,read,update and delete
#Creation
george=pd.Series([10,7,1,22],index=['1968','1969','1970','1970'],name='Georgr_Songs')
george
#The previous example illustrates an interesting feature of pandas_the
#index values are string and they are not unique.
#This can cause some confusion, but can also be useful when duplicate 
#index items are needed.
###########################################################################
#Reading
#To read or select the data from a series
george['1968']
george['1970']
#We can iterate over data in a series
#as well. when iterating over a series
for item in george:
    print(item)
#########################################################################
#Updating:-updating values in a series can be a littele tricky as well.
#To update a value for given index label, the standard index assignment operation works
george['1969']=68
george['1969']
george
########################################################################
'''
03-04-2024
'''
########################################################################
#Deletion:- The del statement appears to have problems with duplicate index
import pandas as  pd
s=pd.Series([2,3,4], index=[1,2,3])
del s[1]
s
#######################################################################
#Convert Types
#string useastype(str)
#numeric use pd.to_numeric
#integer use.astype(int)
#Note that this will fail with NAN
#datetime use pd.to_datetime

songs_66=pd.Series([3.0,None,11.0,9.0], index=['George','Ringo','John','Paul'], name='counts')
songs_66.dtypes
#dtype('float64')
pd.to_numeric(songs_66.apply(str))
#there will be error
pd.to_numeric(songs_66.astype(str), errors='coerce')
#If we pass errors='coerce'
#we can see that it supports may formats
songs_66.dtypes
#Dealing with none
#the .fillna method will replace them with a given value
songs_66=songs_66.fillna(-1)
songs_66=songs_66.astype(int)
songs_66.dtypes
#NAN value can be droped from
#the series using .dropna
songs_66=songs_66.dropna()
songs_66
#########################################################
#Append,combining, and joining two series
songs_69=pd.Series([7,16,21,39], index=['Ram','Sham','Ghansham','Krishna'], name='counts')

#To concatenate two series together, simply use the .append()
songs=pd.concat([songs_66,songs_69])
songs
#########################################################
#Plotting series
import matplotlib.pyplot as plt
fig = plt.figure()
songs_69.plot()
plt.legend()
###########################################################
fig = plt.figure()
songs_69.plot(kind='bar')
songs_66.plot(kind='bar', color='r')
plt.legend()
############################################################
'''
Histogram
purpose:- undrstand distrubution of data:normal,gaussian distrubution
'''
import matplotlib.pyplot as plt
import numpy as np 
data = pd.Series(np.random.randn(500),name='500_random')
fig = plt.figure()
ax = fig.add_subplot(111)
data.hist()
############################################################
'''
04-04-2024
'''
#pandas dataframe is a two data structure
#Data frame features
#it supports named rows and columns
#step 1-go to the anaconda navigator
#step-2 select environmenatl tab
#step-3by default it will be base terminal
#step-4on base terminal-pip install pandas
#or conda install pandas
####
#upgrade pandas to latest or specific version on base terminal


#to check the version of pandas
import pandas as pd
pd.__version__
#######################################################
#Create using constructor
#create pandas dataframe from list
import pandas as pd
technologies = [["Spark",20000, "30days"],["pandas",20000, "40days"]]
df=pd.DataFrame(technologies)
print(df)
##########################################################
#Since we have not given labels to column and
#indexes,DataFrame by default assigns
#incremnetal sequence numbers as labels to both rows and columns
#these are called index.
#Add column and row labels to the DataFrame
column_names=["Courses","Fee","Duration"]
row_label=["a","b"]
df=pd.DataFrame(technologies,columns=column_names,index=row_label)
print(df)
###############################################################
#check datatypes
df.dtypes
##############################################################
#You can also assign custom data types to columns.
#set custom types to DataFrame
technologies ={'Courses':["Spark","PySpark","Hadoop","Python","Pandas","Oracle","Java"],
               'Fee':[20000,25000,30000,35000,40000,45000,50000],
               'Duration':["30days","40days","35days","40days","60days","50days","5days"],
               'Discount':[11.8,23.7,13.4,15.7,12.5,25.4,18.4]
               }
df=pd.DataFrame(technologies)
print(df.dtypes)
df
#convert all types to best possible types
df2=df.convert_dtypes() #object=>string
print(df2.dtypes)
#change all columns to same type
df=df.astype(str) #string=>object
print(df.dtypes)
#change type for one or multiple columns
df=df.astype({"Fee":int, "Discount":float})
print(df.dtypes)
#convert data type for all columns in a list
df =pd.DataFrame(technologies)
df.dtypes
cols = ['Fee','Discount']
df[cols] = df[cols].astype('float')
df.dtypes
#Ingnors error   ##Important part
df=df.astype({"Courses":int},errors='ignore')
df.dtypes
#Generates error ##Important part
df=df.astype({"Courses":int},errors='raise')
#converts feed column to numeric type
df=df.astype(str)
print(df.dtypes)
df['Discount']=pd.to_numeric(df['Discount'])
df.dtypes
###################################################
import pandas as pd
#Create DataFrame from dictionary
technologies ={'Courses':["Spark","PySpark","Hadoop","Python","Pandas","Oracle","Java"],
               'Fee':[20000,25000,30000,35000,40000,45000,50000],
               'Duration':["30days","40days","35days","40days","60days","50days","5days"],
               'Discount':[11.8,23.7,13.4,15.7,12.5,25.4,18.4]
               }
df=pd.DataFrame(technologies)
df
############################################
#convert DataFrame to csv
df.to_csv('data_file.csv')
df
##########################################
#Create DataFrame from csv file
df=pd.read_csv('data_file.csv')
############################################
#Pandas DataFrame - basic operation
#Create DataFrame with None/Null to work with example
import pandas as pd
import numpy as np
technologies ={'Courses':["Spark","PySpark","Hadoop","Python",None,"Oracle","Java"],
               'Fee':[20000,25000,30000,np.nan,40000,45000,50000],
               'Duration':["30days","40days","35days","40days","60days"," ","5days"],
               'Discount':[11.8,23.7,13.4,15.7,12.5,25.4,18.4]
               }
row_labels=['r0','r1','r2','r3','r4','r5','r6']
df=pd.DataFrame(technologies,index=row_labels)
print(df)
################################################################
'''
05-04-2024
'''
################################################################
#DataFrame properties
df.shape
#(7,4)
df.size
#28
df.columns
df.columns.values
df.index
df.dtypes
df.info
'''
1)understanding objectives
2)Data dictionary
3)EDA exploratry
4)Data preprocessing
5)Model
6)Perfomance Evaluation
7)Deploy
8)Monitoring and Maintaining
'''
##################################################################
#Accesing one column contents
df['Fee']
#Accessing two columns contents
#Method 1
cols=['Fee','Duration']
df[cols]
#Method 2
df[['Fee','Duration']]
#select certain rows and assign it to another dataframe
#df[rows,columns]
#df['start_rows':'end_rows','start_column':'end_column']
##df["start row":"end rows"]
#6 rows
df2=df[6:]
df2
#0 th 5
df2=df[:6]
df3=df[2:5]
df2
df3
####################################################################
#Accessing certain cell from column 'Duration'
df['Duration'][3]
df['Discount'][4]
#Subtracting specific value fro a column
df['Fee'] = df['Fee'] - 500
df['Fee']
df['Discount']=df['Discount']-2.0
df['Discount']
#Pandas to manipulate DataFrame
#Describe DataFrame
#Describe DataFrame for all numeric columns##It is method
df.describe()
#It will show 5 number summary
###################################################################
#rename()-Rename pandas DataFrame columns
df=pd.DataFrame(technologies, index=row_labels)

#Assign new header by setting new column names.
df.columns=['A','B','C','D']
df
##################################################################
#Rename column names using rename() method
df=pd.DataFrame(technologies,index=row_labels)
df.columns=['A','B','C','D']
#for rows axis=0,,,columns axis=1
df2 = df.rename({'A':'c1','B':'c2'}, axis=1)
df2 = df.rename({'C':'c3','D':'c4'}, axis='columns')
df2 = df.rename(columns={'A':'c1','B':'c2'})
df2
##################################################################
#Drop DataFrame rows and columns
df=pd.DataFrame(technologies,index=row_labels)
df
#Drop rows by labels
df1=df.drop(['r1','r2'])
df1
#Delete rows by position/index
df1=df.drop(df.index[1])
df1
df1=df.drop(df.index[[1,3]])
yp33df1
#Delete rows by index range
df1=df.drop(df.index[2:])
df1
df1=df.drop(df.index[:2])
df1
#When you have default indexes for rows
df=pd.DataFrame(technologies)
df
df1=df.drop(0)
df1
df=pd.DataFrame(technologies)
df1=df.drop([0,3],axis=0) #it will delete row0 and row3
df1
df1=df.drop(range(0,3)) #it will delete 0 and 1
df1
##############################################################
'''
10-04-2024
'''
#Droping of columns
import pandas as pd
technologies ={'Courses':["Spark","PySpark","Hadoop","Python","Pandas","Oracle","Java"],
               'Fee':[20000,25000,30000,35000,40000,45000,50000],
               'Duration':["30days","40days","35days","40days","60days","55days ","5days"]
              }
df=pd.DataFrame(technologies)
print(df)
#Drop column by name
#Drops 'Fee' column

#Explicitly using paramenter name 'labels'
df2=df.drop(labels=['Fee'], axis = 1)
df2

#Alternatively you can also use columns instead of labels
df2=df.drop(columns=['Fee'], axis = 1)
df2
###################################################
#drop column by index
print(df.drop(df.columns[1], axis = 1))
df = pd.DataFrame(technologies)
df
#using inplace=True
df.drop(df.columns[2], axis =1, inplace=True)
print(df)
###################################################
df=pd.DataFrame(technologies)
#Drop Two or more columns by label name
df2=df.drop(["Courses", "Fee"], axis = 1)
print(df2)
####################################################
#Drop two or more columns by index
df=pd.DataFrame(technologies)
df2=df.drop(df.columns[[0,1]], axis = 1)
print(df2)
######################################################
#Drop columns from list to columns
df=pd.DataFrame(technologies)
df
df.columns
lisCol = ["Courses","Fee"]
df2=df.drop(lisCol, axis=1)
print(df2)
######################################################
#Remove columns from DataFrame inplace
df = pd.DataFrame(technologies)
df.drop(df.columns[1], axis =1, inplace=True)
df
#using inplace=True
######################################################
#####################################################
#Important for interview
#pandas select rows by index (Positive/label) use of
import pandas as pd
import numpy as np
technologies ={'Courses':["Spark","PySpark","Hadoop","Python","Pandas","Oracle","Java"],
               'Fee':[20000,25000,30000,35000,40000,45000,50000],
               'Duration':["30days","40days","35days","40days","60days","55days ","5days"],
               'Discount':[1000,2300,1000,1200,2500,1300,1400]
              }
row_labels=['r0','r1','r2','r3','r4','r5','r6']
df=pd.DataFrame(technologies, index=row_labels)
print(df)
#df.iloc[startrow:endrow, startcolumn:endcolumn]
df =pd.DataFrame(technologies, index=row_labels)
#Below are quick example
df2=df.iloc[:,0:2]
df2
df2=df.iloc[0:4,:]
df2=df.iloc[:,:]
df2
#This line uses the slicing operator to get DataFrame items by index.
#The first slice [:] indicates to return all rows.
#The second slice specifies that only columns
#between 0 an 2 (excluding 2) should be returned.

df2=df.iloc[0:2,:]
df2
#In this case, the first slice [0:2] is requesting 
#only rows 0 through 1 of the DataFrame
#The second slice [:] indicates that all columns are required.

#slicing specific rows and columns using iloc attribute
df3=df.iloc[1:2,1:3]
df3

#Another example
df3=df.iloc[:,1:3]
df3
#The second operator [1:3] yields columns 1 and 3 only.
#Select rows by integer index
df2=df.iloc[2]  #select rows by index
df2

df2=df.iloc[[2,3,6]]  #select rows by index list
df2=df.iloc[1:5]      #selct rows by integer index range
df2=df.iloc[:1]       #select first row
df2=df.iloc[:3]       #select first three row
df2=df.iloc[-1:]      #select last row 
df2=df.iloc[-3:]      #select last three row
df2=df.iloc[::2]      #select alternative rows
df2

#select rows by index labels
###***important part for interview***###
df2=df.loc[['r2']]   #select row by label
df2
df2=df.loc[['r2','r3','r6']]  #select row by index
df2=df.loc['r1':'r5']     #select rows by label index
df2=df.loc['r1':'r5':2]   #select alternative rows with index
df2
##################################################################
#Pandas select columns by Name or index
#By using df[]
df2=df['Courses']
df2
##select multiple columns
df2=df[["Courses","Fee","Duration"]]

#using loc[] to take column slices
#loc[] syntax to slice columns
#df.loc[:,start:stop:step]
#select multiple columns
df2=df.loc[:,["Courses","Fee","Duration"]]
df2
#select random columns
df2=df.loc[:,["Courses","Fee","Discount"]]
df2
#select column between two columns
df2=df.loc[:,'Fee':'Discount']
df2
##select columns by range
df2=df.loc[:,'Duration']
df2
#select columns by range
df2=df.loc[:,:'Duration']
df2
##select every alternative column
df2=df.loc[:,::2]
df2
##############################################
#############################################
#Pandas DataFrame.query() by example
#Query all rows with courses equals 'spark'
df2=df.query("Courses=='Spark'")
print(df2)
##################################################
#Not equal condition 
df2=df.query("Courses != 'Spark'")
df2
##############################################
import pandas as pd
import numpy as np
technologies ={'Courses':["Spark","PySpark","Hadoop","Python","Pandas"],
               'Fee':[20000,25000,30000,35000,40000],
               'Discount':[0.1,0.2,0.5,0.1,0.6]
              }
df=pd.DataFrame(technologies)
print(df)
#########################################################
#pandas add column to DataFrame
#Add new column to the Dataframe
tutors = ['Om','Sai','Ram','Aditya','Ramesh']
df2=df.assign(TutorsAssigned=tutors)
print(df2)
##########################################################
#Add multiple columns to the DataFrame
MNCCompanies=['TATA','HCL','INFOSYS','GOOGLE','AMAZON']
df2=df.assign(MNC=MNCCompanies,tutors=tutors)
df2
##################################################
#Derive New column from Existing column
df=pd.DataFrame(technologies)
df2=df.assign(Discount_Percent=lambda x: x.Fee * x.Discount / 100)
print(df2)
#######################################################
#Append column to existing pandas DataFrame
#Add new column to the existing DataFrame
df=pd.DataFrame(technologies)
df["MNCCompanies"]= MNCCompanies
print(df)
###############################################################
#Add new column at the specific position
df=pd.DataFrame(technologies)
df.insert(0,'Tutors',tutors)
print(df)
###
df=pd.DataFrame(technologies)
df.insert(1,'MNCCompanies',MNCCompanies)
print(df)
################################################
#Pandas column with example
import pandas as pd
technologies ={'Courses':["Spark","PySpark","Hadoop","Python","Pandas","Oracle","Java"],
               'Fee':[20000,25000,30000,35000,40000,45000,50000],
               'Duration':["30days","40days","35days","40days","60days","55days ","5days"]
              }
df=pd.DataFrame(technologies)
print(df)
df.columns
print(df.columns)
#Pandas rename column name
#Rename a single column
df2=df.rename(columns = {'Courses':'Courses_List'})
df2
print(df2.columns)
#Alternatively, you can also write
df2=df.rename({'Courses':'Courses_List'}, axis=1)
df2
df2=df.rename({'Courses':'Courses_List'}, axis='columns')
df2
#In order to change columns on the existing DataFrame
#without copying to the new DataFrame,
#you have to use inplace=True.
df.rename({'Courses':'Courses_List'}, axis='columns',inplace=True)
df
print(df.columns)
#####################################################
'''
15-04-2024
'''
#Rename multiple columns with inplace
df.rename(columns = {'Courses':'Courses_List','Fee':'Courses_Fee','Duration':'Courses_Duration'}, inplace=True)
print(df.columns)
##########################################################
#########################################################
#Quick examples of get the rows in Dataframe
rows_count=len(df.index)
rows_count
rows_count=len(df.axes[0])
rows_count
column_count=len(df.axes[1])
column_count
#######################################################
df=pd.DataFrame(technologies)
row_count=df.shape[0]  #Return no of rows
rows_count
col_count=df.shape[1]  #return no of columns
col_count
#o/p:-3
#######################################################
#Below are quick examples
#using DataFrame.apply() to apply function add column
import pandas as pd
import numpy as np
data={"A":[1,2,3],"B":[4,5,6],"C":[7,8,9]}
df=pd.DataFrame(data)
print(df)
#
def add_3(x):
    return x+3
df2=df.apply(add_3)
df2
#apply() add one column 
df2=((df.A).apply(add_3))
df2
df3=((df.C).apply(add_3))
df3
#########################################
##Important function
#using apply function single column
def add_4(x):
    return x+4
df["B"]=df["B"].apply(add_4)
df["B"]
#Apply to multiple columns
df[['A','B']]=df[['A','B']].apply(add_4)
df
#Apply a lambda function to each column
df2=df.apply(lambda x:x+10)
df
###############################################
#Apply lambda function to single column
#using DataFrame.apply() and lambda function
df["A"]=df["A"].apply(lambda x: x-2)
print(df)
#################################################
#Using pandas.DataFrame.transform() to apply function column
#Using DataFrame.transform()
def add_2(x):
    return x+2
df=df.transform(add_2)
df
###################################
#using pandas.DataFrame.map() to single column
df['A']=df['A'].map(lambda A:A/2.)
df
#########################################
#using numpy function on single column
#using DataFrame.apply() & [] operator
import numpy as np
data={"A":[1,2,3],"B":[4,5,6],"C":[7,8,9]}
df=pd.DataFrame(data)
df
df['A']=df['A'].apply(np.square)
df
df['B']=df['B'].apply(np.square)
df['C']=df['C'].apply(np.square)
#using numpy fuction on multiple column
df[['A','B']]=df[['A','B']].apply(np.square)
df
###############################################
#Using Numpy.square() method
#using numpy.square() and [] operator
#single column
df['A']=np.square(df['A'])
df
#multiple column
df[['A','B']]=np.square(df[['A','B']])
df
###################################################
import pandas as pd
technologies ={'Courses':["Spark","Spark","Hadoop","Python","Pandas","Python","NA"],
               'Fee':[20000,25000,30000,35000,40000,45000,50000],
               'Duration':["30days","40days","35days","40days","60days","55days ","5days"],
               'Discount':[1000,2300,1000,None,2500,1300,0]
              }
df=pd.DataFrame(technologies)
print(df)

#use  groupby() to compute the sum
df2=df.groupby(['Courses']).sum()
print(df2)
df2=df.groupby(['Discount']).sum()
print(df2)
#Group by multiple columns
df2=df.groupby(['Courses','Duration']).sum()
df2
#####################################
#Add index to the grouped data 
#Add row index to the group by result
df2=df.groupby(['Courses','Duration']).sum().reset_index()
df2
#################################################################
import pandas as pd
technologies ={'Courses':["pySpark","Spark","Hadoop","Python","Pandas","Python","NA"],
               'Fee':[20000,25000,30000,35000,40000,45000,50000],
               'Duration':["30days","40days","35days","40days","60days","55days ","5days"],
               'Discount':[1000,2300,1000,20000,2500,1300,0]
              }
df=pd.DataFrame(technologies)
print(df)
df.columns
#Get the of all column names  from headers
column_headers=list(df.columns.values)
print("The column header:",column_headers)
#####################################################
'''
16-04-2024
'''
########################################################
#Pandas shuffle DataFrame rows
import pandas as pd
technologies ={'Courses':["pySpark","Spark","Hadoop","Python","Pandas","Python","Oracle"],
               'Fee':[20000,25000,30000,35000,40000,45000,50000],
               'Duration':["30days","40days","35days","40days","60days","55days ","5days"],
               'Discount':[1000,2300,1000,20000,2500,1300,1500]
              }
df=pd.DataFrame(technologies)
print(df)
#Pandas shuffle DataFrame rows
#Shuffle the DataFrame rows and return all rows
df1=df.sample(frac=1)
df1
##############################################################
#Create a new index starting from zero
df1=df.sample(frac=1).reset_index()
df1
###############################################################
#Drop shuffle index
df1=df.sample(frac=1).reset_index(drop=True)
df1
################################################################
import pandas as pd
technologies ={'Courses':["pySpark","Spark","Hadoop","Python"],
               'Fee':[20000,25000,30000,35000],
               'Duration':["30days","40days","35days","40days"]
              }
index_labels=['r1','r2','r3','r4']
df1=pd.DataFrame(technologies,index=index_labels)
df1
technologies2={'Courses':["Spark","Java","Python","Go"],'Discount':[2000,2300,1200,2000]}
index_labels2=['r1','r6','r3','r5']
df2=pd.DataFrame(technologies2,index=index_labels2)
df2
#Pandas join
df3=df1.join(df2,lsuffix="_left", rsuffix="_right")
df3
#Pandas inner join DataFrame
df3=df1.join(df2,lsuffix="_left", rsuffix="_right",how='inner')
df3
##############################################################
#Pandas left join DataFrames
df3=df1.join(df2,lsuffix="_left", rsuffix="_right",how='left')
df3
##Pandas right join DataFrames
df3=df1.join(df2,lsuffix="_left", rsuffix="_right",how='right')
df3
######################################################
#Pandas Merge DataFrame
import pandas as pd
technologies ={'Courses':["pySpark","Spark","Hadoop","Python"],
               'Fee':[20000,25000,30000,35000],
               'Duration':["30days","40days","35days","40days"]
              }
index_labels=['r1','r2','r3','r4']
df1=pd.DataFrame(technologies,index=index_labels)
df1
technologies2={'Courses':["Spark","Java","Python","Go"],'Discount':[2000,2300,1200,2000]}
index_labels2=['r1','r6','r3','r5']
df2=pd.DataFrame(technologies2,index=index_labels2)
df2
#using pandas.merge():-merge same element in dataframe 1 and dataframe 2
df3=pd.merge(df1,df2)
df3
#Using DataFrame.merge()
df3=df1.merge(df2)
df3
##################################################
#use pandas.concat() to concat two dataframe:-add two dataframe in vertically
df=pd.DataFrame({'Courses':["pySpark","Spark","Hadoop","Python"],'Fee':[20000,25000,30000,35000]})
df
df1=pd.DataFrame({'Courses':["Pandas","Spark","Hadoop","Python"],'Fee':[22000,2000,35000,40000]})
df1
#using pandas.concat to concat two dataframe
data=[df,df1]
df2=pd.concat(data)
df2
#######################################################################
#Concatenate multiple DataFrame using pandas.concat:-add multiple dataframe use any column name they can combine 
df=pd.DataFrame({'Courses':["pySpark","Spark","Hadoop","Python"],'Fee':[20000,25000,30000,35000]})
df
df1=pd.DataFrame({'Courses':["Pandas","Spark","Hadoop","Python"],'Fee':[22000,2000,35000,40000]})
df1
df2=pd.DataFrame({'Duration':["30days","40days","35days","40days","55days"],'Discount':[1000,2300,1000,20000,2500]})
df2

#Appending multiple DataFrames
df3=pd.concat([df,df1,df2])
df3
#################################################
'''
18-04-2024
'''
#Read CSv file into DataFrame 
df=pd.read_csv('Courses.csv')
#write DataFrame to excel file
df.to_exel('Courses.xlsx')
import pandas as pd
#read excel file
df=pd.read_excel('Courses.xlsx') 
df
#############################################
#Using series.values.tolist()
col_list=df.Courses.values
print(col_list)
col_list=df.Courses.values.tolist()
print(col_list)

#Using Series.values
col_list=df["Courses"].values.tolist()
print(col_list)

#Using list() function
col_list=list(df["Courses"])
print(col_list)
import numpy as np
#convert to numpy array
col_list=df['Courses'].to_numpy()
print(col_list)
##########################################################
#What is Numpy?
#->The numpy library is a popular open source python.
'''
While a python list can contain different
data types within a single list,
all of the elements in a Numpy rray
should be homogeneous
'''

#Array in Numpy
#Create ndarray
import numpy as np
arr=np.array([10,20,30])
print(arr)

#Create a Multi-Dimensional array
arr=np.array([[10,20,30],[40,50,60]])
print(arr)
#############################
#Represent the Minimum Dimensions
#Use ndmin parameter to specify how many minimum
#dimensions you wanted to create an array with minimum dimension
arr=np.array([10,20,30,40],ndmin=3)
print(arr)


#Change the data type
#dtype parameter
arr=np.array([10,20,30],dtype=complex)
print(arr)

#Get the dimension of array
arr=np.array([[1,2,3,4],[7,8,9,7],[9,10,11,12]])
print(arr.ndim)  #ndim:-It is property to find out dimension of array
print(arr)

#Finding the size of each item in the array
#long int-2 bytes
#int-2 bytes
arr=np.array([10,20,30])
print("Each item contain in bytes:",arr.itemsize)

#Get the datatype of array
arr=np.array([10,20,30])
print("Each item is of the type",arr.dtype)
#############################################
#Get the shape and size of array
arr=np.array([[10,20,30,40],[50,60,70,80]])
print("Array size is",arr.size)
print("Array size is",arr.shape)
############################
#create Numpy array from list
#creation of arrays
arr=np.array([10,20,30])
print("Array:",arr)
################################
#Creating array from list with type float
arr=np.array([[10,20,40],[30,40,50]],dtype='float')
print("Array created by using list:\n",arr)

######################################################
#Create a sequence of integers using arange()
#create a sequence of integers
#from 0 to 20 with steps of 3
arr=np.arange(0,20,3)
print("A sequence array with steps of 3:\n",arr)
#########################################
#Array indexing in Numpy
#Access single element using index
arr=np.arange(11)
print(arr)
#[ 0  1  2  3  4  5  6  7  8  9 10]
print(arr[2])
print(arr[-2])
print(arr[-5])
print(arr[:2])
#######################################
#Multi-dimensional array indexing
#Access multi-dimensional array element using array indexing
arr=np.array([[10,20,30,40,50],[20,30,50,10,20]])
print(arr)
print(arr.shape) #now x is two dimensional #(2, 5)
print(arr[1,1])
print(arr[0,4])
print(arr[1,-1]) #rows start from 0,we need 1st row and -1 columns
#rows    0   1   2   3   4 <-columns
#0  [10, 20, 30, 40, 50]
#1  [20, 30, 50, 10, 20]
#######################################
#Access array elements using slicing
arr=np.array([0,1,2,3,4,5,6,7,8,9])
x=arr[1:8:2]  #start:end:in step of 3
print(x)

#Example
x=arr[-2:3:-1] #start last but one(-2) upto 3 but not 3
print(x)
#
x=arr[-2:10] #start last but one(-2) upto 10 but not 10
print(x)

##################################################
#Indexing in Numpy
multi_arr=np.array([[10,20,30,40],[40,50,70,90],[60,10,70,80],[30,90,40,30]])
multi_arr
#slicing array
#for multi dimensional Numpy arrays,
#you can access the elements as below
multi_arr[1,2] #To access value at row
multi_arr[1,:] #to get the value at row 1 and all columns
multi_arr[:,1] #to access the value at all rows and 1 column
x=multi_arr[:3,::2] #access the value 0 to 3 , in all selected alternate columns
print(x)
x=multi_arr[::3,::2]
x
###################################################
'''
19-04-2024
'''
#x=3->scalar
#x=[10,20,30]->vector
#x=[[10,20,30,40],[50,60,70,80]]->Matrix(two dimensional)
#x=[[[]]]->tensor=>more than two dimensional
#Integer array indexing
#Integer array indexing allows the selection
arr=np.arange(35).reshape(5,7)
print(arr)
###############################################
#Boolean array indexing
#This advanced indexing occurs when an object is an array object of boolean process
#Use this method when we want to pick elements
#from the array which satisfy some condition.
import numpy as np
#Boolean array indexing
arr=np.arange(12).reshape(3,4)
print(arr)
###################################
rows=np.array([False,True,True]) #not 0th row only
rows
wanted_rows=arr[rows, :]
print(wanted_rows)
###########################################
list=[20,40,60,80]
#Convert array
array=np.array(list)
print("Array:",array)
#################################
#numpy.asarray():using numpy.asarray() function
#use asarray()
list=[20,40,60,80]
array=np.asarray(list)
print("Array:",array)
print(type(array))
####################################
#Numpy array properties
#1)ndarray.shape
#2)ndarray.ndim
#3)ndarray.itemsize
#4)ndarray.size
#5)ndarray.dtype
#1)ndarray.shape:-to get the shape of a python numpy array use numpy
#shape
array=np.array([[1,2,3],[4,5,6]])
array
print(array.shape)

#Resize the array
array=np.array([[10,20,40],[50,60,70]])
array.shape=(3,2)
print(array)

#Reshape usage
array=np.array([[10,20,30,40],[50,60,70,80]])
array.reshape(3,2)
print(array)

##############################################
#Arithematic operators on arrays apply elementwise.
arr1=np.arange(16).reshape(4,4)
arr2=np.array([1,3,2,4])
#add()
add_arr=np.add(arr1,arr2)
print(f"Adding two arrays:\n{add_arr}")

#subtract()
sub_arr=np.subtract(arr1,arr2)
print(f"Subtration two arrays:\n{sub_arr}")

#multiplication()
mul_arr=np.multiply(arr1,arr2)
print(f"Multiplication two arrays:\n{mul_arr}")

#Division()
div_arr=np.divide(arr1,arr2)
print(f"Division two arrays:\n{div_arr}")
##############################################
#numpy.reciprocol()
#This fun returns the reciprocol of argument element-wise. For elements with absolute values
#larger than 1, the result is always 0 because of the way in which

#
arr1=np.array([50,10.3,5,1,200])
rep_arr1=np.reciprocol(arr1)
print(f"After applying reciprocol function to array:\n{rep_arr1}")

#numpy.power():-this numpy power() function treates elements in the input array
#To perform numpy power operation
arr1=np.array([3,10,5])
pow_arr1=np.power(arr1,3)
print(f"After applying power function to array:\n{pow_arr1}")

arr2=np.array([3,2,1])
print("My second array:\n",arr2)
pow_arr2=np.power(arr1,arr2)
print(f"After applying power function to array:\n{pow_arr2}")
#Applying power function again:
#[ 27 100   5]

#mod():-this function returns the remainder of the division of the corresponding elements
#in the input array. The function numpy.remainder() also produces the same result.

#to perform mod function
#on numpy array
import numpy as np
arr1=np.array([7,20,13])
arr2=np.array([3,5,2])
arr1
arr1.dtype
#mod()
mod_arr=np.mod(arr1,arr2)
print(f"Atfer applying mod function to array:\n{mod_arr}")    
###########################################################
#Create empty array
from numpy import empty
a=empty([3,3])
print(a)
###################################
#Create zero array
from numpy import zeros
a=zeros([3,5])
print(a)
#################################
#How to check version of python
np.__version__
################################
#Create one array
from numpy import ones
a=ones([5])
print(a)
####################################
#Create array with vstack
from numpy import array
from numpy import vstack
#create first array
a1=array([1,2,3])
print(a1)
#create second array
a2=array([4,5,6])
print(a2)
#vertical stack
a3=vstack((a1,a2))
print(a3)
print(a3.shape)
#############################
#create array with hstack
from numpy import array
from numpy import hstack
#create first array
a1=array([1,2,3])
print(a1)
#create second array
a2=array([4,5,6])
print(a2)
#create horizontal stack
a3=hstack((a1,a2))
print(a3)
print(a3.shape)
#######################################################
'''
24-04-2024
'''
#######################################################
from numpy import array
#define matrix
A= array([[1,2],[3,4],[5,6]])
print(A)
#calculate tranpose
C=A.T
print(C)
#####################################################
#inverse matrix
from numpy import array
from numpy.linalg import inv
#define matrix
A= array([[1.0,2.0],[3.0,4.0]])
print(A)
#inverse matrix
B= inv(A)
print(B)
#Multipy A and B
#Identity matrix
I=A.dot(B)
print(I)
###################################################
#Sparse matrix
from numpy import array
from scipy.sparse import csr_matrix
#create dense matrix
A= array([[1,0,0,1,0,0],[0,0,2,0,0,1],[0,0,0,2,0,0]])
print(A)
#o/p:-
'''
[[1 0 0 1 0 0]
 [0 0 2 0 0 1]
 [0 0 0 2 0 0]]
'''
#convert to sparse matrix (CSR mrthod)
S=csr_matrix(A)
print(S)
#o/p
'''
(0, 0)	1
(0, 3)	1
(1, 2)	2
(1, 5)	1
(2, 3)	2
'''
#reconstruct dense matrix
B=S.todense()
print(B)
########################################################
#df.describe()
#Matlotlib():- write a python program to draw a line with suitable label in the
import matplotlib.pyplot as plt
X=range(1,50)
Y=[value*3 for value in X]
print("Values of X:")
print(*range(1,50))
'''
This is equivalent to- 
i in range(1,50):
    print(i, end= ' ')
'''
print("Values of Y (thrice of X):")
print(Y)
#plot lines and/or markers to the Axes.
plt.plot(X, Y)
#set the x axis label of the current axis.
plt.xlabel('x - axis')
#set the y axis label of the current axis.
plt.ylabel('y - axis')
#set a title
plt.title('Sample graph!')
#Display the figure.
plt.show()
########################################
import matplotlib.pyplot as plt
#x axis values
x= [1,2,3]
#y axis value
y= [2,4,1]
#plot lines and/or markers to the Axes.
plt.plot(x, y)
#set the x axis label of the current axis.
plt.xlabel('x - axis')
#set the y axis label of the current axis.
plt.ylabel('y - axis')
#set a title
plt.title('Sample graph!')
#Display the figure.
plt.show()
##################################################3
#Write a python program to plot two or more lines
#on same plot with suitable legends of each lines.
import matplotlib.pyplot as plt
#line 1 points
x1=[10,20,30]
y1=[20,40,10]
#line 2 points
x2=[10,20,30]
y2=[40,10,30]
#plotting the line 1 points
plt.plot(x1, y1, label = 'line 1')
#plotting the line 2 points
plt.plot(x2, y2, label = 'line 2')
plt.xlabel('x - axis')
#set the y axis label or the current axis
plt.ylabel('y - axis')
#set a title of the current axes.
plt.title('Two or more lines on same plot with suitable legend')
#show a legend on the plot
plt.legend()
#display a figure
plt.show()
###################################################
import matplotlib.pyplot as plt
#line 1 points
x1=[10,20,30]
y1=[20,40,10]
#line 2 points
x2=[10,20,30]
y2=[40,10,30]
#set the x axis label or the current axis
plt.xlabel('x - axis')
#set the y axis label or the current axis
plt.ylabel('y - axis')
#set a title of the current axes.
plt.title('Two or more lines with different width and colors with suitable legend')
#display a figure
plt.plot(x1,y1, color='blue', linewidth=3, label='line1-width')
plt.plot(x2,y2, color='red', linewidth=5, label='line2-width')
######################################################
import matplotlib.pyplot as plt
#line 1 points
x1=[10,20,30]
y1=[20,40,10]
#line 2 points
x2=[10,20,30]
y2=[40,10,30]
#set the x axis label or the current axis
plt.xlabel('x - axis')
#set the y axis label or the current axis
plt.ylabel('y - axis')
#set a title of the current axes.
plt.title('Two or more lines with different width and colors with suitable legend')
#display a figure
plt.plot(x1,y1, color='blue', linewidth=3, label='line1-dotted',linestyle='dotted')
plt.plot(x2,y2, color='red', linewidth=5, label='line2-dashed',linestyle='dashed')
######################################################################################
'''
25-04-2024
'''
#EDA:-explotary data analysis
#write a python program to plot two or more lines and set the line markers.
import matplotlib.pyplot as plt
# x axis values
x=[1,4,5,6,7]
# y axis values
y=[2,6,3,6,3]

#plotting the points
plt.plot(x, y, color='red', linestyle='dashdot', linewidth=3, marker='o', markerfacecolor='blue', markersize=12)
#set the y-limits of the current axis
plt.ylim(1,8)
#set the x-limits of the current axis.
plt.xlim(1,8)
#set the x axis label or the current axis
plt.xlabel('x - axis')
#set the y axis label or the current axis
plt.ylabel('y - axis')
#giving a title to my graph
plt.title('Display marker')
#function to show the plot
plt.show()
###############################################################
#write a python programming to display a bar chart of the popularity of programming languages.
import matplotlib.pyplot as plt
x=['Java', 'Python', 'PHP', 'Javascript', 'C#', 'C++']
popularity=[22.2,17.6,8.8,8,7.7,6.7]
x_pos=[i for i, _ in enumerate(x)]
plt.bar(x_pos, popularity, color='blue')
plt.xlabel('Languages')
plt.ylabel('Popularity')
plt.title("Popularity of programming language\n" + "Worldwide, oct 2017 compared to a year ago")
plt.xticks(x_pos, x)
#turn on the grid
plt.minorticks_on()
plt.grid(which='major', linestyle='-', linewidth='0.5', color='red')
plt.show()
##############################################################################
'''
Write a python programming to display a horizontal  bar chart
'''
import matplotlib.pyplot as plt
x=['Java', 'Python', 'PHP', 'Javascript', 'C#', 'C++']
popularity=[22.2,17.6,8.8,8,7.7,6.7]
x_pos=[i for i, _ in enumerate(x)]
plt.barh(x_pos, popularity, color='green')
plt.xlabel('Popularity')
plt.ylabel('Languages')
plt.title("Popularity of programming language\n" + "Worldwide, oct 2017 compared to a year ago")
plt.yticks(x_pos, x)
#turn on the grid
plt.minorticks_on()
plt.grid(which='major', linestyle='-', linewidth='0.5', color='red')
plt.show()
################################################
#write a python programming to display a bar graph in uniform color
import matplotlib.pyplot as plt
x=['Java', 'Python', 'PHP', 'Javascript', 'C#', 'C++']
popularity=[22.2,17.6,8.8,8,7.7,6.7]
x_pos=[i for i, _ in enumerate(x)]
plt.bar(x_pos, popularity, color=['green','red','black','pink','blue','yellow'])
plt.xlabel('Popularity')
plt.ylabel('Languages')
plt.title("Popularity of programming language\n" + "Worldwide, oct 2017 compared to a year ago")
plt.xticks(x_pos, x)
#turn on the grid
plt.minorticks_on()
plt.grid(which='major', linestyle='-', linewidth='0.5', color='red')
plt.show()
#####################################################################
'''Histogram:-to understand normal distribution of data'''
#std.deviation:-distributed mean
import matplotlib.pyplot as plt
blood_sugar=[113,85,90,150,149,88,93,115,135,80,77,82,129]
plt.hist(blood_sugar, rwidth=0.8) #by default no. of bins is set to 10
plt.hist(blood_sugar,rwidth=0.5,bins=4)
'''
Histogram showing normal,pre-diabetic and diabetic patient distribution
 80-100:Normal
 100-125:Pre-diabetic
 125 onwards:diabetic
'''
plt.xlabel('Sugar level')
plt.ylabel('No. of patient')
plt.hist(blood_sugar, bins=[80,100,125,150], rwidth=0.95, color='green' )
######################################################################
'''
Boxplot
'''
#import libraries
import matplotlib.pyplot as plt
import numpy as np
#creating dataset
np.random.seed(10)
data= np.random.normal(100, 20, 200)
fig=plt.figure(figsize =(10,7))
#creating plot
plt.boxplot(data)
#show plot
plt.show()
##############################################################
import matplotlib.pyplot as plt
import numpy as np
#creating dataset
np.random.seed(10)
data1= np.random.normal(100, 20, 200)
data2= np.random.normal(90, 20, 200)
data3= np.random.normal(80, 20, 200)
data4= np.random.normal(70, 40, 200)
data=[data1,data2,data3,data4]
fig=plt.figure(figsize =(10,7))
#creating axes instance
ax=fig.add_axes([0,0,1,1])
#creating plot
bo=ax.boxplot(data)
#show plot
plt.show()
###########################################################
'''
29-04-2024
'''
import seaborn as sns
import pandas as pd
sales=pd.read_excel("C:/1-python/cars.csv")
sales.head()
sales.columns

cars=pd.read_csv("C:/1-python/cars.csv")
cars.columns
sns.relplot(x='HP',y='MPG',data=cars)
sns.relplot(x='HP',y='MPG',data=cars,kind='line')

sns.relplot('Sales','Profit',data=sales)
sns.relplot('Sales','Profit',data=sales,hue='Order Date')
sns.relplot('Order Date','Sales',data=sales,kind='line','Profit')

#####################3
sns.catplot(x='HP',y='MPG',data=cars,kind='box')

sns.catplot(x='Product Category',y='Sales',data=sales)

#Histogram
sns.ditplot(cars.HP)

##############################################################
#Multiple correlation regression analysis
import pandas as pd
import numpy as np
import seaborn as sns
cars=pd.read_cas("C:/1-python/cars.csv")
cars.describe()
#Exploratory data analysis
#1.Measure the central tendency
#2.Measure the dispersion
#3.Third moment business decision
#4.Fourth moment business decision
#5.Probability distribution
#6.Graphical representation(box,histogram)
import matplotlib.pyplot as plt
import numpy as np
plt.bar(height=cars.HP,x=np.arange(1,82,1))
sns.distplot(cars.HP)
#data is right skewed
plt.boxplot(cars.HP)
##There are several outliers in HP columns
#similar operations are expected for other three columns
sns.distplot(cars.MPG)
#data is slightly left distributed
plt.boxplot(cars.MPG)
##There is no outliers
sns.distplot(cars.VOL)
#data is slightly left distributed
plt.boxplot(cars.VOL)
sns.distplot(cars.SP)
#data is slightly right distributed
plt.boxplot(cars.SP)
###there several outliers
sns.distplot(cars.WT)
plt.boxplot(cars.WT)
#there several outliers
#now let us plot joit plot, joint plot is to show scatter
#histogram
import seaborn as sns
sns.jointplot(x=cars['HP'],y=cars['MPG'])

#now let us plot count plot
plt.figure(1,figsize=(16,10))
sns.countplot(cars['HP'])