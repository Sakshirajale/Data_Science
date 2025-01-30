# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 09:17:45 2025

@author: ACER
"""
import pandas as pd
walmart=pd.read_csv("C:/DataScience_Dataset/Walmart Footfalls Raw.csv")
month=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']
#In walmart data we have Jan-1991 in 0th column, we need only first three letters
#example- Jan from each cell
p=walmart["Month"][0]
p[0:3]  #o/p: 'Jan'
# before we will extract, let us create new column called months to store extracted values
walmart['month']=0
#you can check the dataframe with months name with all values 0
#the total records are 159 in walmart
for i in range(159):
    p=walmart["Month"][i]
    walmart["month"][i]=p[0:3]
    #for all these months create dummy variables
month_dummies=pd.DataFrame(pd.get_dummies(walmart['month']))
#now let us concatenate these dummy values to dataframe
walmart1=pd.concat([walmart,month_dummies],axis=1)
#you can check the dataframe walmart1

#similarly we need to create column
import numpy as np
walmart1['t']=np.arange(1,160)
walmart1['t_squared']=walmart1['t']*walmart1['t']
walmart1['log_footfalls']=np.log(walmart1['Footfalls'])
walmart1.columns
'''
O/P-->
Index(['Month', 'Footfalls', 'month', 'Apr', 'Aug', 'Dec', 'Feb', 'Jan', 'Jul',
       'Jun', 'Mar', 'May', 'Nov', 'Oct', 'Sep', 't', 't_squared',
       'log_footfalls'],
      dtype='object')
'''
#Now let us check the visuals of the football
walmart.Footfalls.plot()
#You will get exponential tren with first decreasing and then increasing
#we have to forecast footfalls in next 12 months, hence horizon=12, even
#season=12, so validating data will be 12 and training will 159-12=147
Train=walmart1.head(147)
Test=walmart1.tail(12)
#Now let us apply linear regression
import statsmodels.formula.api as smf
##Linear model
linear_model=smf.ols("Footfalls~t",data=Train).fit()
pred_linear=pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear=np.sqrt(np.mean((np.array(Test['Footfalls'])-np.array(pred_linear))**2))
rmse_linear
#O/P-->209.92559265462546

##Exponetial model
Exp_model=smf.ols("log_footfalls~t",data=Train).fit()
pred_Exp=pd.Series(Exp_model.predict(pd.DataFrame(Test['t'])))
rmse_Exp=np.sqrt(np.mean((np.array(Test['Footfalls'])-np.exp(pred_Exp))**2))
rmse_Exp
#O/P--> 217.05263566813173

##Quadratic model
Quad=smf.ols("Footfalls~t+t_squared",data=Train).fit()
pred_Quad=pd.Series(Quad.predict(pd.DataFrame(Test[["t","t_squared"]])))
rmse_Quad=np.sqrt(np.mean((np.array(Test['Footfalls'])-np.array(pred_Quad))**2))
rmse_Quad
#O/P--> 137.15462741356484
##############################################################################
################ Additive Seasonality ###################################

add_sea=smf.ols('Footfalls ~ Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
add_sea.summary()
pred_add_sea=pd.Series(add_sea.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']]))
rmse_add_sea=np.sqrt(np.mean((np.array(Test['Footfalls'])-np.array(pred_add_sea))**2))
rmse_add_sea
#O/P-->264.6643900568774
################################################################################
############## Multiplicative Seasonality model ###############################
mul_sea=smf.ols("log_footfalls~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov", data=Train).fit()
pred_mul_sea=pd.Series(mul_sea.predict(Test))
rmse_mul_sea=np.sqrt(np.mean((np.array(Test['Footfalls'])-np.exp(pred_mul_sea))**2))
rmse_mul_sea
#o/p=268.197032530917
#############################################################################
################## Additive Seasonality with trend ######################
add_sea_quad=smf.ols('Footfalls~t+t_squared+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_add_sea_quad=pd.Series(add_sea_quad.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','t','t_squared']]))
rmse_add_sea_quad=np.sqrt(np.mean((np.array(Test['Footfalls'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad
#o/p=50.60724584048495
####### Multiplicative seasonability linear model
mul_add_sea=smf.ols("log_footfalls~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov",data=Train).fit()
pred_mul_add_sea=pd.Series(mul_add_sea.predict(Test))
rmse_mul_add_sea=np.sqrt(np.mean((np.array(Test['Footfalls'])-np.array(np.exp(pred_mul_add_sea)))**2))
rmse_mul_add_sea
#O/P-->172.7672678466982

#let us create a dataframe and add all these rmse_values
data={"Model":pd.Series(['rmse_linear','rmse_Exp','rmse_Quad','rmse_add_sea','rmse_mul_sea','rmse_add_sea_quad','rmse_mul_add_sea']),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_mul_sea,rmse_add_sea_quad,rmse_mul_add_sea])}
data
'''
O/P-->
{'Model': 0          rmse_linear
 1             rmse_Exp
 2            rmse_Quad
 3         rmse_add_sea
 4         rmse_mul_sea
 5    rmse_add_sea_quad
 6     rmse_mul_add_sea
 dtype: object,
 'RMSE_Values': 0    209.925593
 1    217.052636
 2    137.154627
 3    264.664390
 4    268.197033
 5     50.607246
 6    172.767268
 dtype: float64}
'''

#Now let us test the model with full data
predict_data=pd.read_excel("C:/DataScience_Dataset/Predict_new.xlsx")
model_full=smf.ols('Footfalls ~t+t_squared+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=walmart1).fit()

pred_new=pd.Series(model_full.predict(predict_data))
pred_new
'''
O/P-->
0     2193.807626
1     2229.969736
2     2200.670308
3     2311.293957
4     2356.071452
5     2036.848947
6     2187.241826
7     2181.480859
8     2234.104508
9     1999.997498
10    1972.995363
11    2280.493228
dtype: float64
'''
predict_data["forecast_Footfalls"]=pd.Series(pred_new)
predict_data["forecast_Footfalls"]
'''
O/P-->
0     2193.807626
1     2229.969736
2     2200.670308
3     2311.293957
4     2356.071452
5     2036.848947
6     2187.241826
7     2181.480859
8     2234.104508
9     1999.997498
10    1972.995363
11    2280.493228
Name: forecast_Footfalls, dtype: float64
'''
########################################################################






