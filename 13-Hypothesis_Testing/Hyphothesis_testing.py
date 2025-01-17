# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 08:26:50 2025

@author: ACER
"""
import pandas as pd
import numpy as np
import scipy
from scipy import stats
#Provides statistical functions;
#stats contains a variety of statistical tests.
from statsmodels.stats import descriptivestats as sd
#Provides descriptive statistics tools, including the sign_test.
from statsmodels.stats.weightstats import ztest
#Used for conducting z-tests on datasets.

#1 sample sign test
#Whenever there is a single sample and data is not normal.
marks=pd.read_csv("C:/DataScience_Dataset/Signtest.csv")
marks

#Normal QQ plot
import pylab
stats.probplot(marks.Scores, dist='norm', plot=pylab)
#Creates a QQ plot to visually check if the data follows a normal distibution
#Test for normally
shapiro_test=stats.shapiro(marks.Scores)
#Perform the shapiro: Wilk test for normality.
#H0 (null hyphothesis): The data is normally distrubuted.
#H1(alternative hyphothesis): The data is not normally distrubuted.
#Outputs a test statistics and p-value.
print("Shapiro Test:", shapiro_test)
#p-value is 0.024<0.05, data is not normal.

#Describtive Statistics
print(marks.Scores.describe())
#mean=84.20 and median=89.00
#1-Sample sign test
sign_test_result=sd.sign_test(marks.Scores, mu0=marks.Scores.mean())
print("Sign Test Result",sign_test_result)
#Result: p-value=0.82

#Interpretation==>
#H0: The median of Scores is equal to the mean of Scores.
#H1: The median of scores is not equal to the mean of scores.
#Since the p-value (0.82) is greater than 0.05, we fail to reject the null hyphothesis.
#Conclusion:> The median and mean of Scores are statistically not similar.


#######################################################################################
'''
   Z-Test
'''
# 1-Sample z-test
fabric=pd.read_csv("C:/DataScience_Dataset/Fabric_data.csv") 

#Normality Test
fabric_normality=stats.shapiro(fabric)
print("Fabric Normality Test:", fabric_normality)
#p-value=0.1460>0.05


#Fabric mean
fabric_mean= np.mean(fabric)
print("Mean Fabric Length:", fabric_mean)

#Z-test
z_test_result, p_val= ztest(fabric['Fabric_length'], value=150)
print("Z-test Result:", z_test_result, "P-value:",p_val)
#Result: p-value=7.15 x 10^8
'''
#Interpretation==>
#H0: The median of Scores is equal to the mean of Scores.
#H1: The median of scores is not equal to the mean of scores.
#Since the p-value (0.82) is greater than 0.05, we fail to reject the null hyphothesis.
#Conclusion:> The median and mean of Scores are statistically not similar.
'''
#######################################################################
'''
     MANN-WHITNEY TEST
'''
#Mann-Whitney test
fuel=pd.read_csv("C:/DataScience_Dataset/mann_whitney_additive.csv")
fuel.columns=["Without_additive", "With_additive"]

#Normality Test
print("Without Additive Normality:", stats.shapiro(fuel.Without_additive))
#p=0.50>0.05: accept H0
print("With Additive Normality:", stats.shapiro(fuel.With_additive))
#0.04<0.05: reject H0 data is not normal
#Mann-Whitney U test
mannwhitney_result=stats.mannwhitneyu(fuel.Without_additive, fuel.With_additive)
print("Mann-Whitney Test Result:", mannwhitney_result)
#Result: p-value=0.445
#Interpretation:
#H0: No difference in performance between without_additive and With_additive.
#H1: A significant difference exists.
#Since the p-value (0.445) is greater than 0.05, we fail to reject the null hyphothesis.
#COnclusion: Adding fuel additive does not significantly impact performance.
#Apply tha mannwhitney U test to check if there a significant diff between
#H0: No diff in performnace between the 2 groups.
#H1: significant diff in performance.
##############################################################################
'''
      PAIRED T-TEST
      
Objective: Check whether there is diffrence in transaction time of supplier_A
and Supplier_B
'''
sup=pd.read_csv("C:/DataScience_Dataset/paired2.csv")

#Normality Tests
print("Supplier A Normality Test:", stats.shapiro(sup.SupplierA))
#pvalue=0.8961992859840393> 0.05: fails to reject the H0, data is normal
print("Supplier B Normality Test:", stats.shapiro(sup.SupplierB))
#pvalue=0.8961992859840393> 0.05: fails to reject the H0, data is normal
#Paired T-test
t_test_result,p_val=stats.ttest_rel(sup['SupplierA'], sup['SupplierB'])
print("Paired T-test Result:", t_test_result, "p-value:",p_val)
#Result: p-val=0.00
#Interpretation:
#H0: No significant diff in transaction times between Supplier_A and Supplier_B
#H1: Significant Difference exists
#since the p-value (0.00) is less than 0.05, we reject the null hyphothesis.
#Conclusion: There is a significant difference in transaction times between
#objective: is there significant difference between two promotional objective.
############################################################################
'''
      Two-Sample T-Test
'''
offers=pd.read_excel("C:/DataScience_Dataset/Promotion.xlsx")
offers.columns=["InterestRateWaiver","StandardPromotion"]

#Normal Tests
print("InterestWaiver Normality:",stats.shapiro(offers.InterestRateWaiver))
print("StandardPromotion Normality:",stats.shapiro(offers.StandardPromotion))

#Variance Test
levene_test=scipy.stats.levene(offers.InterestRateWaiver, offers.StandardPromotion)
print("Levene Test (Variance):",levene_test)
#p-value=0.2875
#H0=variance equal
#H1=variance unequal
#pvalue=0.2875>0.05 fail to reject null hyphothesis (H0 is accepted)

#Two sample test
ttest_result=scipy.stats.ttest_ind(offers.InterestRateWaiver, offers.StandardPromotion)
print("Two-Sample T-Test:",ttest_result)
#Result=p-value:
#H0= both offers have same mean impact
#H1= The mean impacts of the two offer are different.
#Since the p-value (0.0242) is less than 0.05, we reject the null hyphothesis.
#Conclusion: There is a significant difference between two offers.
#################################################################################
'''
    Mood's Median Test
'''
#Objective--> Is the medians of Pooh, piglet and tigger are statistically equal, it has equal medians or not.
animals=pd.read_csv("C:/DataScience_Dataset/animals.csv")
animals
animals["Pooh"].median()
animals["Piglet"].median()
animals["Tigger"].median()
#Normality Tests
print("Pooh Normality:",stats.shapiro(animals.Pooh))
#p-value=0.0122
print("Piglet Normality:",stats.shapiro(animals.Piglet))
#p-value=0.044
print("Tigger Normality:",stats.shapiro(animals.Tigger))
#p-value=0.0219
#H0:data is normal
#H1: data is not normal
#Since all p value are less than 0.05 hence reject the null hyphothesis
#Data is not normal, hence Mood's test
#Medians Test
median_test_result=stats.median_test(animals.Pooh,animals.Piglet,animals.Tigger)
print("Mood's Median Tests:",median_test_result)
#Result: p-value=0.186
#Interpretation:
#Ho: all groups have equal medians.
#H1: At least one group have different medians.
#Since the p-value (0.186) is greater than 0.05, we reject the null hyphothesis.
###########################################################################################
'''
   One Way Anova Test
'''
#Objective: is the transaction times for the three suppliers are not significantly different.
contract=pd.read_excel("C:/DataScience_Dataset/ContractRenewal_Data(unstacked).xlsx")
contract.columns=["Supp_A","Supp_B","Supp_C"]

#Normality Tests
print("Supp_A Normality:",stats.shapiro(contract.Supp_A))
print("Supp_B Normality:",stats.shapiro(contract.Supp_B))
print("Supp_C Normality:",stats.shapiro(contract.Supp_C))
#ALl p-values are greater than 0.05
#We fail to reject the null hyphothesis.
#i.e H0 is accepted means data is normal.
#Variance Test
levene_test=scipy.stats.levene(contract.Supp_A,contract.Supp_B,contract.Supp_C)
print("Levene Test(Variance):",levene_test)
#H0: data is equal variance
#H1: data have different in variance
#p-value=0.775>0.05, H0 is accepted

###########ANOVA TEST##########
annova_result=stats.f_oneway(contract.Supp_A,contract.Supp_B,contract.Supp_C)
print("One way ANNOVA:",annova_result)
#Result: p-value=0.104
#Interpretation:
#H0: all suppliers have the same mean transaction time.
#H1: At least one supplier has a different mean.
#Since the p-value(0.104) is greater tha 0.05, we fail to rejects it is null hyphothesis.
#Conclusion: The transaction times for the three suppliers are not significantly different.
####################################################################
#16-01-2025
'''
   Two-Proportion Z-Test
'''
#Use a two-sided test when you want to detect a difference without
#assuming beforehand which group will have a higher or lower proportion.
#Example: Testing if there is diff in soft drink consumption between
#adults and children without assuming which group consumes more.
#Objective: there is a significant in soft drink consumption between adults and children.
soft_drink=pd.read_excel()
from statsmodels.stats.proportion import proportion_ztest

#Data Preparation
count=np.array([58,152])
nobs=np.array([480,740])
#The two proportion Z-test compares the proportions of groups. 
#Here: count=[58,152]: The no. of success
#(people consuming soft drinks) in each group.(adult and children)
#Similarly, if 740 children were surveyed and 152 of them reproted
#consuming soft drinks, the second count is 152.
#Thus, count=[58,152]

#Nobs: Represents the total number of individuals surveyed in each group.

#The total number of adults surveyed is 480.
#The total number of children surveyed is 740.
#Hence, nobs=[480,740]

#These values are often extracted from a dataset.
#if your data is in a file (like "JohnyTalkers.xlsx"),
#you can calculate these values as follows:

import pandas as pd
from statsmodels.stats.proportion import proportions_ztest

#load dataset
soft_drink_data=pd.read_excel("C:/DataScience_Dataset/JohnyTalkers.xlsx")
soft_drink_data

#filter the data into children and adults categories
adults= soft_drink_data[soft_drink_data['Person']=='Adults']
children=soft_drink_data[soft_drink_data['Person']=='Children']

#count of success (soft drink consumers)for each group
count_adults=adults[adults['Drinks']=='Purchased'].shape[0]
count_children=children[children['Drinks']=='Purchased'].shape[0]

#Total observation for each griup
nobs_adults=adults.shape[0]
nobs_children=children.shape[0]

#Final arrays for z-test
count=[count_adults,count_children]
nobs=[nobs_adults,nobs_children]

print('Count(soft drink consumers: ',count)
#o/p-->Count(soft drink consumers:  [58, 152]
print('Total observation: ',nobs)
#o/p-->Total observation:  [480, 740]

# Two-sided test
z_stat, p_val = proportions_ztest(count, nobs, alternative = 'two-sided')
print("Two-sided Proportions Test:", z_stat,"P-Value:", p_val)

# Result: p-value = 0.000
# Interpretation:
# H0: Proportions of adults and children consuming the soft drink are the same.
# H1: Proportions are different.
# Since the p-value (0.000) is less than 0.05 we reject the null hypothesis
# Conclusion: There is a significant difference in soft drink
# consumption
################################################################
'''
   Chi-Square Test
'''
#Objective--> is defective proportions are independent of the country?
#The dataset contains two columns:
    
#Defective: Indicated whether an item is defective (likely binary, with 1 for defective
# and 0 for not defective).
#Country: Specifies the country associated with the item(e.g., "India")
#The dataset has 800 entries, and there are
#no missing values in either column. It appears to be designed  to analyze
#defect rates across different countries,
#WHich aligns with the chi-square test you performed earlier
#to determine if defectivness is independent of the country.

Bahaman=pd.read_excel("C:/DataScience_Dataset/Bahaman.xlsx")

#Crosstabulation
count=pd.crosstab(Bahaman["Defective"], Bahaman["Country"])
count

#Chi-square test
chi2_result=scipy.stats.chi2_contingency(count)
print("Chi-Square Test:",chi2_result)
'''
o/p-->
Chi-Square Test: Chi2ContingencyResult(statistic=1.7243932538050184, pvalue=0.6315243037546223, dof=3, expected_freq=array([[178.75, 178.75, 178.75, 178.75],
       [ 21.25,  21.25,  21.25,  21.25]]))
'''
####################################################################



    
