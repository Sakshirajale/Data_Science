
######  4/10/2024  ########

import numpy as np
#let's define a true value that we want to measure 
#we first define the value of 50 this is the reference point
true_value=50

#simulate some measurements
#Accurate but not precise(close to true value but spread out)

'''
Simulating measurements
we simulate two sets of measurements:
    Accurate but not precise:Thses values are centered ariund the true
    value(50),but there is some more spread(random variation),
    this simulates measurements that are accurate(close to  the true value)but not precise(spread out)
'''

accurate_measurements = np.random.normal(loc= true_value, scale=5, size=10)

#precise but not accurate(far from true value but tightly grouped)
'''
precise but not accuarate:
    these values are tightly clustered around 60,not near the true value of 50.
    This simulates measurements that are precise(closely grouped)
    but not accurate(far from true value).
'''
precise_measurements = np.random.normal(loc=60, scale=1, size=10)

#function to calculate accuracy

'''
We calculate the mean(avg) of the measurements.
Then we cslculate how close this avg is to the true value.
The closer the avg to the true value,==>higher accuaracy
The accuracy==>1-(difference/true_value)

Thid gives a number between 0(low accuarcy) and 1(higher accuracy)
'''

def calculate_accuracy(measurements,true_value):
    #Accuracy:how close this avg is to the true value.
    avarage_measurement=np.mean(measurements)
    accuracy=1-abs(avarage_measurement-true_value)/true_value
    return accuracy
    
#Function to calculate precision
'''
precision is calculate by the std deviation of the measurements.Std deviation measures
the how spread out the measurements are.
If STD is small==>mesurements are closer together
precision will be high.
we use 1/std_dev to represent precision,
so a smaller spread gives a higher value for precision
'''    

def calculate_precision(measurements):
    #precision:How close  the memasurements are to each other
    #low std deviation==>high precision
    precision=1/np.std(measurements)  #higher std deviation
    #means lower precision, so we invert it
    return precision

#calculate accuracy and precisiom for both sets
'''
Accuarate measurements:We calculate the accuracy and precision
of the measurements that are close the true value but spread out
precise measurements:We calculate the accuracy and precision
of the measurements that are closely grouped but far from the true value
'''
accuracy_of_accurate = calculate_accuracy(accurate_measurements, true_value)
precision_of_accurate = calculate_precision(accurate_measurements)

accuracy_of_precise = calculate_accuracy(precise_measurements, true_value)
precision_of_precise = calculate_precision(precise_measurements)

#print the result
print('Accurate but not precise Measurements:')
print(f"measuremetns: {accurate_measurements}")
print(f"Accuracy: {accuracy_of_accurate:.2f}")
print(f"Precision: {precision_of_accurate:.2f}")

print("\nprecise but not precise measurements ")
print(f"measurements:{precise_measurements}")
print(f"accuracy:{accuracy_of_precise:2f}")
print(f"precision:{precision_of_precise:2f}")


######  9/10/2024   #######

#more STD ==>less precision

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Target value(True value)
true_value=50
#simulate data
#1. Accuarte and precise(Close to true value and Tightly grouped)
'''
loc=true_value(true value=50):The values  will be centered around the true value(50)
scale 1=STD (spread) is small,
meaning the values are tightly grouped around the true value.
This implies high precision
The measurements will vary only a little from the true value, so they will be 
both accuarte(close to 50)and precise (close to each other)
'''

accurate_precise=np.random.normal(loc=true_value,scale=1,size=10)
accurate_precise

#Accuarate but not precise(close to true value but spread out)
accurate_not_precise=np.random.normal(loc=true_value,scale=10,size=10)
accurate_not_precise

'''
The two lines of code you have highlighted may look similar, 
but they differ in one important aspect:the value of scale,
which controls the spread of the generated values raound the true value(loc)
'''

#3.Precise but not accurate(far from true value but tightly grouped)
precise_not_accurate=np.random.normal(loc=70,scale=1,size=10)
precise_not_accurate

#4.Nighter accurate nor precise(far from ltrue value and spread out)
not_accurate_nor_precise=np.random.normal(loc=70,scale=1,size=10)
not_accurate_nor_precise

#plotting the results
plt.figure(figsize=(10,6))

#Plot 1:Accurate and precise
plt.scatter(accurate_precise,[1]*10,color="green",label="Accurate And Precise")

#plot 2:Accurate_not_precise
plt.scatter(accurate_not_precise,[2]*10,color="blue",label="accurate_not_precise")

#plot 3:precise_not_accurate
plt.scatter(precise_not_accurate,[3]*10,color="orange",label="precise_not_accurate")

#plot4:not accurate_nor_precise
plt.scatter(not_accurate_nor_precise,[4]*10,color="red",label="not_accurate_nor_precise")


#Adding target line
plt.axvline(true_value,color='black',linestyle='--',label='True Value')

#labels and legend
plt.yticks([1,2,3,4],['Accurate and Precise','Accurate but not precise','precise but not accurate','not accurate not precise'])
plt.xlabel('Measurement Value')
plt.legend()
plt.title('Accuracy and Demonstration')
plt.show()

#Instance Based Classifiers


######    11/10/2024     ######




















