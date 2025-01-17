# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 09:02:42 2024

@author: ACER
"""

'''
1. Check if email address valid or not in Python 
e.g. Input: my.ownsite@ourearth.org 
Output: Valid Email 
Input: ankitrai326.com 
Output: Invalid Email  
'''
char=input("enter your char:")
name=input("enter your name:")
string=input("enter your string:")
email=(char+"."+name+"@"+string+".org")
email
if email=="%s.%s@%s.org":
    print("Valid email")
else:
    print("Invalid email")

'''2. Write a Python program to find the median of below three values. 
Values: (25,55,65) 
'''
a=25
b=55
c=65
median=(a+b+c)/3
print(median)
'''
3. Write a program to create a decorator function to measure the 
execution time of a function. 
'''

'''
4. Write a python program that opens a file and handles a 
FileNotFoundError exception if the file does not exist.
'''
import pandas as pd
df=pd.read_csv("C:/1-python/buzzers.csv.xls")
df
'''
5. Write a python program to find the intersection of two given arrays 
using Lambda. 
Original arrays: 
[1, 2, 3, 5, 7, 8, 9, 10] 
[1, 2, 4, 8, 9] 
Intersection of the said arrays: [1, 2, 8, 9]
'''
a=[1,2,3,5,7,8,9,10]
b=[1,2,4,8,9]
s1=



