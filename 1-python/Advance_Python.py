#####################  DAY 4 #######################
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 15:00:56 2024

@author: ACER
"""
##############ADVANCE PYTHON################
        
####list comprehension####
list=[]
for num in range(0,20):
    list.append(num)
print(list)
######################################
##by list comprehension
list=[num for num in range(0,20)]
print(list)
######################################
#capitalize list
names=["dada","mama","kaka"]
list=[name.capitalize() for name in names]
print(list)
######################################
#comprehension with if statement
def is_even(num):
    return num%2==0
list=[num for num in range(21) if is_even(num)]
print(list)
####################################
#comphrension with nested for loop 
list=[f"{x}{y}"for x in range(3) for y in range(3)]
print(list)
#########################################
##dict comprehension
dict={x:x*x for x in range(3)}
print(dict)
########################################################
###Generator
'''It is another way of creating iterators in a simple way
where it uses the key word "yield".Instead of returning
it in a defined function.
Generators are implemented using a function'''

#generator for all values
gen=(x for x in range(3))
print(gen)
for num in gen:
    print(num)
############################################
#generator for single values
gen=(x for x in range(3))
next(gen)
next(gen)
#####################################
##functions which return multiple values
def range_even(end):
    for num in range(0,end,2):
        yield num
for num in range_even(6):
    print(num)
###########################################################    
#now instead of usimg for loop we can write create generators
gen=range_even(6)
next(gen)
next(gen)
    
###chaining generators
def lengths(itr):
    for ele in itr:
        yield len(ele)
def hide(itr):
    for ele in itr:
        yield ele*"*"
password=["not good","give m-pass"]    
for password in hide(lengths(password)):
    print(password)
    
        
'''
"ele*" appears to be a placeholder for an element
from an iterable.The asterisk(*) is likely just a character
used to represent a placeholder or a wildcard.
for instance if u r iteratngover a lst of elemts,"ele*"
could symbolize any representation that does not  correspond to any spevific syntax
in python or itertolls'''


##take password from user and hide it
adj=input("Enter an adj:")
noun=input("Enter a noun:")
number=input("Enter a number:")
sc=input("Enter a special character:")
password=adj+noun+str(number) +sc
print("Your password is: %s"%password) 
def lengths(itr):
    for ele in itr:
        yield len(ele)
def hide(itr):
    for ele in itr:
        yield ele* "*"
password=adj+noun+str(number)+sc    
for password in hide(lengths(password)):
    print(password,end="") 
######################################################
#Enumerate 
#printing list with index
lst=["milk","Egg","Bread"]
for index in range(len(lst)):
    print(f'{index+1} {lst[index]}')
##################
#same code can be implemented using enumerate
lst=["milk","Egg","Bread"]
for index,item in enumerate(lst,start=1):
    print(f'{index} {item}')
#############################################
#use of zip function
name=['dada','mama','kaka']
info=[9850,6032,9785]
for nm,inf in zip(name,info):
    print(nm,inf)
#############################################
#use of zip function with mis match list
name=['dada','mama','kaka','baba']
info=[9850,6032,9785]
for nm,inf in zip(name,info):
    print(nm,inf)
#it will not display excess mismatch item in name i.e. baba
##########################
#zip longest
from itertools import zip_longest
name=['dada','mama','kaka','baba','nana']
info=[9850,6032,9785]
for nm,inf in zip_longest(name,info):
    print(nm,inf)
##################################
#use of fill value instead none
from itertools import zip_longest
name=['dada','mama','kaka','baba']
info=[9850,6032,9785]
for nm,inf in zip_longest(name,info,fillvalue=0):
    print(nm,inf)
###########################
#use all(), if all the values are true then it will produce output
lst=[2,3,-5,7,-8] #value must be non zero, +ve or -ve
if all(lst):
    print('All values are true')
else:
    print('There are null values')
#####################################
lst=[2,3,0,8,9]
if all(lst):
    print('All values are true')
else:
    print('There are null values')
###########################################
#use of any if any non zero values
lst=[0,0,0,-8,0]
if all(lst):
    print('It has some zero value')
else:
    print('Useless')
###########################################
#use of any
lst=[0,0,0,0,0]
if all(lst):
    print('It has some  value')
else:
    print('All values are null in the list')
##################################################
#count()
from itertools import count
counter=count()
print(next(counter))
print(next(counter))
print(next(counter))
######################################
#now let us start from 1
from itertools import count
counter=count(start=1)
print(next(counter))
print(next(counter))
print(next(counter))
###############################
#cycle()
#suppose you have repeated tasks to be done, then you 
import itertools

instuctions=("Eat","Code","Sleep")
for instuction in itertools.cycle(instuctions):
    print(instuction)
################################
#repeat()
from itertools import repeat
for msg in repeat("keep patience",times=5):
    print(msg)
#########################
#combinations(),combination means ex. picking ball for box
from itertools import combinations
players=['John','Jani','Jay','Sai']
for i in combinations(players,3):
    print(i)
#####################################
#permutations
from itertools import permutations
players=['John','Jani','Jay']
for seat in permutations(players,2):
    print(seat)
############################################
#product() ##pair to two list
from itertools import product
team_a=['Rohit','Pandya','Bumrah']
team_b=['Virat','Manish','Sami']
for pair in product(team_a,team_b):
    print(pair)
##############################
age=[27,17,21,19]
adults=filter(lambda age:age>=18,age)
print([age for age in adults])
#####################################
###***Important for interview***###
#shallow copy and deep copy
'''
In python, assignment statements (obj_b=obj_a)
do not create real copies.
It only creates a new variable with the same reference.
So when you want to make actual copies of mutable objects(lists,acts)
and want to modify the copy without affecting the original, you have to be careful.
For 'real' copies we can use copy module.
However,for compound/nested objects(ex. nested list or dictionary)
and custom objects there is an important difference between shallow and deep copying:
-shallow copies:only one level deep. It creates a new collection and populates it with references to the nested objects.
this means modyfing a nested object in the copy deeper than
-deep copies:A full independent clone. It creates a new collection and then recursively populates it with copies of the nested objects.
'''
##it change object name not change value from this object
list_a=[1,2,3,4,5]
list_b=list_a
list_a[0]=-10
print(list_a)
print(list_b)
######################################################################
###################### DAY 5 ####################
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 08:15:32 2024

@author: ACER
"""

#shallow copy
#one level deep. Modifying on level 1 does not affect
#use copy.copy(), or object specific copy functions
import copy
list_a=[1,2,3,4,5]
list_b=copy.copy(list_a)
#not affects the other list
list_b[0]=-10
print(list_a)
print(list_b)
#########################################
#but with nested objects, modifying on level 2 or deep
list_a=[[1,2,3,4,5],[6,7,8,9,10]]
list_b=copy(list_a)
#affects the other!
list_b[0][0]=-10
print(list_a)
print(list_b)
#########################################
#Deep copy
#Full independent clones. Use copy.deepcopy().
import copy
list_b=copy.deepcopy(list_a)
#not affects the other!
list_b[0][0]=-10
print(list_a)
print(list_b)
########################################

'''
Reading data in various format
'''
import pandas as pd
f1=pd.read_csv('C:/1-python/buzzers.csv.xls')
f1
########################################################################




