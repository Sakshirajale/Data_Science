# -*- coding: utf-8 -*-
################### DAY 1 ##############
################
x="Hello World"
string1=x[::-2]
print(string1)

#find method,Searchers the string for a specified value at
x="This is pyhton and it is very powerful"
print(x.find("very"))
###################
#string concateness
x="Hello"
y="word"
print(x+y)
##################
#to add white space
print(x+" "+y)
##################
#string format
x=36
y="my name is Anthony"
print(x+y)
#it will give error
print(f"my name is anthony and my age is {x}")
######################
quantity=3
item_no=54
price=67
print(f"I want {quantity} pieces and item number is {item_no}, its price is {price}")
#######################################################################

my_order="I want {} pieces and item number is {}, its price is {}"
print(my_order.format(quantity,item_no,price))
##########################################3
quantity=3
item_no=54
price=67
my_order="I want {0} pieces and item number is {1}, its price is {2}"
print(my_order.format(quantity,item_no,price))
#############################################
#The escap character allows you to use double quotes when
text="This is fun fair and it has got big \"round rigo\""
#text="This is fun fair and it has got big "round rigo""
print(text)
######################################
a=10
b=20
print(a!=b)
#############################
#operator precedence
print(3*3+3/3-3)
"""
Rule for mathematical operations
PEMDAS
p-parenthesis
e-
m-multiplication
d-division
a-addition
s-subtration
"""
################################
#identify operators
print(a is b)
print(a is not b)
#########################
########################
#python list
lst=["cherry","banana","appple"]
print(lst)
############
#list items are indexed, the first item has index [0], the
print(lst[0])
print(lst[2])
#################
#append() adds a element at the end of the list
lst=["cherry","banana","appple"]
lst.append("Mango")
print(lst)
########################
#clear removes all element of list
lst=["cherry","banana","appple"]
lst.clear()
print(lst)
###################
#copy() list
lst=["cherry","banana","appple"]
lst2=lst.copy()
print(lst2)
######################
#count() return the number of times the value 
lst=["cherry","banana","appple"]
lst.count("cherry")
######################
#extend() add the elements of cars to the fruits
lst=[1,2,3]
lst1=[4,5,6]
lst.extend(lst1)
print(lst)
########################
#insert() method, insert the value "mango" as the
lst=["cherry","banana","appple"]
lst.insert(2,"Mango")
print(lst)
#################
#pop() removes the elements at the specified position
lst=["cherry","banana","appple"]
lst.pop(2)
print(lst)
#################
#remove() removes the item with the speciafied value
lst=["cherry","banana","appple"]
lst.remove("cherry")
print(lst)
##########################
#reverse()  list
lst=["cherry","banana","appple"]
lst.reverse()
print(lst)
############################
#sort() sort the list alphabetically
lst=["cherry","banana","appple"]
lst.sort()
print(lst)
########################
###########Tuple
tup=("cherry","cherry","banana")
print(tup)
print(tup[2])
########################
#once tuple is created, you can not change its value
x=("apple","banana","cherry")
x[1]='kiwi'
#first convert into list
y=list(x)
y[1]="kiwi"
#convert list to tuple
x=tuple(y)
print(x)
####################
x=("apple",2,"cherry")
print(x)
####################
#you can access tuple items by referring to the 
x=("apple","banana","cherry")
print(x[1])
######################
#addition of two tuple
tuple1= ("a", "b", "c")
tuple2= (1, 2, 3)
tup1=tuple1+tuple2
print(tup1)
##################
#dictionary
dic1={"Brand":["Maruti","Mahendra","Toyota"],"Model":["a","b","c"],"Year":[2011,2013,2022]}
print(dic1)
#######################
dic1={"Brand":"Maruti","Model":"a","Year":2011}
print(dic1)
print(len(dic1))
print(type(dic1))
##################
dic1.get("Model")
dic1.keys()
##########################################################################
###################  Day 2 ##########################
car = {"Brand":"Ford","Model":"Mustang","Year":1964}
x=car.keys()
print(x)
car["color"]="white"
car
x=car.keys()
print(x)
####################
#remove the element from dictionary
car = {"Brand":"Ford","Model":"Mustang","Year":1964}
car.pop("Model")
print(car)
#######################
#accessing keys in the dictionary
for x in car:
     print(car[x])
#if you want to access both keys and value
#Very important
for key,value in car.items():
    print("%s = %s" % (key,value))
#########################
#copying dictionary
car = {"Brand":"Ford","Model":"Mustang","Year":1964}
car2=car.copy()
car2
###########################
#another way to make is copy dictionary 
thisdict={"Brand":"Ford","Model":"Mustang","Year":1964}
dict1=dict(thisdict)
dict1
#########################
#A dictionary can contain dictionaries,
#this is called nested dictionaries.
our_family={"child1":{"Name":"Ram","DOB":"21-05-2008"},"child2":{"Name":"Sham","DOB":"01-01-2008"}}
our_family
################################
#dictionary methods
#clear():remove all elements from the car
car = {"Brand":"Ford","Model":"Mustang","Year":1964}
car.clear()
car
####
x=car.copy()
print(x)
#####################
#fromkey()
#create a dictionary wirh 3 keys, all with
x={'key1','key2','key3'}
y=0
thisdict=dict.fromkeys(x,y)
thisdict
##################
#get():to get value of dictionary
car = {"Brand":"Ford","Model":"Mustang","Year":1964}
car.get("Model")
######################
#items() return the dictionarys key value
car = {"Brand":"Ford","Model":"Mustang","Year":1964}
car.items()
##################
car.keys()
################
#pop() removes the specified key
car = {"Brand":"Ford","Model":"Mustang","Year":1964}
car.pop("Model")
car
#####################
#values():display all the values of dictinary
car = {"Brand":"Ford","Model":"Mustang","Year":1964}
car.values()
#######################
#update():insert an item to the dictionary
car = {"Brand":"Ford","Model":"Mustang","Year":1964}
car.update({"color":"white"})
car
############################
#for loop
fruits=["Apple","Banana","Cherry"]
for i in fruits:
    print(i)
#################################
#use of break statement
fruits=["Apple","Banana","Orange"]
for i in fruits:
    print(i)
    if(i=="Banana"):
        break
#here first item it will print apple,check 
#second time print banana check the condition
#now condition is true hence stops the print
fruits=["Apple","Banana","Orange"]
for i in fruits:
    if(i=="Banana"):
       break 
    print(i)
#############
fruits=["Apple","Banana","Cherry"]
for x in fruits:
    if x=="Banana":
        break
    print(x)
##################
#continue: with the continue statement we
fruits=["Apple","Banana","Cherry"]
for x in fruits:
    if x=="Banana":
        continue
    print(x)
####################
#from 2 to 6 (but not including 6)
for x in range(2,6):
    print(x)
##############
for x in range(2,30,3):
    print(x)
###################
#a nested loop is a loop inside a loop.
#the "inner loop" will executed one time
colors=["green","yellow","red"]
fruits=["guava","banana","apple"]
for x in colors:
    for y in fruits:
        print(x,y)
########################
def my_function():
    print("Hello from a function")
my_function()
#########################
def my_function(name):
    print("Hello"+name)
my_function("Sai")
############################
def my_function(name1,name2):
    print(name1+" "+name2)
my_function("World","Hello")
####################################
#arbitary arguments, *args
#if you do not know how many arguments that will passed into your function
#add a* before the parameter name the function name.
def my_function(*kids):
    print(kids[0]+" "+kids[1])
    print(kids[0]+" "+kids[2])
my_function("Hello","world","India")
###################################
'''
We use the name kwargs with the duble star.
The reason 

'''
def myFun(**kwargs):
    for key, value in kwargs.items():
        print("%s == %s" % (key,value))
myFun(first='Papalal',mid='Mohanlal',last='Goyal')    
#####################################################
#following ex shows how to use a default parameter
#If we call the function without arguments,
#it uses the default value:
def my_function(country = "Norway"):
    print("I am from " +country)
my_function("Sweden")
my_function("India")
my_function()
my_function("Brazil")
#################################
#Passing a list as an argument
#you can send data types of arguments to a function
fruits=["orange","banana","guava"]
def my_function(fruits):
    for x in fruits:
        print(x)
my_function(fruits)
#########################
#Return values
#to let a function return a value, use the return statement
def my_function(x):
    return x*5
my_function(12)
############################
#pass function
def my_function1():
    pass
#having an empty function definition like this, 
#would raise an error without the pass function
###########################################
#factorial of a number is the product of all the integer
#from 1 to that number.For example, the factorial of 2 is 6
def factorial(x):
    if x==1:
        return 1
    else:
        return(x*factorial(x-1))
factorial(3)
factorial(6)
#when you are sending x=6,it will check x==1 no
#then it will return 6*factorial(5)
#now 5 will be compared for condition x==1 no then it will return 5*factorial(4)
##########################
#lamda function
def add(a):
    sum=a+10
    return sum
add(20)

add=lambda a:a+10
print(add(20))
#######################
#lambda fun can take any num of arguments
add=lambda a,b:a+b
print(add(5,6))
###########################
#finding odd num from the list
lst=[34,12,64,55,75,13,63]
#odd number using lambda fun
odd_lst=list(filter(lambda x:(x%2 !=0),lst))
print(odd_lst)
#even number using lambda fun
even_lst=list(filter(lambda x:(x%2 ==0),lst))
print(even_lst)
#the filter() method accepts 2 arguments in python:
#a fun and an iterable such as a list.


#the fuc is called for every item of the list,
#and a new iterable or list is returned that holds just those elements that returned true when supplied to the

#####################################################
#square of num using map function 
lst=[2,4,5,7,8,3,6,11,12,13,15]
sqr_lst=list(map(lambda x:(x**2),lst))
print(sqr_lst)
#################################################################
#################### DAY 3 ##########################
'''Write a python code using logical operators and if elif.
so as to check height as well as
so as to allow for roller coaster also ask user age and charge
ticket accordingly
'''
#Write python code using
print("***Welcome to the roller coaster***")
height=int(input("Enter heigth in coaster:"))
if height>=120:
    print("You are eligible for roller coaster.")
    age=int(input("Enter your age:"))
    bill=0
    if age<12:
        print("Child ticket is 5$")
        bill=5
    elif age>12 and age<18:
        print("ticket amount is 10$")
        bill=10
    elif age>=18 and age<45:
        print("Ticket amount is 15$")
        bill=15
    elif age>=45 and age<=55:
        print("Adults ticket amount is 20$")
        bill=20
    want_photo=input("Do you want photo Y or N:")
    if want_photo=='Y':
        bill+=3
        print(f"You need to pay {bill} in $")
    else:
        print(f"You need to pay {bill} in $")
else:
    print("You are not eligible for roller coaster")
############################################################

#BMI calculate
height=float(input("Please enter your height in meter:"))
weight=float(input("Please enter your weight in kg:"))
BMI=round((weight/(height*height)),2)
BMI
if BMI<18.5:
    print(f"You are under weight and your BMI is:{BMI}")
elif BMI>18.5 and BMI<25:
    print(f"You are normal weight and your BMI is:{BMI}")
elif BMI>25 and BMI<30:
    print(f"You are over weight and your BMI is:{BMI}")
elif BMI>30 and BMI<35:
    print(f"You are obese and your BMI is:{BMI}")
elif BMI>35:
    print(f"You are clinically obese and your BMI is:{BMI}")
#############################################################
#sorted number
lst1=[1,2,3,4,5,6]
def is_duplicate(lst1):
    for i in range(len(lst1)-1):
        #compare current number with next number
        if(lst1[i]==lst1[i+1]):
            return True
    return False
print(is_duplicate(lst1))
#############################################################
#unsorted list
lst1=[6,2,7,4,5,6]
lst1.sort()
print(lst1)
def is_duplicate(lst1):
    for i in range(len(lst1)-1):
        #compare current number with next number
        if(lst1[i]==lst1[i+1]):
            return True
    return False
print(is_duplicate(lst1))
###############################################################

# program for leap year 
def is_leap_year(year):
    if((year>0) and (year%4==0) and (year%100!=0) or (year%400==0)):
     return True
    return False
print(is_leap_year(2024))
print(is_leap_year(2015))
###################################################################
###Important###
"""
 Write a program to display mario pyramid
"""
for i in range(4):
    for j in range(4):
        print("#",end=" ")
    print()
################################    
for i in range(4):
    for j in range(i+1):
        print("#",end=" ")
    print()
 ############################       
for i in range(4):
    for j in range(4-i):
        print("#",end=" ")
    print()        
#################################
#write program to find minimum and maximum value
lst=[23,45,2,1,5,7,8,12]
def min_max(lst):
    min=lst[0]
    for i in lst:
        if i<min:
             min=i
    print("The minimum value",min)

    max=lst[0]
    for i in lst:
        if i>max:
             max=i
    print("The maximum value",max)               
print(min_max(lst))
############################################
#write program to find out the sentence is palindrom or not
def is_palindrome(input):
    if input==" ":
        print("You entered wrong input")
    else:
        string=input[::-1]
        if string==input:
            return True
    return False
print(is_palindrome("step on no pets"))
print(is_palindrome("Nikita"))
###############################################
users=["user","Manager","Employee","Worker","staff"]
for x in users:
    if x=="user":
        print("hello admin, would you like to see a status report?")
    elif x=="Manager":
        print("hello manager")
    elif x=="Employee":
        print("hello employee")
    elif x=="Worker":
        print("hello worker")
    elif x=="staff":
        print("hello staff")
    else:
        print("hello")
#################################################
#users=["user","Manager","Employee","Worker","staff"]
users=input("Enter your roll:")
if x=="user":
    print("hello admin, would you like to see a status report?")
elif x=="Manager":
        print("hello manager")
elif x=="Employee":
        print("hello employee")
elif x=="Worker":
        print("hello worker")
elif x=="staff":
        print("hello staff")
else:
     print("hello")
#######################################
#pick the adjective
adjectives=['sleepy','slow','smelly','wet','fat','red','orange','yellow','green','blue','purple','fluffy','white','proud','brave']
#pick the noun
nouns=['apple','dinosaur','ball','toaster','goat','dragon','hammer','duck','panda']
#pick the words
import random
import string
adjective=random.choice(adjectives)
noun=random.choice(nouns)
#select number
number=random.randrange(0,100)
#select a special character
special_char=random.choice(string.punctuation)
#create the new secure password
password=adjective + noun + str(number) + special_char
print("your new password is: %s" % password)
#another one?
#you can use while loop to generate another
print("Welcome to password picker!")
while True:
    adjective=random.choice(adjectives)
    noun=random.choice(nouns)
    number=random.randrange(0,100)
    special_char=random.choice(string.punctuation)
    password=adjective+noun+str(number)+special_char
    print("Your new password is: %s" %password)
    response=input("Would you like genrate another password? Type y or n:")
    if response=='n':
        break
###########################################################################
