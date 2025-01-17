# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 22:45:52 2024

@author: ACER
"""
####################################################################
#Exercise 1: Mailing Address
#Write a program that displays your name and complete mailing address formatted in
#the manner that you would usually see it on the outside of an envelope. Your program
#does not need to read any input from the user.
print("Rajale Sakshi Prasad")
print("At post:Kharwandi,Taluka:Newasa,Dist:A.Nagar")
####################################################################


#Exercise 2:Area of a Room
#Write a program that asks the user to enter the width and length of a room. Once
#the values have been read, your program should compute and display the area of the
#room. The length and the width will be entered as floating point numbers. Include
#units in your prompt and output message; either feet or meters, depending on which
#unit you are more comfortable working with.
length=int(input("Enter the length:"))
width=int(input("Enter the width:"))
#converting int to float
conv_length=float(length)
conv_width=float(width)
#Area of room
Area=conv_length*conv_width
#Display area
print("Area of room is",Area,"sq.m.")
####################################################################


#Exercise 3:Area of a Field
#Create a program that reads the length and width of a farmer’s field from the user in
#feet. Display the area of the field in acres.
#Hint: There are 43,560 square feet in an acre.
length=int(input("Enter the length:"))
width=int(input("Enter the width:"))
#converting int to float data type
conv_length=float(length)
conv_width=float(width)
#the area of the field in acres
Area=conv_length*conv_width/43560
#Display the result
print("The area of the field is",Area,"acre.")
####################################################################


#Exercise 4: Bottle Deposits
#In many jurisdictions a small deposit is added to drink containers to encourage people
#to recycle them. In one particular jurisdiction, drink containers holding one liter or
#less have a $0.10 deposit, and drink containers holding more than one liter have a
#$0.25 deposit.
#Write a program that reads the number of containers of each size from the user.
#Your program should continue by computing and displaying the refund that will be
#received for returning those containers. Format the output so that it includes a dollar
#sign and always displays exactly two decimal places.
#To read the number of container of each size from user
num_one_liter_less=float(input("Enter the number of bottles of one liter or less here:"))
refund_one_liter_less=0.10*num_one_liter_less

if num_one_liter_less>=0.0:
    print("Your refund for one liter or less bottle is:","$",refund_one_liter_less,".","00")

num_more_one_liter=float(input("Enter the number of bottles more than one liter here:"))
refund_more_one_liter=0.25*num_more_one_liter

if num_more_one_liter>=0.0:
    print("Your refund for more than one liter bottle is:","$",refund_more_one_liter,".","00")

if num_one_liter_less or num_more_one_liter>=0.0:
    print("Your refund for all bottle is:","$",refund_more_one_liter+refund_one_liter_less)



####################################################################


#Exercise 5:Tax and Tip
#The program that you create for this exercise will begin by reading the cost of a meal
#ordered at a restaurant from the user. Then your program will compute the tax and
#tip for the meal. Use your local tax rate when computing the amount of tax owing.
#Compute the tip as 18 percent of the meal amount (without the tax). The output from
#your program should include the tax amount, the tip amount, and the grand total for
#the meal including both the tax and the tip. Format the output so that all of the values
#are displayed using two decimal places.
cost_m=int(input("Enter the cost of meal:"))
#calculate tax of meal
tax=0.20*cost_m
#calculate tip of meal
tip=0.18*cost_m
#calculate total amount payable
total=tax+tip+cost_m
#display cost,tax,tip,total amount
print("cost of meal:",cost_m)
print("Tax on meal is:",tax)
print("Tip on meal is:",tip)
print("Total amount payable:",total)
####################################################################


#Exercise 6:   Height Units
#Many people think about their height in feet and inches, even in some countries that
#primarily use the metric system. Write a program that reads a number of feet from
#the user, followed by a number of inches. Once these values are read, your program
#should compute and display the equivalent number of centimeters.
height1=float(input("Enter height in feets:"))
print("Height in Centimeters",height1*30.48)
height2=float(input("Enter  Height in inches:"))
print("Height in Centimeters",height2*2.54)
####################################################################




