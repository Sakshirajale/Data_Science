# -*- coding: utf-8 -*-"""
'''
1.Write a program that reads a letter of the alphabet from the
user. If the user enters a, e, i, o or u then your program should display a message
indicating that the entered letter is a vowel. If the user enters y then your program
should display a message indicating that sometimes y is a vowel, and sometimes y is
a consonant. Otherwise your program should display a message indicating that the
letter is a consonant.
'''
letter=input("Enter letter from alphabet:")
if letter.lower() in ['a','e','i','o','u']:
    print("The entered letter is a vowel")
elif letter.lower()=='y':
    print("that sometimes y is a vowel, and sometimes y is a consonant.")
else:
    print("The entered letter is a consonant.")
#####################################################################################

'''
2. Write a program that determines the name of a shape from its number of sides. Read
the number of sides from the user and then report the appropriate name as part of
a meaningful message. Your program should support shapes with anywhere from 3
up to (and including) 10 sides. If a number of sides outside of this range is entered
then your program should display an appropriate error message.
'''
num_sides=int(input("Enter the number of size:"))
if num_sides==3:                                
    print("The shape with",{num_sides}, "sides is a triangle.")
elif num_sides==4:
    print("The shape with", {num_sides}," sides is a quatrilateral.")
elif num_sides==5:
    print("The shape with",{num_sides},"sides is a pentagon.")
elif num_sides==6:
    print("The shape with",{num_sides},"sides is a hexagon.")
elif num_sides==7:
    print("The shape with",{num_sides},"sides is a heptagon.")
elif num_sides==8:
    print("The shape with",{num_sides},"sides is a octagon.")
elif num_sides==9:
    print("The shape with",{num_sides},"sides is a nonagon.")
elif num_sides==10:
    print("The shape with",{num_sides},"sides is a decagon.")
else:
    print("appropriate error message.")
    
######################################################################################

'''
3. The length of a month varies from 28 to 31 days. In this exercise you will create
a program that reads the name of a month from the user as a string. Then your
program should display the number of days in that month. Display “28 or 29 days”
for February so that leap years are addressed.
'''
month = input("Enter the name of the month: ")

if month.lower() in ["january", "march", "may", "july", "august", "october", "december"]:
    print("This month has 31 days.")
elif month.lower() in ["april", "june", "september", "november"]:
    print("This month has 30 days.")
elif month.lower() == "february":
    print("This month has 28 or 29 days.")
else:
    print("Wrong output")
####################################################################################
'''
4. A triangle can be classified based on the lengths of its sides as equilateral, isosceles
or scalene. All 3 sides of an equilateral triangle have the same length. An isosceles
triangle has two sides that are the same length, and a third side that is a different
length. If all of the sides have different lengths then the triangle is scalene.
Write a program that reads the lengths of 3 sides of a triangle from the user.
Display a message indicating the type of the triangle.
'''
side1 = int(input("Enter first side of triangle: "))
side2 = int(input("Enter second side of triangle: "))
side3 = int(input("Enter third side of triangle: "))
if side1 == side2 == side3:
    print("The triangle is an Equilateral triangle.")
elif side1 == side2 or side1 == side3 or side2 == side3:
    print("The triangle is an Isosceles triangle.")
elif side1 != side2 and side2 != side3 and side1 != side3:
    print("The triangle is a Scalene triangle.")
#####################################################################################
'''
5. The year is divided into three seasons: summer, rainy and winter. While the
exact dates that the seasons change vary a little bit from year to year because of the
way that the calendar is constructed, Write a program to display the season if date is given
'''
month = int(input("Enter the month (1-12): "))
day = int(input("Enter the day (1-31): "))
if (month == 3 and day >= 20) or (month == 4) or (month == 5) or (month == 6 and day < 21):
        print("Spring season")
elif (month == 6 and day >= 21) or (month == 7) or (month == 8) or (month == 9 and day < 23):
        print("Summer season")
elif (month == 9 and day >= 23) or (month == 10) or (month == 11) or (month == 12 and day < 21):
        print("Rainy season")
elif (month == 12 and day >= 21) or (month == 1) or (month == 2) or (month == 3 and day < 20):
        print("Winter season")
else:
        print("Invalid ")
        
######################################################################################
