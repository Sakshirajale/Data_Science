# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 16:08:36 2024

@author: ACER
"""
from wrapper import time_it
@time_it
def calc_square(numbers):
    result= []
    for number in numbers:
        result.append(number*number)
    return result

@time_it
def calc_cube(numbers):
    result= []
    for number in numbers:
        result.append(number*number*number)
    return result

array=range(1,100000)
out_square=calc_square(array)
out_cube=calc_cube(array)