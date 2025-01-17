###########################################################################
# Association_Ruless

import pandas as pd
# Pandas library for data manipulation   
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

#Sample dataset
transactions = [['Milk','Bread','Butter'],['Bread','eggs'],['Milk','Bread','Eggs','Butter'],['Bread','Eggs','Butter'],['Milk','Bread','Eggs']]

#Step-1: Convert the dataset into a format 
# This dataset need to be converted into one-hot encoded format (1,0 --> item is present not present respectively) in order to apply apriori algorithm
# And we will convert this data with the help of TransactionEncoder
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns = te.columns_)
# pd.DataFrame(te_ary): Creates a DatFrame from binary array te_ary.
# columns = te.columns_ : This sets the column names of the DataFrame to the item names from the transactions

#Step-2: Apply the Apriori algorithm to find frequent itemsets
frequent_itemsets = apriori(df, min_support = 0.5, use_colnames=True)
# min_support = 0.5: Only interested in itemsets that appear in at least 50% of the transactions
# use_colnames = True : Actually use column names instead of column indices.
# The output is a DataFrame containing the frequent itemsets (item combination)

# Bread appears in 100% transactions
# Egg, Butter, Milk each appears in 60% of transactions.
# The combination also appear in 60% of transactions

#Step-3: Generate association rules from the frequent itemsets
# Now, generating association rules
# Generating association rules based on frequent itemsets.
rules = association_rules(frequent_itemsets, metric = 'lift', min_threshold=1)
# metric='lift': This specifies that we want to calculate the lift of the association rules. 
#  Lift measures how much more likely two items are to be bought together than if they were independent (i.e., bought randomly).

# min_threshold=1:  This tells the algorithm to only return association rules where the lift is 1 or higher, 
# meaning we are only interested in associations that are as likely or more likely than chance.

# Step-4: Output the results
print("Frequent Itemsets: ")
print(frequent_itemsets)
'''
o/p--->
support         itemsets
0      1.0          (Bread)
1      0.6         (Butter)
2      0.6           (Eggs)
3      0.6           (Milk)
4      0.6  (Bread, Butter)
5      0.6    (Eggs, Bread)
6      0.6    (Milk, Bread)
'''

print("\nAssociation Rules: ")
print(rules[['antecedents','consequents','support','confidence','lift']])
'''
antecedents consequents  support  confidence  lift
0     (Bread)    (Butter)      0.6         0.6   1.0
1    (Butter)     (Bread)      0.6         1.0   1.0
2      (Eggs)     (Bread)      0.6         1.0   1.0
3     (Bread)      (Eggs)      0.6         0.6   1.0
4      (Milk)     (Bread)      0.6         1.0   1.0
5     (Bread)      (Milk)      0.6         0.6   1.0
'''
###########################################
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Step-1: Simulating healthcare transactions (symptoms/diseases/treatment)
healthcare_data = [
    ['Fever','Cough','COVID-19'],
    ['Cough','Sore Throat','Flu'],
    ['Fever','Cough','Shortness of Breadth','COVID-19'],
    ['Cough','Sore Throat','Flu','Headache'],
    ['Fever','Body Ache','Flu'],
    ['Fever','Cough','COVID-19','Shortness of Breadth'],
    ['Sore Throat','Headache','Cough'],
    ['Body Ache','Fatigue','Flu']
    ]

# Step-2: 
te = TransactionEncoder()
te_ary = te.fit(healthcare_data).transform(healthcare_data)
df = pd.DataFrame(te_ary, columns = te.columns_)
 

#Step-3: Apply the Apriori algorithm to find frequent itemsets
frequent_itemsets = apriori(df, min_support = 0.3, use_colnames=True)

#Step-4: Generate association rules from the frequent itemsets
rules = association_rules(frequent_itemsets, metric = 'confidence', min_threshold=0.7)

# Step-5: Output the results
print("Frequent Itemsets: ")
print(frequent_itemsets)
'''
o/p-->
support                  itemsets
0    0.375                (COVID-19)
1    0.750                   (Cough)
2    0.500                   (Fever)
3    0.500                     (Flu)
4    0.375             (Sore Throat)
5    0.375         (Cough, COVID-19)
6    0.375         (Fever, COVID-19)
7    0.375            (Fever, Cough)
8    0.375      (Cough, Sore Throat)
9    0.375  (Fever, Cough, COVID-19)
'''

print("\nAssociation Rules: ")
print(rules[['antecedents','consequents','support','confidence','lift']])
'''
o/p-->
antecedents        consequents  support  confidence      lift
0         (COVID-19)            (Cough)    0.375        1.00  1.333333
1            (Fever)         (COVID-19)    0.375        0.75  2.000000
2         (COVID-19)            (Fever)    0.375        1.00  2.000000
3            (Fever)            (Cough)    0.375        0.75  1.000000
4      (Sore Throat)            (Cough)    0.375        1.00  1.333333
5     (Fever, Cough)         (COVID-19)    0.375        1.00  2.666667
6  (Fever, COVID-19)            (Cough)    0.375        1.00  1.333333
7  (Cough, COVID-19)            (Fever)    0.375        1.00  2.000000
8            (Fever)  (Cough, COVID-19)    0.375        0.75  2.000000
9         (COVID-19)     (Fever, Cough)    0.375        1.00  2.666667
'''
###################################################
# Step-1: Simulate e-commerce transactions
transactions = [
    ['Laptop','Mouse','Keyboard'],
    ['Smartphone','Headphones'],
    ['Laptop','Mouse','Headphones'],
    ['Smartphone','Charger','Phone Case'],
    ['Laptop','Mouse','Monitor'],
    ['Headphones','Smartwatch'],
    ['Laptop','Keyboard','Monitor'],
    ['Smartphone','Charger','Phone Case','Screen Protector'],
    ['Mouse','Keyboard','Monitor'],
    ['Smartphone','Headphones','Smartwatch']]

# Step-2: 
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns = te.columns_)
 

#Step-3: Apply the Apriori algorithm to find frequent itemsets
frequent_itemsets = apriori(df, min_support = 0.2, use_colnames=True)

#Step-4: Generate association rules from the frequent itemsets
rules = association_rules(frequent_itemsets, metric = 'confidence', min_threshold=0.5)

# Step-5: Output the results
print("Frequent Itemsets: ")
print(frequent_itemsets)
'''
o/p-->
support                           itemsets
0       0.2                          (Charger)
1       0.4                       (Headphones)
2       0.3                         (Keyboard)
3       0.4                           (Laptop)
4       0.3                          (Monitor)
5       0.4                            (Mouse)
6       0.2                       (Phone Case)
7       0.4                       (Smartphone)
8       0.2                       (Smartwatch)
9       0.2              (Charger, Phone Case)
10      0.2              (Smartphone, Charger)
11      0.2           (Smartphone, Headphones)
12      0.2           (Smartwatch, Headphones)
13      0.2                 (Laptop, Keyboard)
14      0.2                (Monitor, Keyboard)
15      0.2                  (Keyboard, Mouse)
16      0.2                  (Laptop, Monitor)
17      0.3                    (Laptop, Mouse)
18      0.2                   (Monitor, Mouse)
19      0.2           (Smartphone, Phone Case)
20      0.2  (Smartphone, Charger, Phone Case)
'''

print("\nAssociation Rules: ")
print(rules[['antecedents','consequents','support','confidence','lift']])
'''
o/p-->
antecedents               consequents  ...  confidence      lift
0                  (Charger)              (Phone Case)  ...    1.000000  5.000000
1               (Phone Case)                 (Charger)  ...    1.000000  5.000000
2               (Smartphone)                 (Charger)  ...    0.500000  2.500000
3                  (Charger)              (Smartphone)  ...    1.000000  2.500000
4               (Smartphone)              (Headphones)  ...    0.500000  1.250000
5               (Headphones)              (Smartphone)  ...    0.500000  1.250000
6               (Smartwatch)              (Headphones)  ...    1.000000  2.500000
7               (Headphones)              (Smartwatch)  ...    0.500000  2.500000
8                   (Laptop)                (Keyboard)  ...    0.500000  1.666667
9                 (Keyboard)                  (Laptop)  ...    0.666667  1.666667
10                 (Monitor)                (Keyboard)  ...    0.666667  2.222222
11                (Keyboard)                 (Monitor)  ...    0.666667  2.222222
12                (Keyboard)                   (Mouse)  ...    0.666667  1.666667
13                   (Mouse)                (Keyboard)  ...    0.500000  1.666667
14                  (Laptop)                 (Monitor)  ...    0.500000  1.666667
15                 (Monitor)                  (Laptop)  ...    0.666667  1.666667
16                  (Laptop)                   (Mouse)  ...    0.750000  1.875000
17                   (Mouse)                  (Laptop)  ...    0.750000  1.875000
18                 (Monitor)                   (Mouse)  ...    0.666667  1.666667
19                   (Mouse)                 (Monitor)  ...    0.500000  1.666667
20              (Smartphone)              (Phone Case)  ...    0.500000  2.500000
21              (Phone Case)              (Smartphone)  ...    1.000000  2.500000
22     (Smartphone, Charger)              (Phone Case)  ...    1.000000  5.000000
23  (Smartphone, Phone Case)                 (Charger)  ...    1.000000  5.000000
24     (Charger, Phone Case)              (Smartphone)  ...    1.000000  2.500000
25              (Smartphone)     (Charger, Phone Case)  ...    0.500000  2.500000
26                 (Charger)  (Smartphone, Phone Case)  ...    1.000000  5.000000
27              (Phone Case)     (Smartphone, Charger)  ...    1.000000  5.000000

[28 rows x 5 columns]
'''