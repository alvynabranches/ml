import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score

# read csv file
df = pd.read_csv(r'weight_height.csv')


# changing the gender columns from categorical data to binary data
def g2b(v):
    if v == 'Male':
        return 0
    else:
        return 1

# applying the function
df['GenderBinary'] = df['Gender'].apply(g2b)

# seperating male and female data into 2 seperate tables
male = df[df['Gender'] == 'Male']
female = df[df['Gender'] == 'Female']

# putting the data into x and y data
x_male = np.array([male['Height']]).T
y_male = np.array(male['Weight'])
x_female = np.array([df['Height']]).T
y_female = np.array(df['Weight'])

# create instance of linear regression class
lr_male = LinearRegression()
lr_female = LinearRegression()

x_male_train, x_male_test, y_male_train, y_male_test = train_test_split(x_male, y_male)
x_female_train, x_female_test, y_female_train, y_female_test = train_test_split(x_female, y_female)


# fitting the data to the linear regression object
lr_male.fit(x_male_train, y_male_train)
lr_female.fit(x_female_train, y_female_train)


# predicting the y values for the x data
y_male_train_pred = lr_male.predict(x_male_train)
y_male_test_pred = lr_male.predict(x_male_test)
y_female_train_pred = lr_female.predict(x_female_train)
y_female_test_pred = lr_female.predict(x_female_test)

print("Male")
print("R2 Score Train", end='\t')
print(str(r2_score(y_male_train, y_male_train_pred)))
print("R2 Score Test", end='\t')
print(str(r2_score(y_male_test, y_male_test_pred)))

print("Female")
print("R2 Score Train", end='\t')
print(str(r2_score(y_female_train, y_female_train_pred)))
print("R2 Score Test", end='\t')
print(str(r2_score(y_female_test, y_female_test_pred)))

print("Accuracy", end='\t')
try:
    accuracy_score(y_true=y_male_train, y_pred=y_male_train_pred)
except Exception as e:
    print(e)