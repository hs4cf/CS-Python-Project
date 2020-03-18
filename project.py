import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
%matplotlib inline


# Read the csv file:
url = 'https://raw.githubusercontent.com/hs4cf/CS-Python-Project/master/hate_crimes.csv'
dataset = pd.read_csv(url, error_bad_lines=False)
#df

#dataset = pd.read_csv('hate_crimes.csv',header=0, encoding = "ISO-8859-1")
df = pd.read_csv('hate_crimes.csv',header=0, encoding = "ISO-8859-1")

#Take user input
input_state = input("Which state's data would you like to see? ")
lower_input = input_state.lower()
stateOfInterest = lower_input.capitalize()

df[df['state'].isin([stateOfInterest])] 

dataset.shape
dataset.describe()
dataset.plot(x='median_household_income', y='avg_hatecrimes_per_100k_fbi', style='o')  
dataset.plot(x='median_household_income', y='hate_crimes_per_100k_splc', style='o')  
dataset.plot(x='avg_hatecrimes_per_100k_fbi', y='gini_index', style='o')  


plt.title('median_household_income vs avg_hatecrimes_per_100k_fbi')  
plt.xlabel('median_household_income')  
plt.ylabel('avg_hatecrimes_per_100k_fbi')  
plt.show()

plt.figure(figsize=(15,10))
plt.tight_layout()
seabornInstance.distplot(dataset['avg_hatecrimes_per_100k_fbi'])

X = dataset['median_household_income'].values.reshape(-1,1)
y = dataset['avg_hatecrimes_per_100k_fbi'].values.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, y,  random_state=0)
regressor = LinearRegression()  
regressor.fit(X_train, y_train) #training the algorithm
#To retrieve the intercept:
print("intercept: ")
print(regressor.intercept_)
#For retrieving the slope:
print("slope: ")
print(regressor.coef_)
y_pred = regressor.predict(X_test)

df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
df

df1 = df.head(25)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
#plt.show()

plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()


#Take user input
stateOfInterest= input("Which state's data would you like to see?")
df[df['state'].isin([stateOfInterest])] 
