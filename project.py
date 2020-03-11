
from numpy import *
import pandas as pd

# Read the csv file:
df = pd.read_csv('hate_crimes.csv',header=0, encoding = "ISO-8859-1")

#Take user input
stateOfInterest= input("Which state's data would you like to see?")
df[df['state'].isin([stateOfInterest])] 


#testing
#HanimTest
#Anvartest
