
from numpy import *
import pandas as pd

# Read the csv file:
df = pd.read_csv('hate_crimes.csv',header=0, encoding = "ISO-8859-1")

print("  *** Using file: hate_crimes.csv ***")
# Print the first 5 data points (using "head")
print("\n****************")
#print('*** The first 5 data points\n',df.head(6))

# Sort by column avg_hatecrimes_per_100k_fbi
df.sort_values('avg_hatecrimes_per_100k_fbi')
print('*** The first 5 data points\n',df.sort_values('avg_hatecrimes_per_100k_fbi').head(6))
print('*** The last 5 data points\n',df.sort_values('avg_hatecrimes_per_100k_fbi').tail(6))

# Sort by column avg_hatecrimes_per_100k_splc
df.sort_values('hate_crimes_per_100k_splc')
print('*** The first 5 data points\n',df.sort_values('hate_crimes_per_100k_splc').head(6))
print('*** The last 5 data points\n',df.sort_values('hate_crimes_per_100k_splc').tail(6))

#testing
#HanimTest
#Anvartest
