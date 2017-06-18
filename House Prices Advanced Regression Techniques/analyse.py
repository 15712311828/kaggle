from loadData import loadTrainData
import matplotlib.pyplot as plt
import pandas as pd


data = loadTrainData()
prices = data["SalePrice"]
del data["SalePrice"]
for attr in data.columns:
    notnull = pd.notnull(data[attr])
    column = data[attr][notnull]
    i = 0
    while(not notnull[i]):
        i = i+1
    if(not isinstance(column[i], str)):
        plt.figure(0)
        p = prices[notnull]
        plt.plot(column, p, 'o')
        plt.xlabel(attr)
        plt.savefig('House Prices Advanced Regression Techniques\\pic\\'
                    + attr + '.png')
        plt.close(0)
