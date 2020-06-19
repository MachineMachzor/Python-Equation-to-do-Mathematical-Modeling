from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import datasets
import pandas as pd
import numpy as np
import csv
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

#Questions
#Which values do we need to calculate?
#Explain more about what the value C means

days = [] #the input day
area = [] #target data


"""

HOW TO IMPLEMENT YOUR OWN FUNCTION ON CUSTOM DATA

1. Import a new csv file with new data. It can have a header row with columns, and change the bacteriaCsvFile variable
2. Change the logFunc to be your own logFunc
3. Print the equation under modeledLogEquation to see the new equation (change the b and k variables/add new ones to print the full function)
4. Change modeledLogEquation with the new parameters 
5. Add new custom data for the variable newArea1D to test it on new data 
6. Profit



"""





f = open("Bacteria Area Over Days.csv") #Open the file
bacteriaCsvFile = csv.reader(f) #For reading


index = 0
for row in bacteriaCsvFile: #
	if index != 0: #If the index is not 0, or the header row
		days.append(row[0]) #Append the first item in that list (the days)
	index += 1 #Append the index so it can start adding


f = open("Bacteria Area Over Days.csv") #Second file for the second column
bacteriaCsvFile = csv.reader(f)



index2 = 0
for row in bacteriaCsvFile:
	if index2 != 0: #Same process except we're just adding the second element in the list
		area.append(row[1])
	index2 += 1




days = np.array(days)
days = days.reshape(-1,1)
days = days.astype(np.float64)

area = np.array(area)
area = area.reshape(-1,1)
area = area.astype(np.float64)






days1DArray = []
area1DArray = []
predArea1DArray = [] #They should all have the same length


for day in days:
	days1DArray.append(day[0])

for itemArea in area:
	area1DArray.append(itemArea[0])



#def logFunc(x,a,k):
#	return a * k ** x #pemdas

def logFunc(x, b, k): 
	return 63.61 / (1 + b * (2 ** (-k * x))) #Logistic models

#Exponential regression
#Needs 1d array

days1DArray = np.array(days1DArray)
area1DArray = np.array(area1DArray)

#Had to be a 1d numpy array. The p0 values have to be those pretty much, but look into it
popt, pcov = curve_fit(logFunc, days1DArray, area1DArray, p0=[5, 0.1]) #logFunctionToCalculate, Input data, output data

#Calculating r^2 value for our training data
residuals = area1DArray - logFunc(days1DArray, *popt)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((area1DArray-np.mean(area1DArray)) ** 2)
rSquaredTrainingData = 1 - (ss_res / ss_tot)
#print(rSquaredTrainingData)

newArea1D = np.array([0, 0, 0.78, 0.78, 0.78, 0.78])#Column 2 testing data
residualsv2 = newArea1D - logFunc(days1DArray, *popt)
ss_resv2 = np.sum(residualsv2**2)
ss_totv2 = np.sum((newArea1D-np.mean(area1DArray)) ** 2)
rSquaredTestingData = 1 - (ss_resv2 / ss_totv2)

#print(rSquaredTestingData)




b = popt[0].round(4) #These are the variables in the logFunc equation
k = popt[1].round(4)



#print(f"63.61/1 + {b}^({k}*x)")
#Print the equation then model it with this logFunction
def modeledLogEquation(x): #x is the day
	return 63.61 / (1 + 59.475 * (2 ** (-0.7319 * x)))


print(modeledLogEquation(100))

x_plot = np.linspace(0,10,100) #Evenly spaced numbers over an internal

plt.title("Bacteria Area In Relation to Time")
plt.xlabel("Days")
plt.ylabel("Area (mm squared)")

plt.plot(x_plot, logFunc(x_plot, *popt), "-r",
	markersize=8, linewidth=4,
	markerfacecolor="white",
	markeredgecolor="grey",
	markeredgewidth=2,
	label="Logarithmic Regression",
	)

#Equation y = a * k^x
a=popt[0].round(4)
k=popt[1].round(4)

#print(f"The equation of the regression line is y={a}*{k}^x")






#Make a predicted values list


daysPredict = days
areaPredict = []


for day in daysPredict:
	#change
	guessBacteriaArea = modeledLogEquation(day[0]) #Predicts the bacteria area for each day
	areaPredict.append(guessBacteriaArea)
	pass

print(areaPredict)



areaPredict = np.array(areaPredict) #Make the array good and ready to print





"""
https://jakevdp.github.io/PythonDataScienceHandbook/04.02-simple-scatter-plots.html

plt.plot(x, y, '-ok', color='gray',
         markersize=15, linewidth=4,
         markerfacecolor='white',
         markeredgecolor='gray',
         markeredgewidth=2)
plt.ylim(-1.2, 1.2);

"""


#undo

plt.plot(days, area, '-ok', color='gray',
	markersize=8, linewidth=4,
	markerfacecolor="white",
	markeredgecolor="gray",
	markeredgewidth=2,
	label="Bacteria Experiment")



days1DArray = []
area1DArray = []
predArea1DArray = [] #They should all have the same length


for day in days:
	days1DArray.append(day[0])

for itemArea in area:
	area1DArray.append(itemArea[0])




"""

#How to plot the coordinates for a graph
from matplotlib import pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)

A = -0.75, -0.25, 0, 0.25, 0.5, 0.75, 1.0
B = 0.73, 0.97, 1.0, 0.97, 0.88, 0.73, 0.54




#Note: These should be 1D arrays, meaning like [29,29,29]
plt.plot(A,B)
for xy in zip(A, B):                                       # <--
    ax.annotate('(%s, %s)' % xy, xy=xy, textcoords='data') # <--

plt.grid()
plt.show()
"""


""" #undo
for xy in zip(days1DArray, area1DArray):
	plt.annotate('(%s, %s)' % xy, xy=xy, textcoords='data')
"""

for xy in zip(days1DArray, area1DArray): #Print the points for our data
	plt.annotate('(%s, %s)' % xy, xy=xy, textcoords='data')


areaRounded = []
for areaPredicted in areaPredict:
	areaRounded.append(areaPredicted.round(2))

areaRounded = np.array(areaRounded)


for xy in zip(days1DArray, areaPredict): #Print the points for the model
	plt.annotate('(%s, %s)' % xy, xy=xy, textcoords='data')










#theEquation = "y = " + str(a.round(4)) + " * " + str(k.round(4)) + "^x"



#plt.text(0,40, "Logarithmic Regression Equation: {}".format(theEquation))





"""
#Calculating global R^2 value for the exponential logFunction
correlation_matrix = np.corrcoef(days1DArray,areaPredict)
correlation_xy = correlation_matrix[0,1]
r_squared = correlation_xy**2
"""







r_squared = "" #Actually calculate this bruh






plt.text(0,35, f"R^2 for this data: {rSquaredTrainingData.round(3)}")

plt.text(0, 30, f"R^2 for 2nd column (new data)): {rSquaredTestingData.round(3)}")


plt.text(0, 25, f"Logarithmic Model Equation: y = 63.61 / 59.475^(-0.7319x)")


plt.legend() #Plot the legend of the plot. This requires you to add a label to each

plt.show()















