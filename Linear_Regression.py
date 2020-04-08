# Importing libraries
# plotting library and numpy
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
#linear regression library
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# read CSV file
# An example on how the path works
# path = "C:\\Users\\Mauricio\\Downloads\\dataSet1.csv"
#if if csv is not in the same dir use ^
path="dataSet1.csv"
data = pd.read_csv(path, delimiter=',',names=['X','Y'])

# putting data into np arrays
x = np.array(data['X']).reshape((-1,1))
y = np.array(data['Y'])

# linear Regression fomula is performed
model = LinearRegression().fit(x,y)

#x_predict is a copy of X
X_predict=x
# y_predict is the predicted values of our model
y_predict = model.predict(X_predict)
# print(y_predict)
# ^ shows all y-predicted values

#predict the value for y when x is 27
y_predict_num = model.predict([[27]])
print("Predicted value for y, when x = 27: ", y_predict_num)

#predict the value for y when x is 150
y_predict_num = model.predict([[150]])
print("Predicted value for y, when x = 150: ", y_predict_num)

#plot regular graph
plt.scatter(x,y, color="black")
plt.xlabel('X')
plt.ylabel('Y')
plt.title('X vs. Y: Without Linear Regression')

#plot linear
#black dots
plt.scatter(x,y, color="black")
#blue regression line
plt.plot(x,y_predict, color="blue")
plt.xlabel('X')
plt.ylabel('Y')
plt.title('X vs. Y: With Linear Regression')