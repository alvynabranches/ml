import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt


# read csv
ds = pd.read_csv('2017.csv')


x = ds['Economy..GDP.per.Capita.'].values.reshape(-1,1)
y = ds['Happiness.Score'].values.reshape(-1,1)

lr = LinearRegression()

lr.fit(x,y)

print("Beta 0: {}\nBeta 1: {}".format(lr.intercept_,lr.coef_))

y_pred_regression_fit = lr.predict(x)

rmse = sqrt(mean_squared_error(y,y_pred_regression_fit))
print("RMSE regression fit: {}".format(rmse))


# weights initialized randomly
b_0, b_1 = 0,1

# initiatial values
h_x = []
n = len(x)
g_b_0 = 0
g_b_1 = 0
err_fun = []
err = 0

# predicted values for the parameters initialized randomly
h_x = [(b_0 + b_1 * x[i]) for i in range(n)]

# error is high so gradient descent is used
rmse_g = sqrt(mean_squared_error(y,h_x))
print("RMSE: {}".format(rmse_g))

c = 0
# list of rmse
rmse_lis = [rmse_g]


# gradient decent algorithm
b_0_plot = []
b_1_plot = []
tolerance = 0.00001
eta = 0.00000001
while 1:
    
    err = 0
    for j in range(n):
        # derivative with respect to b0
        derv_b0 = (y[j]-h_x[j])
        # derivative with respect to b1
        derv_b1 = (y[j]-h_x[j])*x[j]
        # new value of gradient b0
        g_b_0 = g_b_0 - derv_b0
        # new value of gradient b1
        g_b_1 = g_b_1 - derv_b1
        # updated error
        err += ((y[j]-h_x[j])**2)
    err_fun.append(1/2*1/n*err)
    
    #new b0
    b_0 = b_0 - (eta * g_b_0)
    #new b1
    b_1 = b_1 - (eta * g_b_1)
    #new model
    h_x = [(b_0 + b_1 * x[i]) for i in range(n)]
    # rmse for the gradient
    rmse_g = sqrt(mean_squared_error(y,h_x))

    # appending for plotting
    b_0_plot.append(b_0)
    b_1_plot.append(b_1)
    
    # stopping criteria
    if c>=1:
        if abs(err_fun[c]-err_fun[c-1])<tolerance:
            rmse_lis.append(rmse_g)
            c+=1
            break
    print(err_fun[c])
    c+=1
    # appending to the gradient
    rmse_lis.append(rmse_g)

print("No. of iterations: {}\nRMSE Gradient fit: {}".format(c,rmse_g))

print("Beta 0: {}\nBeta 1: {}".format(b_0,b_1))


# plotting x and y variables ie Economy..GDP.per.Capita. vs Happiness Score
plt.scatter(x, y, color = 'red', marker='o', label='Actual')
plt.plot(x, h_x, color = 'blue', label='Predicted gradient fit',marker='x')
plt.plot(x, y_pred_regression_fit, color = 'yellow', label='Predicted regression fit')

plt.title('Economy..GDP.per.Capita. vs Happiness Score')
plt.legend()
plt.xlabel('Economy..GDP.per.Capita.')
plt.ylabel('Happiness Score')
plt.show()

print("Predicted Value using gradient:\n{}\nPredicted Value using regression:\n{}\nActual Value:\n{}".format(h_x,y_pred_regression_fit,y))

len(h_x),len(y_pred_regression_fit),len(y)

# plotting b0 b1 vs cost
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(b_0_plot,b_1_plot,err_fun,c='r',marker='.')
ax.set_xlabel('Beta 0')
ax.set_ylabel('Beta 1')
ax.set_zlabel('Cost')
plt.show()


# plotting Economy..GDP.per.Capita. vs Hapiness Score
plt.scatter(x,y,c='r')
plt.plot(x,y_pred_regression_fit)
plt.xlabel('Economy..GDP.per.Capita.')
plt.ylabel('Happiness Score')
plt.show()

