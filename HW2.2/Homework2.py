#!/usr/bin/env python
# coding: utf-8

# In[62]:


import json
import matplotlib.pyplot as plt
#from scipy.optimize import minimize
import numpy as np
import pandas as pd
import math
import random


# In[39]:


with open('weight.json') as file:
    jsondata = json.load(file)


# In[40]:


jsondata


# In[44]:


rawdata = pd.DataFrame([jsondata['x'], jsondata['y'], jsondata['is_adult']]).T
rawdata.columns = ['x', 'y', 'is_adult']
rawdata.head()

data = rawdata.sample(n=int(len(rawdata['x'])*0.7))
test = rawdata.sample(n=int(len(rawdata['x'])*0.3))

with open('planar_x1_x2_y.json') as file:
    jsondata = json.load(file)

complexdata = pd.DataFrame([jsondata['x1'], jsondata['x2'], jsondata['y']]).T
complexdata.columns = ['x1', 'x2', 'y']


# In[45]:


plt.scatter(data['x'], data['y'])


# In[17]:


plt.scatter(data['y'], data['is_adult'])


# In[138]:


# Linear Function
# Parameters:
#  x -- The float x-value to calculate f(x) from
#  p -- An Array of 2 values: p[0] = m (slope). p[1] = b (intercept)
# Returns: A float f(x) value for the function y = mx+b
def lin_f(x, p):
    if type(x) is not list:
        return p[0] * x + p[1]
    n_parameters = len(x)
    return [x[i] * p[i] + p[-1] for i in range(n_parameters)]

# Logistic Function
# Parameters:
#  x -- The float x-value to calculate f(x) from
#  p -- An Array of 4 values: p[0] = A. p[1] = w. p[2] = x0. p[3] = S (intercept)
# Returns: A float f(x) value for the general logistic function y = A / (1 + exp(-(x - x0)/w)) + S
def log_f(x, p):
    if type(x) is list:
        n_parameters = len(x)
        result = [p[-1] / math.exp(-1 * x[i] * p[i]) for i in range(n_parameters)]
    else:
        try:
            result = (p[0] / (1 + math.exp(-(x - p[2])/p[1]))) + p[3]
        except OverflowError:
            # If the math.exp function has an overflow, then that whole first term is going to 0
            result = p[3]
    return result

# Function to minimize: Calculate least squares distance (y-distance for simplicity) between actual points and their 
#  corresponding points on the line
# Parameters:
#  p -- Array that contains the arguments needed for the function. p[0] is x, p[1:] represents all other arguments. Refer to each function to see what those are
#  point_x -- The x-values for the training data points. Array of floating point numbers
#  point_y -- The y-values for the training data points. Array of floating point numbers
#  f_name -- The String name of the generalized function being called. MUST be the name of a function in this file. Examples are 'lin_f' and 'log_f'
# Returns: The sum of the squared distances for each training point
def min_squares_f(p, point_x, point_y, f_name):
    m, b = p[0], p[1]
    y = [globals()[f_name](i, p) for i in point_x]
    np_y = np.array(point_y, np.float32)
    distance = [np_y[i] - y[i] for i in range(len(y))]
    return sum([d**2 for d in distance])


# Funtion to run a minimization using Gradient Descent
# Runs Full Batch, Mini Batch, or Stochastic versions of the algorithm, based on the specified number of batches
# Has the option to run with or without momentum
# Parameters:
#  func -- the minimization function. Recommend min_squares_f, as it is already written in this file. Should be a FUNCTION
#  init_guess -- initial guess/starting point for the parameters. Useful for mimicking the scipy minimize function. Should be an ARRAY of FLOATS, of length equal to the number of params being determined
#  args -- any additional arguments to be passed to the function (ARRAY)
#  method -- STRING specification of which method to use. Currently supports 'GD' (Gradient Descent) and 'GD momentum' (GD with momentum)
#  tol -- carried over from Scipy; a FLOAT that represents how much wiggle is allowed before the algorithm calls it quits
#  n_batches -- an INT that specifies how many batches to use. n_batches=1: FULL BATCH. n_batches=len(data): STOCHASTIC. Anything in between: MINI BATCH. Default is 1 (FULL)
def minimize(func, init_guess, args, method, tol, n_batches=1):
    points_x, points_y, func_name = args
    points = list(zip(points_x, points_y))
    batch = random.sample(points, int(len(points_x) / n_batches))
    batch_x = [b[0] for b in batch]
    batch_y = [b[1] for b in batch]
    dx = 1e-4
    learning_rate = 1e-4
    n_coords = len(init_guess)
    momentum = [0 for i in range(n_coords)]
    max_steps = 1000000
    steps = 0
    at_local_minima = False
    current_coord = init_guess # initialize the current coordinate
    current_sse = func(current_coord, batch_x, batch_y, func_name)
    
    back_sse = [func([current_coord[p] - dx if p == i else current_coord[p] for p in range(n_coords)], batch_x, batch_y, func_name) for i in range(n_coords)]
    front_sse = [func([current_coord[p] + dx if p == i else current_coord[p] for p in range(n_coords)], batch_x, batch_y, func_name) for i in range(n_coords)]
    gradient = [(front_sse[i] - back_sse[i])/(dx * 2) for i in range(n_coords)]
    next_coord = [current_coord[i] + learning_rate * gradient[i] for i in range(n_coords)]
    next_sse = func(next_coord, batch_x, batch_y, func_name)
    if 'momentum' in method:
        momentum = [momentum[i] + gradient[i] for i in range(n_coords)]
    while (not at_local_minima) and not (steps >= max_steps):
        batch = random.sample(points, int(len(points_x) / n_batches))
        batch_x = [b[0] for b in batch]
        batch_y = [b[1] for b in batch]
        steps += 1
        current_coord = next_coord
        current_sse = next_sse
        
        back_sse = [func([current_coord[p] if p != i else current_coord[p] - dx for p in range(n_coords)], batch_x, batch_y, func_name) for i in range(n_coords)]
        front_sse = [func([current_coord[p] if p != i else current_coord[p] + dx for p in range(n_coords)], batch_x, batch_y, func_name) for i in range(n_coords)]
        #if front_sse > back_sse:
        gradient = [(front_sse[i] - back_sse[i])/(dx * 2) for i in range(n_coords)]
        #else:
        #    gradient = [(back_sse[i] - front_sse[i])/(dx * 2) for i in range(n_coords)]
        next_coord = [current_coord[i] + learning_rate * gradient[i] + momentum[i] for i in range(n_coords)]
        next_sse = func(next_coord, batch_x, batch_y, func_name)
        if 'momentum' in method:
            momentum = [((momentum[i] * 0.5) + gradient[i]) if (momentum[i] >= 0) else ((-1 * (abs(momentum[i]) * 0.5)) + gradient[i]) for i in range(n_coords)]
        #print(steps)
        #print(current_coord)
        #print(current_sse)
        #print(next_coord)
        #print(next_sse)
        #print(gradient)
        # See if the gradient has slowed down enough to stop
        print(current_coord)
        print(next_coord)
        print(next_sse)
        print(current_sse)
        if (next_sse - current_sse) < tol:
            at_local_minima = True
    return next_coord


# In[143]:

# Note: Instead of normalizing, I'm feeding the params into the minimization function. This is a little more intuitive for me to think about, and removes the step of un-normalizing the results


# ===== LINEAR PLANAR =====

lin_min = minimize(min_squares_f, [1, 1], args=(complexdata[['x1', 'x2']].values.tolist(), complexdata['y'], 'lin_f'), method='GD', tol=1e-5)
#lin_min
print("Linear Classification completed.")
print(lin_min)


# =========


# \\\\\\\\\\ Full Batch //////////

# ===== LINEAR REGRESSION ======

lin_min = minimize(min_squares_f, [1, 1], args=(data[data['x'] < 18]['x'], data[data['x'] < 18]['y'], 'lin_f'), method='GD', tol=1e-5)
#lin_min
print("Linear Classification completed.")
print(lin_min)

# In[144]:


params = lin_min
#params


# In[145]:

# Plot the results of the Linear Regression!

plt.scatter(data['x'], data['y'])
plt.scatter(test['x'], test['y'])
x = np.linspace(min(rawdata[rawdata['x'] < 18]['x']),max(rawdata[rawdata['x'] < 18]['x']),100)
y = params[0]*x+params[1]
plt.plot(x, y, '-r')
plt.xlabel('Age')
plt.ylabel('Weight')
plt.title('Linear Regression Full Batch Results')
plt.savefig('linear_regression.png')

plt.clf()

# Parity Plot -- predicted vs actual weights
plt.scatter(data[data['x'] <18]['y'], [params[0]*i+params[1] for i in data[data['x'] < 18]['x']])
plt.scatter(test[test['x'] <18]['y'], [params[0]*i+params[1] for i in test[test['x'] < 18]['x']])
plt.xlabel('Predicted Weight')
plt.ylabel('Actual Weight')
plt.title('Linear Regression Full Batch Parity')
plt.savefig('parity_lin_regression.png')
print("Linear figure saved to linear_regression.png")
plt.clf()

# ===== ===== =====

# In[146]:

# ===== LOGISTIC REGRESSION =====

log_min = minimize(min_squares_f, [1, 1, 1, 1], args=(data['x'], data['y'], 'log_f'), method='GD', tol=1e-5)
#log_min
print("Logistic Regression completed.")
print(log_min)


# In[147]:

# Plot the results of the logistic regression!

params = log_min
plt.scatter(data['x'], data['y'])
plt.scatter(test['x'], test['y'])
x = np.linspace(min(rawdata['x']),max(rawdata['x']),100)
y = [(params[0] / (1 + math.exp(-(i - params[2])/params[1]))) + params[3] for i in x]
plt.plot(x, y, '-r')
plt.xlabel('Age')
plt.ylabel('Weight')
plt.title('Logistic Regression Full Batch Results')
plt.savefig('logistic_regression.png')

plt.clf()

# Parity plot -- predicted vs actual weights
plt.scatter(data['y'], [log_f(i, params) for i in data['x']])
plt.scatter(test['y'], [log_f(i, params) for i in test['x']])
plt.xlabel('Predicted Weight')
plt.ylabel('Actual Weight')
plt.title('Logistic Regression Full Batch Parity')
plt.savefig('parity_log_regression.png')
print("Logistic plot saved to logistic_regression.png")
plt.clf()

# In[148]:


# ===== BINARY CLASSIFICATION =====

# For this, adjusting the starting parameters helps a lot when generating an accurate sigmoid
# For simplicity, the classifications are done at plot-time
# Threshold is set to 0.5 -- halfway between 0 and 1 (the classifications)
threshold = 0.5

log_min = minimize(min_squares_f, [1, 1, np.mean(data['y']), min(data['is_adult'])], args=(data['y'], data['is_adult'], 'log_f'), method='GD', tol=1e-5)
#log_min
print("Binary Logistic Regression completed.")
print(log_min)

# In[149]:

# Plot the results of the binary classification!

params = log_min
plt.scatter(data['y'], data['is_adult'])
plt.scatter(test['y'], test['is_adult'])
x = np.linspace(min(rawdata['y']),max(rawdata['y']),100)
y = [0 if log_f(i, params) < threshold else 1 for i in x]
plt.plot(x, y, '-r')
plt.xlabel('Age')
plt.ylabel('Weight')
plt.title('Binary Logistic Regression Full Batch Results')
plt.savefig('logistic_binary_regression.png')

plt.clf()

# Parity plot doesn't make sense, so here is a confusion matrix of the results
# Based on result from https://stackoverflow.com/questions/2148543/how-to-write-a-confusion-matrix-in-python
pred = pd.Series([0 if log_f(i, params) < threshold else 1 for i in x], name='Predicted')
df_confusion = pd.crosstab(data['is_adult'], pred, rownames=['Actual'], colnames=["Predicted"], margins=True)

plt.matshow(df_confusion)
plt.colorbar()
tick_marks = np.arange(len(df_confusion.columns))
#plt.xticks(tick_marks, df_confusion.columns, rotation=45)
#plt.yticks(tick_marks, df_confusion.index)
plt.title('Binary Logistic Regression Full Batch Confusion Matrix')
plt.savefig('binary_confusion.png')

plt.clf()

print("Binary Logistic plot saved to logistic_binary_regression.png")

# ////////// Full Batch Done \\\\\\\\\\


# \\\\\\\\\\ Mini Batches -- 10 Batches //////////


# ===== LINEAR REGRESSION ======

lin_min = minimize(min_squares_f, [1, 1], args=(data[data['x'] < 18]['x'], data[data['x'] < 18]['y'], 'lin_f'), method='GD', tol=1e-5, n_batches=10)
#lin_min
print("Linear Classification completed.")
print(lin_min)

# In[144]:


params = lin_min
#params


# In[145]:

# Plot the results of the Linear Regression!

plt.scatter(data['x'], data['y'])
plt.scatter(test['x'], test['y'])
x = np.linspace(min(rawdata[rawdata['x'] < 18]['x']),max(rawdata[rawdata['x'] < 18]['x']),100)
y = params[0]*x+params[1]
plt.plot(x, y, '-r')
plt.xlabel('Age')
plt.ylabel('Weight')
plt.title('Linear Regression Mini Batch Results')
plt.savefig('linear_regression_mini_batch.png')

plt.clf()

# Parity Plot -- predicted vs actual weights
plt.scatter(data[data['x'] <18]['y'], [params[0]*i+params[1] for i in data[data['x'] < 18]['x']])
plt.scatter(test[test['x'] <18]['y'], [params[0]*i+params[1] for i in test[test['x'] < 18]['x']])
plt.xlabel('Predicted Weight')
plt.ylabel('Actual Weight')
plt.title('Linear Regression Mini Batch Parity')
plt.savefig('parity_lin_regression_mini_batch.png')
print("Linear figure (mini batch) saved to linear_regression_mini_batch.png")

plt.clf()

# ===== ===== =====

# In[146]:

# ===== LOGISTIC REGRESSION =====

log_min = minimize(min_squares_f, [1, 1, 1, 1], args=(data['x'], data['y'], 'log_f'), method='GD', tol=1e-5, n_batches=10)
#log_min
print("Logistic Regression completed.")
print(log_min)


# In[147]:

# Plot the results of the logistic regression!

params = log_min
plt.scatter(data['x'], data['y'])
plt.scatter(test['x'], test['y'])
x = np.linspace(min(rawdata['x']),max(rawdata['x']),100)
y = [(params[0] / (1 + math.exp(-(i - params[2])/params[1]))) + params[3] for i in x]
plt.plot(x, y, '-r')
plt.xlabel('Age')
plt.ylabel('Weight')
plt.title('Logistic Regression Mini Batch Results')
plt.savefig('logistic_regression_mini_batch.png')

plt.clf()

# Parity plot -- predicted vs actual weights
plt.scatter(data['y'], [log_f(i, params) for i in data['x']])
plt.scatter(test['y'], [log_f(i, params) for i in test['x']])
plt.xlabel('Predicted Weight')
plt.ylabel('Actual Weight')
plt.title('Logistic Regression Mini Batch Parity')
plt.savefig('parity_log_regression_mini_batch.png')
print("Logistic plot (mini batch) saved to logistic_regression_mini_batch.png")

plt.clf()


# In[148]:


# ===== BINARY CLASSIFICATION =====

# For this, adjusting the starting parameters helps a lot when generating an accurate sigmoid
# For simplicity, the classifications are done at plot-time
# Threshold is set to 0.5 -- halfway between 0 and 1 (the classifications)
threshold = 0.5

log_min = minimize(min_squares_f, [1, 1, np.mean(data['y']), min(data['is_adult'])], args=(data['y'], data['is_adult'], 'log_f'), method='GD', tol=1e-5, n_batches=10)
#log_min
print("Binary Logistic Regression completed.")
print(log_min)

# In[149]:

# Plot the results of the binary classification!

params = log_min
plt.scatter(data['y'], data['is_adult'])
plt.scatter(test['y'], test['is_adult'])
x = np.linspace(min(rawdata['y']),max(rawdata['y']),100)
y = [0 if log_f(i, params) < threshold else 1 for i in x]
plt.plot(x, y, '-r')
plt.xlabel('Age')
plt.ylabel('Weight')
plt.title('Binary Logistic Regression Mini Batch Results')
plt.savefig('logistic_binary_regression_mini_batch.png')

plt.clf()

# Parity plot doesn't make sense, so here is a confusion matrix of the results
# Based on result from https://stackoverflow.com/questions/2148543/how-to-write-a-confusion-matrix-in-python
pred = pd.Series([0 if log_f(i, params) < threshold else 1 for i in x], name='Predicted')
df_confusion = pd.crosstab(data['is_adult'], pred, rownames=['Actual'], colnames=["Predicted"], margins=True)

plt.matshow(df_confusion)
plt.colorbar()
tick_marks = np.arange(len(df_confusion.columns))
#plt.xticks(tick_marks, df_confusion.columns, rotation=45)
#plt.yticks(tick_marks, df_confusion.index)
plt.title('Binary Logistic Regression Mini Batch Confusion Matrix')
plt.savefig('binary_confusion_mini_batch.png')

print("Binary Logistic plot (mini batch) saved to logistic_binary_regression_mini_batch.png")

plt.clf()

# ////////// Mini Batches Done \\\\\\\\\\


# \\\\\\\\\\ Stochastic //////////

# ===== LINEAR REGRESSION ======

lin_min = minimize(min_squares_f, [1, 1], args=(data[data['x'] < 18]['x'], data[data['x'] < 18]['y'], 'lin_f'), method='GD', tol=1e-5, n_batches=len(data[data['x'] < 18]['x']))
#lin_min
print("Linear Classification completed.")
print(lin_min)

# In[144]:


params = lin_min
#params


# In[145]:

# Plot the results of the Linear Regression!

plt.scatter(data['x'], data['y'])
plt.scatter(test['x'], test['y'])
x = np.linspace(min(rawdata[rawdata['x'] < 18]['x']),max(rawdata[rawdata['x'] < 18]['x']),100)
y = params[0]*x+params[1]
plt.plot(x, y, '-r')
plt.xlabel('Age')
plt.ylabel('Weight')
plt.title('Linear Regression Stochastic Results')
plt.savefig('linear_regression_stochastic.png')

plt.clf()

# Parity Plot -- predicted vs actual weights
plt.scatter(data[data['x'] <18]['y'], [params[0]*i+params[1] for i in data[data['x'] < 18]['x']])
plt.scatter(test[test['x'] <18]['y'], [params[0]*i+params[1] for i in test[test['x'] < 18]['x']])
plt.xlabel('Predicted Weight')
plt.ylabel('Actual Weight')
plt.title('Linear Regression Stochastic Parity')
plt.savefig('parity_lin_regression_stochastic.png')
print("Linear figure (stochastic) saved to linear_regression_stochastic.png")

plt.clf()

# ===== ===== =====

# In[146]:

# ===== LOGISTIC REGRESSION =====

log_min = minimize(min_squares_f, [1, 1, 1, 1], args=(data['x'], data['y'], 'log_f'), method='GD', tol=1e-5, n_batches=len(data['x']))
#log_min
print("Logistic Regression completed.")
print(log_min)


# In[147]:

# Plot the results of the logistic regression!

params = log_min
plt.scatter(data['x'], data['y'])
plt.scatter(test['x'], test['y'])
x = np.linspace(min(rawdata['x']),max(rawdata['x']),100)
y = [(params[0] / (1 + math.exp(-(i - params[2])/params[1]))) + params[3] for i in x]
plt.plot(x, y, '-r')
plt.xlabel('Age')
plt.ylabel('Weight')
plt.title('Logistic Regression Stochastic Results')
plt.savefig('logistic_regression_stochastic.png')

plt.clf()

# Parity plot -- predicted vs actual weights
plt.scatter(data['y'], [log_f(i, params) for i in data['x']])
plt.scatter(test['y'], [log_f(i, params) for i in test['x']])
plt.xlabel('Predicted Weight')
plt.ylabel('Actual Weight')
plt.title('Logistic Regression Stochastic Parity')
plt.savefig('parity_log_regression_stochastic.png')
print("Logistic plot (stochastic) saved to logistic_regression_stochastic.png")


plt.clf()

# In[148]:


# ===== BINARY CLASSIFICATION =====

# For this, adjusting the starting parameters helps a lot when generating an accurate sigmoid
# For simplicity, the classifications are done at plot-time
# Threshold is set to 0.5 -- halfway between 0 and 1 (the classifications)
threshold = 0.5

log_min = minimize(min_squares_f, [1, 1, np.mean(data['y']), min(data['is_adult'])], args=(data['y'], data['is_adult'], 'log_f'), method='GD', tol=1e-5, n_batches=len(data['x']))
#log_min
print("Binary Logistic Regression completed.")
print(log_min)

# In[149]:

# Plot the results of the binary classification!

params = log_min
plt.scatter(data['y'], data['is_adult'])
plt.scatter(test['y'], test['is_adult'])
x = np.linspace(min(rawdata['y']),max(rawdata['y']),100)
y = [0 if log_f(i, params) < threshold else 1 for i in x]
plt.plot(x, y, '-r')
plt.xlabel('Age')
plt.ylabel('Weight')
plt.title('Binary Logistic Regression Stochastic Results')
plt.savefig('logistic_binary_regression_stochastic.png')

plt.clf()

# Parity plot doesn't make sense, so here is a confusion matrix of the results
# Based on result from https://stackoverflow.com/questions/2148543/how-to-write-a-confusion-matrix-in-python
pred = pd.Series([0 if log_f(i, params) < threshold else 1 for i in x], name='Predicted')
df_confusion = pd.crosstab(data['is_adult'], pred, rownames=['Actual'], colnames=["Predicted"], margins=True)

plt.matshow(df_confusion)
plt.colorbar()
tick_marks = np.arange(len(df_confusion.columns))
#plt.xticks(tick_marks, df_confusion.columns, rotation=45)
#plt.yticks(tick_marks, df_confusion.index)
plt.title('Binary Logistic Regression Stochastic Confusion Matrix')
plt.savefig('binary_confusion_stochastic.png')

plt.clf()

print("Binary Logistic plot (stochastic) saved to logistic_binary_regression_stochastic.png")

# ////////// Stochastic Done \\\\\\\\\\


# \\\\\\\\\\ Full Batch with Momentum //////////

# ===== LINEAR REGRESSION ======

lin_min = minimize(min_squares_f, [1, 1], args=(data[data['x'] < 18]['x'], data[data['x'] < 18]['y'], 'lin_f'), method='GD momentum', tol=1e-5)
#lin_min
print("Linear Classification completed.")
print(lin_min)

# In[144]:


params = lin_min
#params


# In[145]:

# Plot the results of the Linear Regression!

plt.scatter(data['x'], data['y'])
plt.scatter(test['x'], test['y'])
x = np.linspace(min(rawdata[rawdata['x'] < 18]['x']),max(rawdata[rawdata['x'] < 18]['x']),100)
y = params[0]*x+params[1]
plt.plot(x, y, '-r')
plt.xlabel('Age')
plt.ylabel('Weight')
plt.title('Linear Regression Full Batch Momentum Results')
plt.savefig('linear_regression_momentum.png')

plt.clf()

# Parity Plot -- predicted vs actual weights
plt.scatter(data[data['x'] <18]['y'], [params[0]*i+params[1] for i in data[data['x'] < 18]['x']])
plt.scatter(test[test['x'] <18]['y'], [params[0]*i+params[1] for i in test[test['x'] < 18]['x']])
plt.xlabel('Predicted Weight')
plt.ylabel('Actual Weight')
plt.title('Linear Regression Full Batch Momentum Parity')
plt.savefig('parity_lin_regression_momentum.png')
print("Linear figure saved to linear_regression_momentum.png")

plt.clf()


# ===== ===== =====

# In[146]:

# ===== LOGISTIC REGRESSION =====

log_min = minimize(min_squares_f, [1, 1, 1, 1], args=(data['x'], data['y'], 'log_f'), method='GD momentum', tol=1e-5)
#log_min
print("Logistic Regression completed.")
print(log_min)


# In[147]:

# Plot the results of the logistic regression!

params = log_min
plt.scatter(data['x'], data['y'])
plt.scatter(test['x'], test['y'])
x = np.linspace(min(rawdata['x']),max(rawdata['x']),100)
y = [(params[0] / (1 + math.exp(-(i - params[2])/params[1]))) + params[3] for i in x]
plt.plot(x, y, '-r')
plt.xlabel('Age')
plt.ylabel('Weight')
plt.title('Logistic Regression Full Batch Momentum Results')
plt.savefig('logistic_regression_momentum.png')

plt.clf()

# Parity plot -- predicted vs actual weights
plt.scatter(data['y'], [log_f(i, params) for i in data['x']])
plt.scatter(test['y'], [log_f(i, params) for i in test['x']])
plt.xlabel('Predicted Weight')
plt.ylabel('Actual Weight')
plt.title('Logistic Regression Full Batch Momentum Parity')
plt.savefig('parity_log_regression_momentum.png')
print("Logistic plot saved to logistic_regression_momentum.png")

plt.clf()

# In[148]:


# ===== BINARY CLASSIFICATION =====

# For this, adjusting the starting parameters helps a lot when generating an accurate sigmoid
# For simplicity, the classifications are done at plot-time
# Threshold is set to 0.5 -- halfway between 0 and 1 (the classifications)
threshold = 0.5

log_min = minimize(min_squares_f, [1, 1, np.mean(data['y']), min(data['is_adult'])], args=(data['y'], data['is_adult'], 'log_f'), method='GD momentum', tol=1e-5)
#log_min
print("Binary Logistic Regression completed.")
print(log_min)

# In[149]:

# Plot the results of the binary classification!

params = log_min
plt.scatter(data['y'], data['is_adult'])
plt.scatter(test['y'], test['is_adult'])
x = np.linspace(min(rawdata['y']),max(rawdata['y']),100)
y = [0 if log_f(i, params) < threshold else 1 for i in x]
plt.plot(x, y, '-r')
plt.xlabel('Age')
plt.ylabel('Weight')
plt.title('Binary Logistic Regression Full Batch Momentum Results')
plt.savefig('logistic_binary_regression_momentum.png')

plt.clf()

# Parity plot doesn't make sense, so here is a confusion matrix of the results
# Based on result from https://stackoverflow.com/questions/2148543/how-to-write-a-confusion-matrix-in-python
pred = pd.Series([0 if log_f(i, params) < threshold else 1 for i in x], name='Predicted')
df_confusion = pd.crosstab(data['is_adult'], pred, rownames=['Actual'], colnames=["Predicted"], margins=True)

plt.matshow(df_confusion)
plt.colorbar()
tick_marks = np.arange(len(df_confusion.columns))
#plt.xticks(tick_marks, df_confusion.columns, rotation=45)
#plt.yticks(tick_marks, df_confusion.index)
plt.title('Binary Logistic Regression Full Batch Momentum Confusion Matrix')
plt.savefig('binary_confusion_momentum.png')

plt.clf()

print("Binary Logistic plot saved to logistic_binary_regression_momentum.png")

# ////////// Full Batch Momentum Done \\\\\\\\\\

print("Completed all three example regressions!")
