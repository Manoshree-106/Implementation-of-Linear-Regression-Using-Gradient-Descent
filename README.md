# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Get the independent variable X and dependent variable Y.
2.Calculate the mean of the X -values and the mean of the Y -values.
3.Find the slope m of the line of best fit using the formula.
<img width="314" height="135" alt="Screenshot 2026-04-27 092214" src="https://github.com/user-attachments/assets/673890b2-e573-4bf8-b3ba-eb3069f212b4" />

4.Compute the y -intercept of the line by using the formula:
<img width="222" height="47" alt="Screenshot 2026-04-27 092221" src="https://github.com/user-attachments/assets/c82e1355-12bc-4269-a6b7-93d14872f02f" />

5.Use the slope m and the y -intercept to form the equation of the line. 6. Obtain the straight line equation Y=mX+b and plot the scatterplot.



## Program:
```
/*
Developed by: MANOSHREE N
RegisterNumber:212225040228
# Step 1: Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Step 2: Create Dataset
data = {
    "Hours_Studied": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Marks_Scored":  [35, 40, 50, 55, 60, 65, 70, 80, 85, 95]
}
df = pd.DataFrame(data)

# Step 3: Prepare Data
X = df["Hours_Studied"].values
y = df["Marks_Scored"].values

# Normalize (optional but helps gradient descent)
X = X / np.max(X)

# Step 4: Initialize parameters
m = 0   # slope
b = 0   # intercept

learning_rate = 0.01
epochs = 1000
n = len(X)

# Step 5: Gradient Descent
for i in range(epochs):
    y_pred = m * X + b
    
    # Compute gradients
    dm = (-2/n) * np.sum(X * (y - y_pred))
    db = (-2/n) * np.sum(y - y_pred)
    
    # Update parameters
    m = m - learning_rate * dm
    b = b - learning_rate * db

# Step 6: Final Model Parameters
print("Slope (m):", m)
print("Intercept (b):", b)

# Step 7: Predictions
y_pred = m * X + b

# Step 8: Visualization
plt.figure(figsize=(8,6))
plt.scatter(X, y, label="Actual Data")
plt.plot(X, y_pred, linewidth=2, label="Regression Line")
plt.xlabel("Hours Studied (Normalized)")
plt.ylabel("Marks Scored")
plt.title("Linear Regression using Gradient Descent")
plt.legend()
plt.grid(True)
plt.show()

# Step 9: Predict custom value
hours = 7.5
hours_norm = hours / 10   # same normalization
predicted_marks = m * hours_norm + b

print(f"\nPredicted marks for {hours} hours = {predicted_marks:.2f}") 
*/
```

## Output:
<img width="894" height="765" alt="Screenshot 2026-04-28 141218" src="https://github.com/user-attachments/assets/331d47e8-d96b-465f-a3a4-83d4754913a4" />



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
