import numpy as np
import pandas as pd
import xlrd 
import matplotlib.pyplot as plt
from keras.layers import Dense, Activation, Dropout
from keras.models import Sequential
from keras.models import model_from_json
from sklearn.preprocessing import StandardScaler
from keras.callbacks import EarlyStopping
from keras import backend
from keras import optimizers
import h5py
import sklearn.metrics, math
from sklearn import model_selection
from sklearn.linear_model import LinearRegression  
from sklearn.utils import check_array
from keras import regularizers

#-----------------------------------------------------------------------------
#  Custom Loss Functions
#-----------------------------------------------------------------------------

# root mean squared error (rmse) for regression
def rmse(y_obs, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_obs), axis=-1))

# mean squared error (mse) for regression
def mse(y_obs, y_pred):
    return backend.mean(backend.square(y_pred - y_obs), axis=-1)

# coefficient of determination (R^2) for regression
def r_square(y_obs, y_pred):
    SS_res =  backend.sum(backend.square(y_obs - y_pred)) 
    SS_tot = backend.sum(backend.square(y_obs - backend.mean(y_obs))) 
    return (1 - SS_res/(SS_tot + backend.epsilon()))

def mean_absolute_percentage_error(y_obs, y_pred): 
    y_obs, y_pred = np.array(y_obs), np.array(y_pred)
    y_obs=y_obs.reshape(-1,1)
    #y_obs, y_pred =check_array(y_obs, y_pred)
    return  np.mean(np.abs((y_obs - y_pred) / y_obs)) * 100

#-----------------------------------------------------------------------------
#   Data Preparation
#-----------------------------------------------------------------------------

### add address of the dataset
loc = ("https://github.com/CIDARLAB/neural-optimizer/transfer_learning/dataset/DAFD_final_cutoff.xlsx")

### Read data
wb = xlrd.open_workbook(loc) 
sheet = wb.sheet_by_index(0)

### Extract input and output
X=[]
Y=[]
for i in range(1,sheet.nrows): ## data
    X.append(sheet.row_values(i))
    Y.append(sheet.cell_value(i,12)) ## Regimes
	
X=np.array(X)
X=X[:,1:9] # Geometry Features
Y=np.array(Y) # Regime labels

X1=[] #Regime 1 data-set
X2=[] #Regime 2 data-set
Y11=[] # Regime 1 Output 1 (generation rate)
Y12=[] # Regime 1 Output 2 (size)
Y21=[] # Regime 2 Output 1 (generation rate)
Y22=[] # Regime 2 Output 2 (size)

for i in range(len(Y)):
    if Y[i]==1 :
        X1.append(X[i,:])
        Y11.append(sheet.cell_value(i+1,10))
        Y12.append(sheet.cell_value(i+1,11))
    
    elif Y[i]==2 :
        X2.append(X[i,:])
        Y21.append(sheet.cell_value(i+1,10))
        Y22.append(sheet.cell_value(i+1,11))

###data scaling
X2 = StandardScaler().fit_transform(X2)

###train-test split
validation_size = 0.20

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X2, Y22, test_size=validation_size) #Regime 2 Output 2

X_train =np.array(X_train)
Y_train=np.array(Y_train)
X_test =np.array(X_test)
Y_test =np.array(Y_test)
Y22 = np.array(Y22)
#-----------------------------------------------------------------------------
#   Training Nueral Network Model
#-----------------------------------------------------------------------------

### Initializing NN
model = Sequential()
### Adding the input layer and the first hidden layer
model.add(Dense(units = 16, activation = 'relu', input_dim = 8 , name='dense_1'))#, kernel_regularizer=regularizers.l2(0.001))) #more options to avoid over-fitting
### Droput to avoid further over-fitting
#model.add(Dropout(0.3))
### Adding the second hidden layer
model.add(Dense(units = 16, activation = 'relu', name='dense_2'))#, kernel_regularizer=regularizers.l2(0.001)))
#model.add(Dropout(0.3))
### Adding the third hidden layer
model.add(Dense(units = 16, activation = 'relu', name='dense_3'))#, kernel_regularizer=regularizers.l2(0.001)))
#model.add(Dropout(0.3))
### Adding the 4th hidden layer
model.add(Dense(units = 8, activation = 'relu', name='dense_4'))#, kernel_regularizer=regularizers.l2(0.001)))
#model.add(Dropout(0.3))
### Adding the output layer
model.add(Dense(units = 1, name='dense_5'))#, kernel_regularizer=regularizers.l2(0.001)))

### Optimizer 
adam=optimizers.Adam()#(beta_1=0.9, beta_2=0.999, amsgrad=False)

### Compiling the NN
model.compile(optimizer = adam, loss = 'mean_squared_error',metrics=['mean_squared_error', rmse, r_square] )

### Early stopping
earlystopping=EarlyStopping(monitor="mean_squared_error", patience=20, verbose=1, mode='auto')

### Fitting the model to the train set
result = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size = 16, epochs = 400, callbacks=[earlystopping])

#-----------------------------------------------------------------------------
#   Predictions of the Trained Nueral Network Model
#-----------------------------------------------------------------------------

### Test-set prediction
y_pred = model.predict(X_test)
### train-set prediction
y_pred_train = model.predict(X_train)
### total data-set prediction
y_pred_total = model.predict(X2)

#-----------------------------------------------------------------------------
#   Save the Trained Nueral Network Model
#-----------------------------------------------------------------------------

# Save model weights
model.save_weights('Y22_weights.h5')

##-----------------------------------------------------------------------------
##  Plot Predictions and Learning Curves 
##-----------------------------------------------------------------------------

### Test-set Prediction
plt.plot(Y_test, color = 'blue', label = 'Real data')
plt.plot(y_pred, color = 'red', label = 'Predicted data')
plt.title('Prediction')
plt.legend()
plt.show()
plt.close() 
### Predicted VS Observed 
plt.scatter(Y_test, y_pred, color='red', label= 'Predicted data')
plt.plot(Y_test, Y_test, color='blue', linewidth=2,label = 'y=x')
plt.xlabel('observed')
plt.ylabel('predicted')
plt.show()
plt.close()
### Learning curve for RMSE  
plt.plot(result.history['rmse'])
plt.plot(result.history['val_rmse'])
plt.title('rmse')
plt.ylabel('rmse')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.close() 
### Learning curve for MSE
plt.plot(result.history['mean_squared_error'])
plt.plot(result.history['val_mean_squared_error'])
plt.title('loss function')
plt.ylabel('mean squared error')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()
plt.close() 

##-----------------------------------------------------------------------------
##  statistical Summary
##-----------------------------------------------------------------------------

print("\n")
print("Mean absolute error (MAE) for test-set:      %f" % sklearn.metrics.mean_absolute_error(Y_test,y_pred))
print("Mean squared error (MSE) for test-set:       %f" % sklearn.metrics.mean_squared_error(Y_test,y_pred))
print("Root mean squared error (RMSE) for test-set: %f" % math.sqrt(sklearn.metrics.mean_squared_error(Y_test,y_pred)))
print("R square (R^2) for test-set:                 %f" % sklearn.metrics.r2_score(Y_test,y_pred))

print("\n")
print("Mean absolute error (MAE) for train-set:      %f" % sklearn.metrics.mean_absolute_error(Y_train, y_pred_train))
print("Mean squared error (MSE) for train-set:       %f" % sklearn.metrics.mean_squared_error(Y_train, y_pred_train))
print("Root mean squared error (RMSE) for train-set: %f" % math.sqrt(sklearn.metrics.mean_squared_error(Y_train, y_pred_train)))
print("R square (R^2) for train-set:                 %f" % sklearn.metrics.r2_score(Y_train, y_pred_train))

print("\n")
print("Mean absolute error (MAE) for total data-set:      %f" % sklearn.metrics.mean_absolute_error(Y22,y_pred_total))
print("Mean squared error (MSE) for total data-set:       %f" % sklearn.metrics.mean_squared_error(Y22,y_pred_total))
print("Root mean squared error (RMSE) for total data-set: %f" % math.sqrt(sklearn.metrics.mean_squared_error(Y22,y_pred_total)))
print("R square (R^2) for total data-set:                 %f" % sklearn.metrics.r2_score(Y22,y_pred_total))