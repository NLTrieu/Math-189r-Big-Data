
# coding: utf-8

# In[ ]:

import pandas as pd
import numpy as np
import scipy
import random
import matplotlib.pyplot as plt
import math
import datetime
import copy


# In[ ]:

def convertToDateTime(date_str,time_str):
    date_array = date_str.split('/')
    month = int(date_array[0])
    day = int(date_array[1])
    year = int(date_array[2])
    h = int(time_str[0:2])
    m = int(time_str[2:])
    return datetime.datetime(year,month,day,hour=h,minute=m)

def incrementDate(curr_datetime):
    end_of_day = datetime.datetime(curr_datetime.year, curr_datetime.month,curr_datetime.day,hour=15,minute=59)
    curr_datetime = curr_datetime + datetime.timedelta(minutes=1)
    if (curr_datetime > end_of_day):
        curr_datetime = curr_datetime - datetime.timedelta(minutes=1)
        curr_datetime = datetime.datetime(curr_datetime.year, curr_datetime.month,curr_datetime.day,hour=9,minute=30)
        daysDelta = 1
        if (curr_datetime.isoweekday()==5):
            daysDelta = 3
        curr_datetime = curr_datetime + datetime.timedelta(days=daysDelta)
    return curr_datetime

def fillData(latest_start_datetime,earliest_end_datetime,inputfilename,outputfilename):
    print("Starting work for file " + inputfilename)
    
    f = open(inputfilename, 'r')
    outputfile = open(outputfilename,'w')
    

    curr_datetime = copy.copy(latest_start_datetime)
    f.readline()
    while curr_datetime <= earliest_end_datetime:
        line = f.readline()
        array = line.split(',')
        date = array[0]
        time = array[1]
        data_date = convertToDateTime(date,time)
        
        if (data_date < curr_datetime):
            continue
            
        while(curr_datetime < data_date):
            month_val = curr_datetime.month
            day_val = curr_datetime.day
            month_prefix = ""
            day_prefix = ""
            if month_val < 10:
                month_prefix="0"
            if day_val < 10:
                day_prefix = "0"
            array[0] = month_prefix+str(curr_datetime.month)+"/"+day_prefix+str(curr_datetime.day)+"/"+str(curr_datetime.year)
            hour_val = curr_datetime.hour
            hour_prefix = ""
            if (hour_val < 10):
                hour_prefix = "0"
            minute_val = curr_datetime.minute
            minute_prefix=""
            if (minute_val<10):
                minute_prefix="0"
            array[1] = hour_prefix+str(hour_val)+minute_prefix+str(curr_datetime.minute)
            outputfile.write(",".join(array))
            curr_datetime = incrementDate(curr_datetime)
            
        if (data_date == curr_datetime):
            outputfile.write(line)
            curr_datetime = incrementDate(curr_datetime)
            
    outputfile.close()
    f.close()
    
def addFeatures(origfilename,modfilename):
    print("Starting work for file: " + origfilename)
    orig = open(origfilename,'r')
    mod = open(modfilename,'w')
    new_vals = []
    last_close = 0
    for line in orig:
        array = line.split(',')
        date = array[0]
        time = int(array[1])
        op = float(array[2])
        high = float(array[3])
        low = float(array[4])
        close = float(array[5])
        vol = float(array[6])
        OC_Dif = op-close
        if (len(new_vals)!=0):
            OC_Dif = abs(close-last_close)

        OC_Mag = abs(OC_Dif)
        HL_Mag = abs(high-low)
        OC_Dir = np.sign(OC_Dif)
        OC_Chain = OC_Dir
        OC_PerMag = OC_Mag/op * 100
        OC_PerChain = OC_PerMag
        if (len(new_vals) != 0 and OC_Dir == new_vals[2]):
            OC_Chain = OC_Dir + new_vals[3]
            OC_PerChain = OC_PerChain + new_vals[5]

        new_vals = [OC_Mag,HL_Mag,OC_Dir,OC_Chain,OC_PerMag,OC_PerChain]
        mod_line = line.rstrip()
        for val in new_vals:
            mod_line = mod_line + ","+str(val)
        mod_line = mod_line + "\n"
        last_close = close
        mod.write(mod_line)
    orig.close()
    mod.close()
def checkForSplitsAndMultiples(filename):
    f = open(filename,'r')
    previous_close = 0
    first = True
    for line in f:
        array = line.split(',')
        close = float(array[5])
        if first:
            first = False
            previous_close = close
            continue
        if (close > 2*previous_close-close/20):
            print("Doubled: " + stockname)
        if (close < previous_close/2+close/5):
            print("Halved: " + stockname)
        previous_close = close
    f.close()


# In[ ]:

transport_stocks = ["AAL","ALK","CAR","CHRW","CSX","DAL","EXPD","FDX","IYT","JBHT","JBLU","KEX","KSU","LSTR","LUV","NSC","R","UAL","UNP","UPS"]


# In[ ]:

# check for splits
for stockname in transport_stocks:
    filename = "C:\\Users\\Nic\\Desktop\\transport_data_modded\\"+stockname+"_filled.txt"
    checkForSplitsAndMultiples(filename)
spyfilename = "C:\\Users\\Nic\\Desktop\\etf_data\\SPY_filled.txt"
checkForSplitsAndMultiples(spyfilename)


# In[ ]:

# Fill in time gaps
transport_stocks = ["AAL","ALK","CAR","CHRW","CSX","DAL","EXPD","FDX","IYT","JBHT","JBLU","KEX","KSU","LSTR","LUV","NSC","R","UAL","UNP","UPS"]
latest_start_date = "05/03/2007"
latest_start_time = "0932"
latest_start_datetime = convertToDateTime(latest_start_date,latest_start_time)
earliest_end_date = "10/07/2016"
earliest_end_time = "1559"
earliest_end_datetime = convertToDateTime(earliest_end_date,earliest_end_time)
for stockname in transport_stocks:
    inputfilename = "C:\\Users\\Nic\\Desktop\\transport_data\\"+stockname+".txt"
    outputfilename = "C:\\Users\\Nic\\Desktop\\transport_data_modded\\"+stockname+"_filled.txt"
    fillData(latest_start_datetime,earliest_end_datetime,inputfilename,outputfilename)
spyinput = "C:\\Users\\Nic\\Desktop\\etf_data\\SPY.txt"
spyoutput = "C:\\Users\\Nic\\Desktop\\etf_data\\SPY_filled.txt"
fillData(latest_start_datetime,earliest_end_datetime,spyinput,spyoutput)


# In[ ]:

# Add new features
new_features = ["OC_Mag","HL_Mag","OC_Dir","OC_Chain","Per_Change","Per_Chain"]
for stockname in transport_stocks:
    origfilename = "C:\\Users\\Nic\\Desktop\\transport_data_modded\\"+stockname+"_filled.txt"
    modfilename = "C:\\Users\\Nic\\Desktop\\transport_data_modded\\"+stockname+"_filled_modded.txt"
    addFeatures(origfilename,modfilename)
spyorig = spyoutput
spymod = "C:\\Users\\Nic\\Desktop\\etf_data\\SPY_filled_modded.txt"
addFeatures(spyoutput,spymod)


# In[ ]:

# Load data
# non-numeric     0     1      2      3      4      5       6         7        8        9         10          11        12
allfeatures = ["Date","Time","Open","High","Low","Close","Volume","OC_Mag","HL_Mag","OC_Dir","OC_Chain","Per_Change","Per_Chain"]
# numeric only          0      1      2      3      4       5         6        7        8          9          10        11
ignore_cols = [0,3,4,5,8,9,11]
data_multi = None
data_open = None
data_perchange = None
first = True
transport_stocks = ["AAL","ALK","CAR","CHRW","CSX","DAL","EXPD","FDX","IYT","JBHT","JBLU","KEX","KSU","LSTR","LUV","NSC","R","UAL","UNP","UPS"]
for stockname in transport_stocks:
    mod = open("C:\\Users\\Nic\\Desktop\\transport_data_modded\\"+stockname+"_filled_modded.txt",'r')
    list_result = []
    list_open = []
    list_perchange = []
    for line in mod:
        array = line.split(',')
        numeric_data = list(map(float,array[1:]))
        list_result.append(numeric_data)
        list_open.append(numeric_data[1])
        list_perchange.append(numeric_data[10])#*numeric_data[8])
    mod.close()
    multifeatures = np.delete(list_result, ignore_cols, axis = 1).tolist()
    if first:
        data_multi = multifeatures
        data_open = [list_open]
        data_perchange = [list_perchange]
        first = False
        print("first")
        continue
    else:
        print("appending")
        data_multi = np.append(data_multi,multifeatures,axis = 1)
        data_open = np.append(data_open,[list_open],axis=0)
        data_perchange = np.append(data_perchange,[list_perchange],axis=0)
    mod.close()
print("finished with transport stocks")
SPY_file = open("C:\\Users\\Nic\\Desktop\\etf_data\\SPY_filled_modded.txt",'r')
y_pc = []
for line in SPY_file:
    array = line.split(',')
    y_pc.append(float(array[11]))#*float(array[9])) # append pc
SPY_file.close()
y_pc = np.matrix(y_pc).reshape(-1,1)
data_multi = np.matrix(data_multi)
data_open = np.matrix(data_open).T
data_perchange = np.matrix(data_perchange).T

data_multi = np.hstack((np.ones_like(y_pc),data_multi))
data_open = np.hstack((np.ones_like(y_pc),data_open))
data_perchange = np.hstack((np.ones_like(y_pc),data_perchange))

data_multi = data_multi.getA()
data_open = data_open.getA()
data_perchange = data_perchange.getA()
y_pc = y_pc.getA()


# In[ ]:

SPY_file = open("C:\\Users\\Nic\\Desktop\\etf_data\\SPY_filled_modded.txt",'r')
y_pc = []
for line in SPY_file:
    array = line.split(',')
    y_pc.append(float(array[11])) # append pc
SPY_file.close()
y_pc = np.matrix(y_pc).reshape(-1,1)
y_pc = y_pc.getA()


# In[ ]:

SPY_file = open("C:\\Users\\Nic\\Desktop\\etf_data\\SPY_filled_modded.txt",'r')
y_open = []
for line in SPY_file:
    array = line.split(',')
    y_open.append(float(array[2])) # append open
SPY_file.close()
y_open = np.matrix(y_open).reshape(-1,1)
y_open = y_open.getA()


# In[ ]:

# Check that the dimensions align, adjust feature set
print(data_multi.shape)
print(data_open.shape)
print(data_perchange.shape)
print(y_pc.shape)
features = np.delete(allfeatures,list(map(lambda x: x+1,ignore_cols))+[0]).tolist()
print(features)


# In[ ]:

def objective(X, y, w, reg=1e-6):
    err = X @ w - y
    err = math.sqrt(float(err.T @ err))
    return (err + reg * np.abs(w).sum())/len(y)

def grad_objective(X, y, w):
    return X.T @ (X @ w - y) / len(y)

def prox(x, gamma):
    #lasso proximal operator.
    #Note: modifies x in-place
    x[np.abs(x) <= gamma] = 0.
    x[x > gamma] = x[x > gamma] - gamma
    x[x < -gamma] = x[x < -gamma] + gamma
    return x

def lasso_grad(
    X, y, reg=1e-6, lr=1e-3, tol=1e-6,
    max_iters=300, batch_size=256, eps=1e-5,
    verbose=False, print_freq=5,
):
    y = y.reshape(-1,1)
    w = np.linalg.solve(X.T @ X, X.T @ y)
    
    ind = np.random.randint(0, X.shape[0], size=batch_size)
    obj = [objective(X[ind], y[ind], w, reg=reg)]
    grad = grad_objective(X[ind], y[ind], w)
    
    while len(obj)-1 <= max_iters and np.linalg.norm(grad) > tol:
        ind = np.random.randint(0, X.shape[0], size=batch_size)
        grad = grad_objective(X[ind], y[ind], w)
        w = prox(w - lr * grad, reg*lr)
        obj.append(objective(X[ind], y[ind], w, reg=reg))
    return w, obj

def lasso_path(
    X, y, reg_min=1e-8, reg_max=10,
    regs=10, **grad_args
):
    W = np.zeros((X.shape[1], regs))
    tau = np.linspace(reg_min, reg_max, regs)
    for i in range(regs):
        W[:,i] = lasso_grad(
            X, y, reg=1/tau[i], max_iters=1000,
            batch_size=1024, **grad_args
        )[0].flatten()
    return tau, W


# In[ ]:

# Run lasso on data
tau, W = lasso_path(data_perchange, y_pc, reg_min=1e-15, reg_max=0.02, regs=20, lr=1e-12)


# In[ ]:

plt.xlabel("tau = lambda^(-1)")
plt.ylabel("w_i")
plt.plot(tau, W[:,:].T)
plt.show()


# In[ ]:

# find most important features
transport_stocks_i = ["AAL - A","ALK - A","CAR - Rent","CHRW - T","CSX - R","DAL - A","EXPD - D","FDX - D","IYT - INDEX","JBHT - T","JBLU - A","KEX - M","KSU - R","LSTR - T","LUV - A","NSC - R","R - Transport","UAL - A","UNP - R","UPS - D"]
important_features=np.argsort([abs(weight) for weight in W[:,-1]])
sorted_weights = np.sort([abs(weight) for weight in W[:,-1]])
numFeatures = len(important_features)
for i in range(numFeatures)[::-1]:
    featureNum = important_features[i]-1
    count = 0
    #while featureNum >= len(features):
    #    featureNum = featureNum - len(features)
    #    count = count + 1
    if (featureNum >= 0):
        print(transport_stocks_i[featureNum] +" "+ str(sorted_weights[i]))
        #print(transport_stocks[count] + ": " + features[featureNum])


# In[ ]:



