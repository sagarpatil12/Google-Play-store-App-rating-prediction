import numpy as np
import pandas as pd

#Function convert app size in KB to MB equivalent aand replace value ="Varies with device" with 0.
def convert(x):
    print(len(x))
    summ = 0.0
    count = 0
    newList = []
    for ele in x:
        if('k' in ele):
            ele = float(ele.replace('k',''))/1000        
        
        if(ele!="Varies with device"):
            summ = summ + float(ele)
            count = count + 1
            newList.append(float(ele))
        if(ele == "Varies with device"):
            newList.append(ele)
    
    nle1 = []
    for ele in newList:
        if ele=="Varies with device":
            ele = 0
        nle1.append(ele)
    print(len(nle1))
    return nle1
            
#Loading Dataset    
dataset = pd.read_csv('googleplaystore.csv')
#Dropping rows with nan values
dataset = dataset.dropna()

#Excluding unnecessary columns
x = dataset.loc[:,dataset.columns!='Rating']
x = x.loc[:,x.columns!='Last Updated']
x = x.loc[:,x.columns!='Current Ver']
x = x.loc[:,x.columns!='Type']

#loading input parametets
x = x.iloc[:,1:]

#loading output parameters (Ratings)
y = dataset.iloc[:, 2].values

#Encoding data
from sklearn.preprocessing import LabelEncoder
lableEncode = LabelEncoder()
x.values[:, 0] =lableEncode.fit_transform(x.values[:, 0])
x.values[:,-1]=lableEncode.fit_transform(x.values[:, -1])
x.values[:,5]=lableEncode.fit_transform(x.values[:, 5])
x.values[:,6]=lableEncode.fit_transform(x.values[:, 6])

#Modify columns having characters in thier values , Convert them to numeric form
x.Size=x.Size.str.replace('M','')
x.Size = convert(x.Size)
x.Installs=(x.Installs.str.replace('+',''))
x.Installs=pd.to_numeric(x.Installs.str.replace(',',''),errors='coerce')
x.Price=pd.to_numeric(x.Price.str.replace('$',''),errors='coerce')
x.Reviews=pd.to_numeric(x.Reviews,errors='coerce')

#splitting the dataset
from sklearn.model_selection import train_test_split
x_training,x_testing,y_training,y_testing = train_test_split(x,y,test_size=0.3, random_state=0)

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
x_training = sc_X.fit_transform(x_training)
x_testing = sc_X.transform(x_testing)

#Applying Random Forest Regression
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 70, random_state = 60)
regressor.fit(x_training, y_training)
y_pred = regressor.predict(x_testing)

from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_testing,y_pred))
