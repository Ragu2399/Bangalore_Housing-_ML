import numpy as np
import pandas as pd


#to intialize pd to show all column in df
pd.set_option("display.max_columns",None)

#importing bengalur csv file
df=pd.read_csv("resources/Bengaluru_House_Data.csv")

#to remove the unwanted colummns from csv
df1=df.drop(["society","availability","area_type"],axis="columns")

#coverting money in lakhs
df2=pd.DataFrame()
df2["prices"]=df1["price"]*100000

#joining two df
df3=pd.concat([df1,df2],axis="columns")

#to check and remove all null features
# print(df3.isnull().sum())
#print(df3.shape)

#dropping all null values
df4=df3.dropna()
# print(df4.head(30))

#to correct the "size" column
df5=df4.copy()
df5["bhk"]=df4["size"].apply(lambda x: int(x.split(" ")[0]))


#to check the total sqrt columns
def is_float(x):
    try:
        float(x)
    except:
        return False
    return True

# df7=df6.copy()
#df7=df6[~df6["total_sqft"].apply(is_float)]

#to return all the range value(235-3563) into single float values in sqrt column
def to_float(x):
    token=x.split("-")
    if len(token)==2:
        return((float(token[0])+float(token[1]))/2)
    try:
        return float(x)
    except:
        return None

df6=df5.copy()
df6["sqrt"]=df5["total_sqft"].apply(to_float)

#to creat a price per sqft column
df7=df6.copy()
df7["price_per_sqrt"]=df6["prices"]/df6["sqrt"]

#to get rid of dimesnional curse in location column during one ot encoding
#first i'm grouping every location
location=df7.groupby("location")["location"].agg("count").sort_values(ascending=False)

#print(location)
#creating a location lesser than 10 in different object
location_less_10=location[location<=10]

#print(location_less_10)

df8=df7.copy()
#we are converting every single location which has less than 10 data points into "others"
df8["locations"]=df7["location"].apply(lambda x: "others" if x in location_less_10 else x)


#to remove datas which has inproper bedroom counts
#since bhk cant be less than 250 sqrt
df9=df8[~((df8.sqrt/df8.bhk)<300)]

#print(df9.price_per_sqrt.describe())
# to remove values which are so far away from mean using standard deviation
def remove_outliers(df):
    a=pd.DataFrame()
    # new=pd.DataFrame
    #grouping the data frame by location
    for key,subdf in df.groupby("location"):
        #the groped location is stored on subdf data frame
        #calculating mean  and standard deviation
        mean=np.mean(subdf["price_per_sqrt"])
        std = np.std(subdf["price_per_sqrt"])
        #making a "new" data frame where it has haves which are in the range above (mean-std) and below (mean+std)
        #consider as a bell curve to visvalize mean
        new=subdf[(subdf.price_per_sqrt>(mean-std)) & (subdf.price_per_sqrt<=(mean+std))]
        #now we are concating the new data frame to the pre  defined empty data from
        # for every iteration we are storing the value in a
        a=pd.concat([a,new],ignore_index=True)

    return a

df10=remove_outliers(df9)


#next i'm removing the houses which has "2" bathrooms more than there no of bedrooms
df11=df10[(df10.bath)<(df10.bhk+2)]



#now we are droping al the unwanted columns
df12=df11.drop(["location","size","total_sqft","price","price_per_sqrt"],axis="columns")

# print(df12.head(50))




#next step is to make one hot encoding for locations column
#since we reduced the number of locations by giving "others" in certain values
#first we rae getting dummies in "dummies" object
dummies=pd.get_dummies(df12.locations)
#then we are going to concatinate withe the old data frame
df13=pd.concat([df12,dummies],axis="columns")

# print(df13)

#now we are removing the locations column since we made dummies of it
df14=df13.drop(["locations"],axis="columns")
# print(df14.head())
# print(df11.shape)
# print(df11)


#now we are splitting into target and input datas
#let the input be "x" and target be "y"

x=df14.drop(["prices"],axis="columns")
#print(x)

y=df14["prices"]
#print(y)

#now making train test split
# from sklearn.model_selection import train_test_split
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
#
# from sklearn.linear_model import LinearRegression
# model=LinearRegression()
# model.fit(x_train,y_train)
# print(model.score(x_test,y_test))

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression


#making a disctionary for model and paramters
model_params = {
    'lasso': {
        'model':Lasso(max_iter=100,tol=0.1),
        'params':{
            'alpha':[20,30,40,50]
        }
    },
    'Ridge': {
        'model': Ridge(max_iter=100),
        'params': {
            'alpha': [20, 30, 40],
            'tol': [0.1, 0.2, 0.4],
            'max_iter':[1000,2000,3333]
        }
    },
    'linear_regression': {
        'model': LinearRegression(),
        'params': {
            'n_jobs':[-1,1]

        }
    }

}



scores = []
best_model={}

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)

from sklearn.model_selection import GridSearchCV
for model_name, mp in model_params.items():
    clf = GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False)
    clf.fit(x_train,y_train)
    scores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
    best_model[model_name] = clf.best_estimator_


d = pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])
print(d)
# print(x)

def predict_price(locations,sqrt,bhk,balcony,bath,x):
    loc_index=np.where(x.columns==locations)[0][0]

    x=np.zeros(len(x.columns))
    x[3]=sqrt
    x[2]=bhk
    x[1]=balcony
    x[0]=bath

    if loc_index >=0:
        x[loc_index]=1

    return linear.predict([x])[0]


#to use the function to predict datas

from sklearn.linear_model import LinearRegression
linear=LinearRegression(n_jobs=-1)
linear.fit(x_train,y_train)
print(linear.score(x_test,y_test))
print(predict_price("Rajaji Nagar",1100,3,3,3,x))
price=predict_price("Thanisandra",1170,3,3,3,x)/100000
print(str(price)+" lakhs")
#


best_training_model=best_model["lasso"]
#to import the model
import joblib
joblib.dump(best_training_model,"bengaluru house.pkl")