# importing required libraries
print("importing required libraries......")
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics
import datetime
import os

def train_model():
    # It takes asin number and forecasts the ranks using time series data
    def getStatisticalRank(asins):
        forecast = []

        for asin in asins:
            #print(asin,end=" ")
            if(len(asin) < 10):                 #asin number should be of length 10
                asin = '0'*(10-len(asin)) + str(asin)    #adding leading zeros
            path = os.path.join(os.getcwd(), f'Datasets\Ranks_Data_V2_1.1\{asin}_com_norm.json')
            #path = f'C:/Books Forecast/Datasets/Ranks_Data/{asin}_com_norm.json'
            try:
                queried_books = pd.read_json(path,orient='index')
            except:
                forecast.append(2500000) #temporary solution if the file is not found
                #print()
            else:
                queried_books.reset_index(inplace=True)
                queried_books.rename( columns={0:'Rank','index':'Date'}, inplace=True )
                queried_books['Step'] = queried_books.index
                series = queried_books['Rank'].astype(float).values
                time = queried_books['Step'].values
                split_time = int(len(queried_books)*0.8)
                naive_forecast = series[split_time - 1:-1]
                ans = int(queried_books['Rank'][len(queried_books)-1])
                #print(ans)
                forecast.append(ans)
        return pd.DataFrame(forecast)

    #It takes ranks as input and gives sales count as output using ML model
    def getSalesCount(ranks):
        path = (os.path.join(os.getcwd(),'Datasets\ranks_to_sales.csv'))
        ranks_to_sales = pd.read_csv('C:\Books Forecast\Datasets\Ranks_to_sales.csv')
        X = ranks_to_sales.drop(columns='sales')
        Y = ranks_to_sales['sales']
        regressor = XGBRegressor()
        regressor.fit(X,Y)
        regressor.save_model("model_sales_count.txt")
        prediction = regressor.predict(ranks)
        return pd.DataFrame(prediction.astype(int))


    #loading the dataset from CSV file to a Pandas DataFrame
    print("loading the dataset from CSV file to a Pandas DataFrame......")
    path = (os.path.join(os.getcwd(),'Datasets\BOOKS_DATA_MINED_CLEANED_V2_1.1.csv'))
    books = pd.read_csv(path)

    #Splitting the published date column into day, month and year
    books[["DAY", "MONTH", "YEAR"]] = books["PUBLISHED_DATE"].str.split("-", expand = True)

    #Converting the datatype of day, month and year to numeric type
    books[["DAY", "MONTH","YEAR"]] = books[["DAY", "MONTH","YEAR"]].apply(pd.to_numeric)

    #dropping the published date column
    books.drop(columns=["PUBLISHED_DATE"],inplace=True)

    #Filling the NA values with the mean of the respective column
    books['YEAR'].fillna(int(books['YEAR'].mean()),inplace=True)
    books['MONTH'].fillna(int(books['MONTH'].mean()),inplace=True)

    #prints categorie value counts
    #books.CATEGORIES.value_counts()
    
    #storing the ASIN number for further use
    res = books['ISBN_13'].tolist()

    #passing the asin number to getSatisticalRank function to get Ranks and then to salescount
    print("passing the asin number to getSatisticalRank function to get Ranks and then to salescount......")
    books_asins = books['ASIN']
    books_ranks = getStatisticalRank(books_asins)
    books_sales = getSalesCount(books_ranks)
    #print(books_sales.value_counts()) 

    # adding column SALES_RANK and SALES_COUNT to books dataframe
    books['SALES_RANK'] = books_ranks
    books['SALES_COUNT'] = books_sales

    #encoding categorical string values to numeric of books dataframe
    encoder = LabelEncoder()
    books['FORMAT'] = encoder.fit_transform(books['FORMAT'])
    books['AUTHOR'] = encoder.fit_transform(books['AUTHOR'])
    books['PUBLISHER'] = encoder.fit_transform(books['PUBLISHER'])
    books['CATEGORIES'] = encoder.fit_transform(books['CATEGORIES'])
    books['LANGUAGE'] = encoder.fit_transform(books['LANGUAGE'])
    books['PRINT_TYPE'] = encoder.fit_transform(books['PRINT_TYPE'])
    books['IS_EBOOK'] = encoder.fit_transform(books['IS_EBOOK'])


    #dropping ASIN and ISBN_13 column of books as they are an identifier
    books.drop(columns=['ASIN','ISBN_13'],inplace=True)

    #splitting the books dataframe for training the ML model into data and label(SALES_COUNT)
    X = books.drop(columns='SALES_COUNT',axis=1)
    Y = books['SALES_COUNT']

    #splitting into train and test for evaluation purpose
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=2)

    #initializing and training the model
    print("initializing and training the model......")
    regressor = XGBRegressor()
    regressor.fit(X_train,Y_train)
    regressor.save_model("model_forecast.txt")


    #Prediction on Training Data
    training_data_prediction = regressor.predict(X_train)

    # R squared Value
    r2_train = metrics.r2_score(Y_train,training_data_prediction)
    print('Train data R Squared Value',r2_train)

    #Prediction on Test Data
    test_data_prediction = regressor.predict(X_test)

    # R squared Value
    r2_test = metrics.r2_score(Y_test,test_data_prediction)
    print('Test data R Squared Value',r2_test)

import time
start_time = time.time()
train_model()
print("Train time --- %s seconds ---" % (time.time() - start_time))