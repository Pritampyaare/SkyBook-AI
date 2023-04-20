# importing required libraries
print("importing required libraries......")
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics
import datetime
import time
from os.path import exists
import os.path as path
import os

def forecast(input_string):

    start_time = time.time()
    print("**************************************************************************************************************")

    #check if the file is older or does not exists
    if(not exists("model_sales_count.txt") or is_file_older_than_x_days("model_sales_count.txt",1)):
        print("#############Training Model#############")
        import training_V2
        #train_model.train_model()
        print("#############Training Ended#############")

    isbn_13 = []
    categories = ""
    if(input_string.isnumeric()):
        isbn_13.append(int(input_string))
    else:
        categories = input_string

    print("Searching for ",input_string)
    
    # It takes asin number and forecasts the ranks using time series data
    def getStatisticalRank(asins):
        forecast = []

        for asin in asins:
            #print(asin,end=" ")
            if(len(asin) < 10):                 #asin number should be of length 10
                asin = '0' * (10-len(asin)) + str(asin)    #adding leading zeros
            path = os.path.join(os.getcwd(), f'Datasets\Ranks_Data_V2_1.1\{asin}_com_norm.json')
            #path = f'C:/Books Forecast/Datasets/Ranks_Data/{asin}_com_norm.json'
            try:
                queried_books = pd.read_json(path, orient='index')
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
        regressor = XGBRegressor()
        regressor.load_model("model_sales_count.txt")
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

    #forecasting for books of particular category
    #queried_books = books.query(f"CATEGORIES == '{categories}'")
    #print(queried_books.head())
    queried_books = books.query(f"CATEGORIES == '{categories}' or ISBN_13 in @isbn_13 ")

    #storing the ASIN number for further use
    res = queried_books['ISBN_13'].tolist()

    #passing the asin number of queried_books to getSatisticalRank function to get Ranks and to salescount
    print("passing the asin number of queried_books to getSatisticalRank function to get Ranks and to salescount......")
    queried_books_asins = queried_books['ASIN']
    queried_books_ranks = getStatisticalRank(queried_books_asins)
    #print(queried_books_ranks)
    queried_books_sales = getSalesCount(queried_books_ranks)
    #print(queried_books_sales) 

    #gets the ranks and sales
    SALES_RANK = queried_books_ranks[0]
    SALES_COUNT = queried_books_sales[0]

    #it sets the index of queried_books according to queried_books_ranks and sales
    queried_books = queried_books.set_index(queried_books_ranks.index)
    queried_books = queried_books.set_index(queried_books_sales.index)

    # adding column SALES_RANK and SALES_COUNT to queired_books dataframe
    queried_books['SALES_RANK'] = queried_books_ranks
    queried_books['SALES_COUNT'] = queried_books_sales
    #print(books)
    #print(queried_books)

    encoder = LabelEncoder()
    #encoding categorical string values to numeric of queired_books dataframe
    queried_books['FORMAT'] = encoder.fit_transform(queried_books['FORMAT'])
    queried_books['AUTHOR'] = encoder.fit_transform(queried_books['AUTHOR'])
    queried_books['PUBLISHER'] = encoder.fit_transform(queried_books['PUBLISHER'])
    queried_books['CATEGORIES'] = encoder.fit_transform(queried_books['CATEGORIES'])
    queried_books['LANGUAGE'] = encoder.fit_transform(queried_books['LANGUAGE'])
    queried_books['PRINT_TYPE'] = encoder.fit_transform(queried_books['PRINT_TYPE'])
    queried_books['IS_EBOOK'] = encoder.fit_transform(queried_books['IS_EBOOK'])

    #dropping ASIN column of books as it is an identifier
    queried_books.drop(columns=['ASIN','ISBN_13'],inplace=True)

    #splitting the queired_books dataframe for training the ML model into data and label(SALES_COUNT)
    x = queried_books.drop(columns='SALES_COUNT',axis=1)
    y = queried_books['SALES_COUNT']


    #loading the model
    regressor = XGBRegressor()
    regressor.load_model("model_forecast.txt")

    
    #predicting for the queired_books
    print("predicting for the queired_books......")
    results = regressor.predict(x)

    #converting the output to an integer
    list = [int(x) for x in results]

    #Making an final_result dataframe by mapping saless_count and ASIN number
    final_result = pd.DataFrame({'ISBN' : res,'SALES COUNT': list})

    #sorting the dataframe based on higher numer of sales_count of the queired books
    print("sorting the dataframe based on higher numer of sales_count of the queired books......")
    #final_result.index = final_result['ISBN']
    #final_result = final_result.astype('str')
    final_result.set_index('ISBN')

    print("Forecast time --- %s seconds ---" % (time.time() - start_time))
    return (final_result.sort_values(by = 'SALES COUNT',ascending = False).head(10))

#check if the file is older than x days
def is_file_older_than_x_days(file, days=1): 
    file_time = path.getmtime(file) 
    # Check against 24 hours
    return ((time.time() - file_time) / 3600 > 24*days)


import gradio as gd
ui = gd.Interface(forecast, inputs=gd.inputs.Textbox(lines=1, placeholder=None, default="", label="category", optional=False), outputs=gd.outputs.Dataframe(label="forecast"),live=False,num_shap=2.0, theme=None, title="Forecast", description="Predict sales")
ui.launch(share=True)

'''#parameters from API
categorie = "Fiction"
#isbn_13 = "9780399589836"
print(forecast(categorie))'''
