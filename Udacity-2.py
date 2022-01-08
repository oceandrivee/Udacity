#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
import statistics
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns
from collections import defaultdict
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
import math
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'


# In[4]:


df_list = pd.read_csv('listings.csv')


# After importing necesessary libraries and the dataset, we'll start to the project with examining the dataset as first step.

# In[5]:


print(df_list.shape)


# In[6]:


df_list.dtypes.value_counts()


# In[7]:


##The following code examines the name of the columns which are float64 data type
df_list.dtypes[df_list.dtypes=='float64']


# In[8]:


#The following code examines the name of the columns which are int64 data type
df_list.dtypes[df_list.dtypes=='int64']


# In[9]:


#The following code examines the name of the columns which are object data type
pd.set_option('display.max_rows', 100)
df_list.dtypes[df_list.dtypes=='object']


# In[10]:


#The following code examines the number of missing values in each column in descending order
df_list.isnull().sum().sort_values(ascending = False)


# In[11]:


#The following code examines the columns names the columns that are without any missing values and all values are unique
df_list.nunique()[df_list.nunique()==len(df_list)]


# In[12]:


#The following code examines the columns names with unique values as well as missing values with descending order
df_list.nunique()[df_list.nunique() != len(df_list)].sort_values(ascending=False)


# In[13]:


df_list['amenities'][0]


# Question 1
# 
# One of the most vast detailed column within the dataset is 'amenities' column. Thus, we will focus on this column more than other columns at first. We will start to answer the question: 
# What are the most common amenities within Boston Airbnb dataset? 
# 
# Let's start!

# In[14]:


#Firstly, create an empty list to fill with amenities available.
list_amenities = []

#Then, create a 'for' loop to append all amenity values from the dataset into the list.
for i in range(len(df_list)):
    list_amenities.append(df_list['amenities'][i])

    
#Due to the fact that each list within the dataset has a list of amenities, we need to retrieve unique amenities from the dataset
#Therefore, we will use the "set" function to set the amenities unique and then convert the set into a list
#to remain consistency of data type.
list_amenities = list((s.strip('\'\{\}') for s in list_amenities))
list_amenities_string = ",".join(list_amenities)
list_amenities = list(set(list_amenities_string.split(",")))

#Now we need to remove each empty string within the list of amenities 
without_empty_amenities = []
for string in list_amenities:
    if (string != ""):
        without_empty_amenities.append(string)
list_amenities = without_empty_amenities


# In[15]:


#The code snippet below is taken from one of the Udacity lecture notes. 
#This function is used to count the number of listings that contains each amenity in the amenities list.
def total_count(df, col1, col2, look_for):
    '''
    INPUT:
    df - the pandas dataframe you want to search
    col1 - the column name you want to look through
    col2 - the column you want to count values from
    look_for - a list of strings you want to search for in each row of df[col]

    OUTPUT:
    new_df - a dataframe of each look_for with the count of how often it shows up
    '''
    new_df = defaultdict(int)
    #loop through list of amenities
    for val in look_for:
        #loop through rows
        for idx in range(df.shape[0]):
            #if the amenity is in the row add 1
            if val in df[col1][idx]:
                new_df[val] += int(df[col2][idx])
    new_df = pd.DataFrame(pd.Series(new_df)).reset_index()
    new_df.columns = [col1, col2]
    new_df.sort_values('count', ascending=False, inplace=True)
    return new_df


# In[17]:


#We will create a new dataframe which is called named 'amenities_common', which has lists of amenities as one column and the count of these amenities lists
#in our main (df_list)
amenities_common = df_list['amenities'].value_counts().reset_index()
amenities_common.rename(columns={'index': 'amenities', 'amenities': 'count'}, inplace=True)

#Counting the number of listings that contains each amenity in the amenities list via total_count function.
amenities_common_df = total_count(amenities_common, 'amenities', 'count', list_amenities)
amenities_common_df.set_index('amenities', inplace=True)

#Visualizing the percentage of each amenity with ascending sort as a bar chart 
(amenities_common_df/len(df_list)).plot(kind='bar', figsize=(12,8), legend=None);
plt.title('Top 20 Most Common Amenities in The Dataset');
plt.ylabel('Percentage of listings that contains the amenity')

plt.show()


# In[18]:


amenities_common_df.head()


# Question 2
# 
# In this part, we will try to find answers for this questions:
# What attracts people most to rent a property? Price? Neighborhood? or Offered amenities?
# 

# In[19]:


#We will focus on 90 days period for availability and booking ratio
df_list['availability_90']


# In[20]:


#Checking missing values within the 'availability_90' column
df_list['availability_90'].isnull().sum()


# In[21]:


#Yet it doesn't have any missing values, we can define our booking ratio column for 90 days
df_list['booking_ratio_90'] = 1 - (df_list['availability_90']/90)


# In[22]:


df_list['booking_ratio_90']


# In[23]:


#Now, we can examine booking ratio for next 90 days for different pricing and neighbourhoods
df_list.groupby(['price','neighbourhood_cleansed'])['booking_ratio_90'].mean().sort_values(ascending=False)


# In[24]:


#Visualizing the booking ratio difference for spectrum of prices for next 90 days as a bar chart
(df_list.groupby(['price'])['booking_ratio_90'].mean().sort_values(ascending=False))[:50].plot(kind='bar',
    figsize=(18,8))
plt.title('Booking Ratio for next 90 days based on pricing');
plt.show()


# Conclusion: There is no significant correlation between pricing and booking ratio

# In[25]:


#Checking missing values withing the 'neighbourhood_cleansed' column 
 
df_list['neighbourhood_cleansed'].isnull().sum()


# In[26]:


#The number of each unique values in the 'neighbourhood_cleansed' column.
df_list['neighbourhood_cleansed'].value_counts().sort_values(ascending=False)


# In[27]:


#Listing the booking ratios of different neighbourhoods for next 90 days
df_list.groupby(['neighbourhood_cleansed'])['booking_ratio_90'].mean().sort_values(ascending=False)


# In[28]:


#Visualizing the booking ratio difference for spectrum of neighbourhoods for next 90 days as a bar chart
(df_list.groupby(['neighbourhood_cleansed'])['booking_ratio_90'].mean().sort_values(ascending=False)).plot(kind='bar',
    figsize=(12,8))
plt.title('Booking Ratio for next 90 dyas on neighbourhood');
plt.show()


# Conclusion: We can observe that some neighbourhood has higher booking ration and consecutively more demand than others.

# In[29]:


#Below code shows the unique review scores rating of rentals and count of each one
df_list['review_scores_rating'].value_counts().sort_values(ascending=False)


# In[32]:


#Checking missing values for the 'amenities' column
df_list['amenities'].isnull().sum()


# In[34]:


#Creating new columns for each of the amenities and fill them all with zeros
for i in range(len(list_amenities)):
    df_list[list_amenities[i]] = 0


# In[35]:


#Now filling the new amenities columns 
#Firstly, checking if the listing has that amenity.
#If the listing has that amenity, I fill it with '1'; and if not, fill it with '0'.
for i in range(len(list_amenities)):
    for t in range(len(df_list)):
        if list_amenities[i] in df_list['amenities'][t]:
            df_list.loc[t , list_amenities[i]] = 1
        else:
            df_list.loc[t , list_amenities[i]] = 0


# In[36]:


#Now, creating a dataframe that will contain information about booking ratio of listings that contain that specific amenity.

a_impact = pd.DataFrame(index=range(0,len(list_amenities)), columns = ["Amenity Name", "Booking ratio with amenity", "Booking ratio without amenity", "Booking Difference"])


#In the below for loop, I write the name of the amenity, calculate the booking ratio of listings that contain that specific amenitiy
#and calculate the booking ratio of listings that do not contain that specific amenity
#and calculate the difference between booking ratios for that specific amenity. 
for i in range(len(list_amenities)):
    a_impact['Amenity Name'][i] = list_amenities[i]
    a_impact['Booking ratio with amenity'][i] = df_list.groupby([list_amenities[i]])['booking_ratio_90'].mean()[1]
    a_impact['Booking ratio without amenity'][i] = df_list.groupby([list_amenities[i]])['booking_ratio_90'].mean()[0]
    a_impact['Booking Difference'][i] = a_impact['Booking ratio with amenity'][i] - a_impact['Booking ratio without amenity'][i]

a_impact.set_index('Amenity Name', inplace=True)


# In[37]:


#Listing the most 15 booking difference to identify which amenities are mostly prefered by guests.
a_impact.sort_values(by='Booking Difference', ascending = False)[:15]


# In[38]:


#Visualizing most preferred 15 amenities by booking difference.
a_impact['Booking Difference'].sort_values(ascending = False)[:15].plot(kind='bar', figsize=(12,8), legend=None)
plt.title('Most in-Demand Amenities');
plt.ylabel('Booking ratio difference for next 90 days')
plt.show()


# In[39]:


#Visualizing the least preferred 15 amenities by booking difference.
a_impact['Booking Difference'].sort_values(ascending = True)[:15].plot(kind='bar', figsize=(12,8), legend=None)
plt.title('Least in-Demand Amenities');
plt.ylabel('Booking ratio difference for upcoming 90 days')
plt.show()


# Question 3
# 
# As a last part of this project, we will train a machine learning model in order to forecast pricing of the listings. The last question to answer is:
# 
# What are the most effectual traits to forecast pricing of a listing? 

# 
# Firstly, we should identify the number of missing values in the features that are relavant to price.

# In[41]:


print(df_list['price'].isnull().sum())
print(df_list['weekly_price'].isnull().sum())
print(df_list['monthly_price'].isnull().sum())
print(df_list['security_deposit'].isnull().sum())
print(df_list['cleaning_fee'].isnull().sum())


# In[42]:


#Listing of features that are expected to be essential for forecasting the pricing of listings
#Creating a new dataframe for listings

#Here is the columns for amenities categorical value

selected_features = ['bathrooms', 'bedrooms', 'beds', 'latitude', 'longitude', 'reviews_per_month',
        'booking_ratio_90', 'accommodates', 'guests_included', '"24-Hour Check-in"', '"Suitable for Events"',
       '"Pets live on this property"', '"Smoking Allowed"',
       '"Other pet(s)"', 'Essentials', '"Wireless Internet"',
       '"Buzzer/Wireless Intercom"', 'TV', 'Gym', 'Washer', 'Doorman', 'Dryer',
       '"Air Conditioning"', '"Pets Allowed"', 'Dog(s)',
       '"Hair Dryer"', '"Fire Extinguisher"', 'Breakfast', '"Washer / Dryer"',
       '"Laptop Friendly Workspace"', '"Free Parking on Premises"',
       '"Lock on Bedroom Door"', 'Hangers', '"Family/Kid Friendly"',
       '"Carbon Monoxide Detector"', '"Safety Card"', 'Kitchen',
       '"Elevator in Building"', 'Internet', 'Shampoo', '"Smoke Detector"',
       '"Paid Parking Off Premises"', '"First Aid Kit"',
       '"Indoor Fireplace"', '"Cable TV"', 'Heating', 'neighbourhood_cleansed',
        'property_type','room_type','bed_type','price','security_deposit',
        'cleaning_fee', 'extra_people', 'instant_bookable', 'cancellation_policy']

df_list_ml = df_list[selected_features]


# In[43]:


#Checking missing values for each columns

df_list_ml.isnull().sum()[df_list_ml.isnull().sum()>0]


# In[44]:


#Replacing missing values of these features with the mean values of each feature in the dataset.

df_list_ml['bathrooms'].fillna(df_list_ml['bathrooms'].mean(), inplace=True)
df_list_ml['bedrooms'].fillna(df_list_ml['bedrooms'].mean(), inplace=True)
df_list_ml['beds'].fillna(df_list_ml['beds'].mean(), inplace=True)
df_list_ml['reviews_per_month'].fillna(df_list_ml['reviews_per_month'].mean(), inplace=True)


# In[45]:


#Replacing missing values of property_type with mode value of the feature in the dataset.

df_list_ml['property_type'].fillna(df_list_ml['property_type'].mode()[0], inplace=True)


# In[46]:


#Replacing missing values of these features with zeros.


df_list_ml['security_deposit'].fillna(0, inplace=True)
df_list_ml['cleaning_fee'].fillna(0, inplace=True)


# In[47]:


df_list_ml.dtypes


# In[48]:


#Removing $ signs and comma signs 

df_list_ml['price'] = df_list_ml['price'].str.replace('$', '')
df_list_ml['security_deposit'] = df_list_ml['security_deposit'].str.replace('$', '')
df_list_ml['cleaning_fee'] = df_list_ml['cleaning_fee'].str.replace('$', '')
df_list_ml['extra_people'] = df_list_ml['extra_people'].str.replace('$', '')

df_list_ml['price'] = df_list_ml['price'].str.replace(',', '')
df_list_ml['security_deposit'] = df_list_ml['security_deposit'].str.replace(',', '')
df_list_ml['cleaning_fee'] = df_list_ml['cleaning_fee'].str.replace(',', '')
df_list_ml['extra_people'] = df_list_ml['extra_people'].str.replace(',', '')

#Filling NaN values with 0 
df_list_ml['security_deposit'].fillna(0, inplace=True)
df_list_ml['cleaning_fee'].fillna(0, inplace=True)


# In[49]:


#Altering data types to float from string
df_list_ml['price'] = df_list_ml['price'].astype(float)
df_list_ml['security_deposit'] = df_list_ml['security_deposit'].astype(float)
df_list_ml['cleaning_fee'] = df_list_ml['cleaning_fee'].astype(float)
df_list_ml['extra_people'] = df_list_ml['extra_people'].astype(float)


# In[50]:


#Establishing a new dataframe as a subset of categorical columns from df_list_ml dataframe
cat_cols = df_list_ml.select_dtypes(include=['object'])


# In[51]:


cat_cols.head()


# In[52]:


#Creating new columns from categorical variables to use these features in ML regression algorithm.
for col in cat_cols:
    try:
        df_list_ml = pd.concat([df_list_ml.drop(col, axis=1), pd.get_dummies(df_list_ml[col], prefix=col, prefix_sep='_', drop_first=True)], axis=1)
    except:
        continue


# In[53]:


#Establishing the X (features) and y (the variable to be modelled) dataframes
y = df_list_ml['price']
X = df_list_ml.drop(columns='price')


# In[54]:


#Splitting the new dataframes into train and test dataframes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)


# In[58]:


#Applying the linear regression, fitting the model, making predictions with the test set and scoring the success of the model
lm_model = LinearRegression(normalize=True)
lm_model.fit(X_train, y_train)
y_test_preds = lm_model.predict(X_test)
y_train_preds = lm_model.predict(X_train)

test_score = r2_score(y_test, y_test_preds)
train_score = r2_score(y_train, y_train_preds)

print(test_score)
print(train_score)




# In[59]:


#Below function is taken from one of the Udacity jupyter notebook examples. I used this code to see coefficients of my model. 
def coef_weights(coefficients, X_train):
    '''
    INPUT:
    coefficients - the coefficients of the linear model 
    X_train - the training data, so the column names can be used
    OUTPUT:
    coefs_df - a dataframe holding the coefficient, estimate, and abs(estimate)
    
    Provides a dataframe that can be used to understand the most influential coefficients
    in a linear model by providing the coefficient estimates along with the name of the 
    variable attached to the coefficient.
    '''
    coefs_df = pd.DataFrame()
    coefs_df['est_int'] = X_train.columns
    coefs_df['coefs'] = lm_model.coef_
    coefs_df['abs_coefs'] = np.abs(lm_model.coef_)
    coefs_df = coefs_df.sort_values('abs_coefs', ascending=False)
    return coefs_df

#Use the function
coef_df = coef_weights(lm_model.coef_, X_train)


# In[60]:


coef_df.head(20)


# In[61]:


def find_optimal_number_of_selectors(X, y, k_samples, plot=True, legend=True):
    '''
    INPUT:
    X dataframe that contains the features.
    y dataframe that contains the variable to be predicted.
    k_samples is a list of k values that will be tested.
    
    OUTPUT:
    A dictionary that contains values of tested k-values as keys and r-squared values for each key.
    The k-value that has the highest r-squared.

    '''   
    result_r_squareds = []
    results = {}
    for kes in k_samples:
        selector = SelectKBest(score_func=f_regression, k=kes)
        selector.fit_transform(X, y)
        selected_cols = selector.get_support(indices=True)
        features_new_X = X.iloc[:,selected_cols]
        
        X_train, X_test, y_train, y_test = train_test_split(features_new_X, y, test_size = 0.3, random_state=42)
        lm_model = LinearRegression(normalize=True)
        lm_model.fit(X_train, y_train)
        y_test_preds = lm_model.predict(X_test)
        result_r_squareds.append(r2_score(y_test, y_test_preds))
        results[str(kes)] = r2_score(y_test, y_test_preds)
    
    if plot:
        plt.plot(k_samples, result_r_squareds, label="r-squared", alpha=0.7)
        plt.xlabel("Different k_values")
        plt.ylabel("R_Squared_Values")
        plt.legend(loc=1)
        plt.show()
    
    best_k = max(results, key=results.get)
    
    return results, best_k


# In[62]:


#Testing a range of k values from 10 to 94 (which is the number of all features) to find the number of features generate the highest r-squared value.
k_samples = range(10, len(X.columns))
k_results, best_k = find_optimal_number_of_selectors(X, y, k_samples)
print(k_results)
print(best_k)
print(k_results[best_k])


# In[ ]:




