# -*- coding: utf-8 -*-
"""Karin_Clean_Data.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1OHqTnIZ6yAhe8i4In0pCH8KTomLv6hdu

# Predict Instruction

Build a model that predicts an accurate delivery time, from picking up a package to arriving at the final destination. An accurate arrival time prediction will help all businesses to improve their logistics and communicate an accurate time to their customers.

## Import packages
"""

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', 60)

# for plotting
import matplotlib.pyplot as plt
# %matplotlib inline
plt.style.use('fivethirtyeight')

import seaborn as sns
sns.set(font_scale=1)

from IPython.core.pylabtools import figsize

# for the Q-Q plots
import scipy.stats as stats

# the dataset for the demo
from sklearn.datasets import load_boston

# Machine Learning Models
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor



# to split and standarize the dataset
from sklearn.preprocessing import StandardScaler

# to evaluate the regression model
from sklearn.metrics import mean_squared_error

# Hyperparameter tuning
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

"""# Import Data"""

train_df = pd.read_csv('https://raw.githubusercontent.com/Maddy-Muir/team_7_regression/master/Data/Train.csv')
test_df = pd.read_csv('https://raw.githubusercontent.com/Maddy-Muir/team_7_regression/master/Data/Test.csv')
riders_df = pd.read_csv('https://raw.githubusercontent.com/Maddy-Muir/team_7_regression/master/Data/Riders.csv')
sample_df = pd.read_csv('https://raw.githubusercontent.com/Maddy-Muir/team_7_regression/master/Data/SampleSubmission.csv')
variable_def = pd.read_csv('https://raw.githubusercontent.com/Maddy-Muir/team_7_regression/master/Data/VariableDefinitions.csv')

"""Display top of tables to get familiar with data"""

riders_df.head()

train_df.head()

test_df.head()

sample_df.head()

"""# See Description of Data Tables"""

train_df.info()

test_df.info()

variable_def

"""# Joining DataFrames

Join riders_df to test_df and train_df
"""

training_df = pd.merge(train_df, riders_df, how = 'inner', on = 'Rider Id')

testing_df = pd.merge(test_df, riders_df, how = 'inner', on = 'Rider Id')

"""#Checking for duplicates"""

training_df[training_df.duplicated(keep=False)]

"""# Cleaning of Data

Check percentage of null values in each column in order to eliminate columns with too many null values.
"""

columns = training_df.columns
percent_missing_values = training_df.isnull().sum()/len(training_df.index)*100
missing_value_df = pd.DataFrame({'column_name': columns,'percent_missing': percent_missing_values})
missing_value_df

"""## Drop columns which are not relevant

Precipitation in millimeters - 97.4% Null values

Temerature at time of order placement - Not relevant

Vehicle type - String for all rows are identical = 'Bike'
"""

training_df = training_df.drop(['Vehicle Type', 'Temperature', 'Precipitation in millimeters' ], axis = 1)

testing_df = testing_df.drop(['Vehicle Type', 'Temperature', 'Precipitation in millimeters' ], axis = 1)

training_df.head()

testing_df.head()

"""#Platform type analysis
Rare labels
"""

# #Plot frequency table of rare labels
# cat_cols = ['Platform Type']
# total_cols = len(training_df)

# for col in cat_cols:

#       temp_df = pd.Series(training_df[col].value_counts() / total_cols)
#       fig = temp_df.sort_values(ascending=False).plot.bar()
#       fig.set_xlabel(col)
#       print(temp_df)

#       fig.set_ylabel('Percentage of platforms')
#       plt.show()

# """#Order analysis"""

# #Weekday of most orders
# cat_cols = ['Placement - Day of Month']

# for col in cat_cols:

#   order_df = pd.Series(training_df[col].value_counts())
#   fig = order_df.sort_index().plot.bar()
#   fig.set_xlabel(col)
#   fig.set_ylabel('Count of Placement Day of Month')
#   plt.show()

# #Weekday of most orders
# cat_cols = ['Placement - Weekday (Mo = 1)']

# for col in cat_cols:

#   order_df = pd.Series(training_df[col].value_counts())
#   fig = order_df.sort_index().plot.bar()
#   fig.set_xlabel(col)
#   fig.set_ylabel('Count of Placement weekday')
#   plt.show()

#Extract time
time_df = training_df.copy()

#Convert to 24 hours
time_df['Placement - Time'] = pd.DatetimeIndex(time_df['Placement - Time']).time
time_df['Confirmation - Time'] = pd.DatetimeIndex(time_df['Confirmation - Time']).time
time_df['Arrival at Pickup - Time'] = pd.DatetimeIndex(time_df['Arrival at Pickup - Time']).time
time_df['Pickup - Time'] = pd.DatetimeIndex(time_df['Pickup - Time']).time

time_df[['Placement_Hour','Placement_Minute','Placement_Seconds']] = time_df['Placement - Time'].astype(str).str.split(':', expand=True).astype(int)
time_df[['Confirmation_Hour','Confirmation_Minute','Confirmation_Seconds']] = time_df['Confirmation - Time'].astype(str).str.split(':', expand=True).astype(int)
time_df[['Arrival_at_Pickup_Hour','Arrival_at_Pickup_Minute','Arrival_at_Pickup_Seconds']] = time_df['Arrival at Pickup - Time'].astype(str).str.split(':', expand=True).astype(int)
time_df[['Pickup_Hour','Pickup_Minute','Pickup_Seconds']] = time_df['Pickup - Time'].astype(str).str.split(':', expand=True).astype(int)

#Hour of day with most orders
time_cols = ['Placement_Hour']

# for col in time_cols:
  
#   time_eda = pd.Series(time_df[col].value_counts())
#   fig = time_eda.sort_index().plot.bar()
#   fig.set_xlabel(col)
#   fig.set_ylabel('Order Count of Placement Hour')
#   plt.show()

# #Hour of day with most pickpus
# time_cols = ['Pickup_Hour']

# for col in time_cols:
  
#   time_eda = pd.Series(time_df[col].value_counts())
#   fig = time_eda.sort_index().plot.bar()
#   fig.set_xlabel(col)
#   fig.set_ylabel('Order Count of Pikcup Hour')
#   plt.show()

"""# Delete additional columns not releveant to model building

Order No, User Id, Platform Type - 
Does not provide any information to assist in the prediction of delivery time from picking up the package to delivery.
"""

training_df = training_df.drop(['User Id', 'Platform Type'], axis = 1)

training_df.head()

testing_df = testing_df.drop(['User Id', 'Platform Type'], axis = 1)

testing_df.head()

"""# Dummy Encoding Function"""

def dummy_encode_columns(input_df, column_name):
    dummy_df = pd.get_dummies(input_df, columns = [column_name], drop_first = True)
    return dummy_df

"""Apply dummy encoding function to 'Personal or Business' column"""

training_df = dummy_encode_columns(training_df, 'Personal or Business')

testing_df = dummy_encode_columns(testing_df, 'Personal or Business')

"""# Comparing 'Day of Month'

Determening if there is a difference between placement, confirmation, Arrival at Pickup, Pickup, Arrival at Destitation. 

If 95% of data as a difference of 0 drop a column.
"""

def diff_check_drop_col(df, col_1, col_2):
  diff_check_drop_col = df[col_1] - df[col_2]
  x = diff_check_drop_col.value_counts()
  if x.loc[0] > len(df.index)*0.95:
    df = df.drop([col_2], axis = 1)
  return df

training_df = diff_check_drop_col(training_df, "Confirmation - Day of Month", "Placement - Day of Month")

training_df = diff_check_drop_col(training_df, "Arrival at Pickup - Day of Month", "Confirmation - Day of Month")

training_df = diff_check_drop_col(training_df,"Pickup - Day of Month" ,"Arrival at Pickup - Day of Month")

training_df = diff_check_drop_col(training_df,'Arrival at Destination - Day of Month',"Pickup - Day of Month")

training_df.head()

training_df = training_df.drop(["Arrival at Destination - Day of Month"], axis = 1)

training_df.head()

"""#Drop all weekday columns - not relevant as all Day of Month columns were dropped

Drop: 

Placement - Weekday (Mo = 1)

Confirmation - Weekday (Mo = 1)	

Arrival at Pickup - Weekday (Mo = 1)	

Pickup - Weekday (Mo = 1)	

Arrival at Destination - Weekday (Mo = 1)
"""

training_df = training_df.drop(['Placement - Weekday (Mo = 1)','Confirmation - Weekday (Mo = 1)','Arrival at Pickup - Weekday (Mo = 1)',
                          'Pickup - Weekday (Mo = 1)','Arrival at Destination - Weekday (Mo = 1)'], axis = 1)

training_df.head()

testing_df = testing_df.drop(['Placement - Weekday (Mo = 1)','Confirmation - Weekday (Mo = 1)','Arrival at Pickup - Weekday (Mo = 1)',
                          'Pickup - Weekday (Mo = 1)'], axis = 1)

testing_df = testing_df.drop(['Confirmation - Day of Month','Placement - Day of Month',"Arrival at Pickup - Day of Month",
                          'Pickup - Day of Month'], axis = 1)

print(testing_df.columns)
#testing_df.head()

"""# Check Correlations & Change Time Format"""

correlations = training_df[training_df.columns].corr()
correlations['Time from Pickup to Arrival'].abs().sort_values()

training_time_cols = ['Placement - Time', 'Confirmation - Time', 'Arrival at Pickup - Time', 
                      'Pickup - Time', 'Arrival at Destination - Time']

for time in training_time_cols:
    training_df[time] = pd.to_datetime(training_df[time])

training_df.head()

training_df['Time Difference - Placement to Confirmation'] = (training_df['Confirmation - Time'] - training_df['Placement - Time']).dt.total_seconds()

training_df.head()

training_df['Time Difference - Confirmation to Arrival at Pickup'] = (training_df['Arrival at Destination - Time'] - training_df['Confirmation - Time']).dt.total_seconds()

training_df.head()

training_df['Time Difference - Arrival at Pickup to Pickup'] = (training_df['Pickup - Time'] - training_df['Arrival at Pickup - Time']).dt.total_seconds()

training_df.head()

training_df['Time Difference - Pickup to Arrival at Destination'] = (training_df['Arrival at Destination - Time'] - training_df['Pickup - Time']).dt.total_seconds()

training_df.head()

testing_time_cols = ['Placement - Time', 'Confirmation - Time', 'Arrival at Pickup - Time', 
                      'Pickup - Time']

for time in testing_time_cols:
    testing_df[time] = pd.to_datetime(testing_df[time])

testing_df['Time Difference - Placement to Confirmation'] = (testing_df['Confirmation - Time'] - testing_df['Placement - Time']).dt.total_seconds()

testing_df['Time Difference - Confirmation to Arrival at Pickup'] = (testing_df['Arrival at Pickup - Time'] - testing_df['Confirmation - Time']).dt.total_seconds()

testing_df['Time Difference - Arrival at Pickup to Pickup'] = (testing_df['Pickup - Time'] - testing_df['Arrival at Pickup - Time']).dt.total_seconds()

testing_df.head()

"""# Drop Time columns

Drop the following columns:

Placement - Time

Confirmation - Time

Arrival at Pickup - Time	

Pickup - Time	

Arrival at Destination - Time
"""

training_df = training_df.drop(['Placement - Time', 'Confirmation - Time', 
                                'Arrival at Pickup - Time', 'Pickup - Time', 
                                'Arrival at Destination - Time'], axis = 1)

training_df.head()

testing_df = testing_df.drop(['Placement - Time', 'Confirmation - Time', 
                                'Arrival at Pickup - Time', 'Pickup - Time'], axis = 1)

testing_df.head()

correlations = training_df[training_df.columns].corr()
correlations['Time from Pickup to Arrival'].abs().sort_values()

training_df = training_df.drop(['Time Difference - Pickup to Arrival at Destination'], axis = 1)

training_df.head()

"""##Harvesine Distance - Converting Longitude/Latitude to a value"""

def haversine(lat1, lon1, lat2, lon2, to_radians = True, earth_radius = 6371):
    """
    Modified version: of http://stackoverflow.com/a/29546836/2901002

    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees or in radians)

    All (lat, lon) coordinates must have numeric dtypes and be of equal length.

    """
    if to_radians:
        lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])

    a = np.sin((lat2-lat1)/2.0)**2 + \
        np.cos(lat1) * np.cos(lat2) * np.sin((lon2-lon1)/2.0)**2

    return earth_radius * 2 * np.arcsin(np.sqrt(a))


training_df['Distance'] = haversine(training_df['Pickup Lat'], 
                                training_df['Pickup Long'],
                                training_df['Destination Lat'], 
                                training_df['Destination Long'])

training_df.head()

testing_df['Distance'] = haversine(testing_df['Pickup Lat'], 
                                testing_df['Pickup Long'],
                                testing_df['Destination Lat'], 
                                testing_df['Destination Long'])

testing_df.head()

"""# Change Rider ID to only be a code"""

def extract_id(input_df):
    input_df['Rider Id'] = input_df['Rider Id'].str.extract(r"([0-9]+)").astype(int)
    return input_df

extract_id(training_df)

extract_id(testing_df)

"""## Target variable distribution"""

figsize(6, 6)

# Histogram of the Energy Star Score
plt.hist(training_df['Time from Pickup to Arrival'].dropna(), bins = 50);
plt.xlabel('Time from Pickup to Arrival'); plt.ylabel('Seconds'); 
plt.title('Time from Pickup to Arrival distribution');

training_df['Time from Pickup to Arrival'].describe()

"""Outlier - A substantial amount of deliveries seems to have been done in just seconds. Comparing min, max and mean"""

training_df['Distance (KM)'].describe()

"""##Target transformation

Create Speed column
"""

training_df['Speed (KM/H)'] = training_df['Distance (KM)']/(training_df['Time from Pickup to Arrival']/3600)
training_df.head()

training_df['Speed (KM/H)'].describe()

"""Check how many orders took less than 60 seconds to deliver"""

training_df[(training_df['Time from Pickup to Arrival'] < 60)].count()

"""Check how many orders were delivered at a speed of more than 80km/h"""

training_df[(training_df['Speed (KM/H)'] > 80)].count()

# """Plot a histogram based on Speed Distribution"""

# figsize(6, 6)
# plt.hist(training_df['Speed (KM/H)'], bins = 30, edgecolor = 'black');
# plt.xlabel('KH/H');
# plt.ylabel('Count'); plt.title('Speed Distribution');

"""Not a normal distribution. Create a new dataframe based only on rows with speed no more than 80km/h"""

speed_df = training_df[(training_df['Speed (KM/H)'] <= 80)]

# figsize(6, 6)
# plt.hist(speed_df['Speed (KM/H)'], bins = 30, edgecolor = 'black');
# plt.xlabel('KH/H');
# plt.ylabel('Count'); plt.title('Speed Distribution');

speed_df['Speed (KM/H)'].describe()

"""Check how many rows are left with a delivery time of less than 60 seconds"""

speed_df[(speed_df['Time from Pickup to Arrival'] < 60)].count()

"""Speed_df represents data with outliers removed"""

speed_df.head()

speed_df.columns

# #Speed interval distribution

# plt.hist(speed_df['Speed (KM/H)'], bins=20)
# plt.xlabel('Speed')
# plt.show()

# #Speed interval distribution

# plt.hist(speed_df['Distance (KM)'], bins=20)
# plt.xlabel('Distance')
# #x= [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
# #plt.xticks(np.arange(min(x), max(x), 5))
# plt.show()

"""#Linear Assumptions"""

#Rearange columns
speed_df = speed_df.filter(['Order No', 'Rider Id',
       'Age','No_Of_Orders', 'Average_Rating', 'No_of_Ratings',
       'Personal or Business_Personal',
       'Time Difference - Placement to Confirmation',
       'Time Difference - Confirmation to Arrival at Pickup',
       'Time Difference - Arrival at Pickup to Pickup',
       'Distance', 'Speed (KM/H)','Distance (KM)', 'Pickup Lat', 'Pickup Long',
       'Destination Lat', 'Destination Long',
       'Time from Pickup to Arrival'], axis=1)

"""<br>- Linear relationship describes a relationship between the independent variables X and the target Y that is given by: Y ≈ β0 + β1X1 + β2X2 + ... + βnXn.
<br>- Normality means that every variable X follows a Gaussian distribution.
<br>- Multi-colinearity refers to the correlation of one independent variable with another. Variables should not be correlated.
<br>- Homoscedasticity, also known as homogeneity of variance, describes a situation in which the error term (that is, the “noise” or random disturbance in the relationship between the independent variables X and the dependent variable Y is the same across all the independent variables.

#Checking Normality
<br>Looking at feature distributions. If not normally distributed, transform them using one of the following methods:
<br>
<br>Logarithmic transformation
<br>Reciprocal transformation
<br>Square root transformation
<br>Exponential transformation
<br>Box-Cox transformation
<br>Yeo-Johnson transformation
"""

transformed_df = speed_df.copy()
#transformed_df.columns = [col.replace("Time Difference","Td") for col in transformed_df.columns]

#Distribution of numerical variables

# transformed_df.hist(figsize=(20,20))
# plt.show()

#Plots to assess normality

def diagnostic_plots(df, variable):
    
    # function to plot a histogram and a Q-Q plot
    # side by side, for a certain variable
    
    plt.figure(figsize=(15,6))
    plt.subplot(1, 2, 1)
    df[variable].hist(bins=30)

    plt.subplot(1, 2, 2)
    stats.probplot(df[variable], dist="norm", plot=plt)

    plt.show()

"""#No_Of_Orders feature

Only displaying all transform methods with No_Of_orders. For the rest, displaying on ly chosen transform method. Boxcox seems to have transformed features best.
"""

#diagnostic_plots(transformed_df, 'No_Of_Orders')

### Logarithmic transformation

transformed_df['No_Of_Orders_log'] = np.log(transformed_df['No_Of_Orders'])

#diagnostic_plots(transformed_df, 'No_Of_Orders_log')

### Reciprocal transformation

transformed_df['No_Of_Orders_reciprocal'] = 1 / (transformed_df['No_Of_Orders']) 

# np.reciprocal(data['GrLivArea'])

#diagnostic_plots(transformed_df, 'No_Of_Orders_reciprocal')

### Square root transformation

transformed_df['No_Of_Orders_sqr'] = transformed_df['No_Of_Orders']**(1/2) 

#diagnostic_plots(transformed_df, 'No_Of_Orders_sqr')

##Exponential transformation

transformed_df['No_Of_Orders_exp'] = transformed_df['No_Of_Orders']**(1/1.5)

#diagnostic_plots(transformed_df, 'No_Of_Orders_exp')

##Box-Cox transformation
transformed_df['No_Of_Orders_boxcox'], param = stats.boxcox(transformed_df['No_Of_Orders']) 

print('Optimal λ: ', param)

#diagnostic_plots(transformed_df, 'No_Of_Orders_boxcox')

##Yeo-Johnson transformation

# to avoid a NumPy error
transformed_df['No_Of_Orders'] = transformed_df['No_Of_Orders'].astype('float')

transformed_df['No_Of_Orders_yeojohnson'], param = stats.yeojohnson(transformed_df['No_Of_Orders']) 

print('Optimal λ: ', param)

#diagnostic_plots(transformed_df, 'No_Of_Orders_yeojohnson')

"""Boxcox seems to fit the No_Of_Orders feature best. Drop other columns"""

transformed_df = transformed_df.drop(['No_Of_Orders', 'No_Of_Orders_log', 'No_Of_Orders_reciprocal', 'No_Of_Orders_sqr', 'No_Of_Orders_exp', 'No_Of_Orders_yeojohnson'],axis=1)

transformed_df.head(3)

"""#Age feature
Boxcox method
"""

#diagnostic_plots(transformed_df, 'Age')

##Box-Cox transformation
transformed_df['Age_boxcox'], param = stats.boxcox(transformed_df['Age']) 

print('Optimal λ: ', param)

#diagnostic_plots(transformed_df, 'Age_boxcox')

"""Boxcox seems to fit the Age feature best. Drop other columns"""

transformed_df = transformed_df.drop(['Age'],axis=1)

transformed_df.head(3)

"""#Distance (KM) Feature - given
Boxcox method
"""

#diagnostic_plots(transformed_df, 'Distance (KM)')

##Box-Cox transformation
transformed_df['Distance (KM)_boxcox'], param = stats.boxcox(transformed_df['Distance (KM)']) 

print('Optimal λ: ', param)

#diagnostic_plots(transformed_df, 'Distance (KM)_boxcox')

transformed_df = transformed_df.drop(['Distance (KM)' ],axis=1)

transformed_df.head(3)

"""#Distance - Harvesine Distance 
Boxcox Method
"""

#diagnostic_plots(transformed_df, 'Distance')

##Box-Cox transformation
transformed_df['Distance_boxcox'], param = stats.boxcox(transformed_df['Distance']) 

print('Optimal λ: ', param)

#diagnostic_plots(transformed_df, 'Distance_boxcox')

transformed_df = transformed_df.drop(['Distance'],axis=1)

transformed_df.head(3)

"""#Speed
Boxcox method
"""

#diagnostic_plots(transformed_df, 'Speed (KM/H)')

##Box-Cox transformation
transformed_df['Speed (KM/H)_boxcox'], param = stats.boxcox(transformed_df['Speed (KM/H)']) 

print('Optimal λ: ', param)

#diagnostic_plots(transformed_df, 'Speed (KM/H)_boxcox')

transformed_df = transformed_df.drop(['Speed (KM/H)'],axis=1)

transformed_df.head(3)

#Renaming columns to original names

transformed_df.columns = [col.replace("_boxcox","") for col in transformed_df.columns]

# New distribution of transformed numerical variables
# transformed_df.hist(figsize=(20,20))
# plt.show()

"""## Feature Selection

Remove features that are redundant
<br>Remove collinear features - features that are highly correlated with each other (60%) - reduce model complexity
<br>Transform numeric values' units
<br>

#Correlations

Filter features
"""

corrs = transformed_df.corr()
corrs = corrs['Time from Pickup to Arrival'].abs().sort_values()
print(corrs)

# fig, ax = plt.subplots(figsize=(13,10))
# sns.heatmap(transformed_df.corr(), annot=True, annot_kws={"fontsize":10}, linewidths=.5, ax=ax)
# plt.show()

"""##Remove collinearity
Remove features that are highly correlated
"""

transformed_df.head()

#Based on above correlation matrix, dropping the following columns with correlation higher that 0.6:
final_features = transformed_df.drop(['Distance (KM)', 'Time Difference - Confirmation to Arrival at Pickup', 'No_of_Ratings', 'No_Of_Orders'], axis=1)

# fig, ax = plt.subplots(figsize=(10,8))
# sns.heatmap(final_features.corr(), annot=True, annot_kws={"fontsize":10}, linewidths=.5, ax=ax)
# plt.show()

"""Update Testing DataFrame accordingly"""

testing_df = testing_df.drop(['Distance (KM)', 'Time Difference - Confirmation to Arrival at Pickup', 'No_of_Ratings', 'No_Of_Orders'], axis=1)

#Rearrange columns
final_features = final_features.filter(['Order No', 'Pickup Lat', 'Pickup Long', 'Destination Lat', 'Destination Long',
       'Rider Id', 'Age', 'Average_Rating','Personal or Business_Personal',
       'Time Difference - Placement to Confirmation',
       'Time Difference - Arrival at Pickup to Pickup', 
       'Distance', 'Speed (KM/H)', 'Time from Pickup to Arrival'], axis=1)

#Drop Speed (KM/H) column not in testing_df
final_features = final_features.drop('Speed (KM/H)', axis=1)

final_features.head()

testing_df.head()



"""## Feature interaction"""





"""##Split into Training and Testing sets
Advised to create validation set - use this for hyperparameter tuning
<br> Advised to use cross validation and not train test split
<br> Cross validation makes model more robust and rely on cross valiadation RMSE score to align with Zindi score
"""

#Drop ID columns for modeling
final_features = final_features.drop(['Order No', 'Rider Id'], axis=1)

from sklearn.model_selection import train_test_split

X = final_features.drop(columns='Time from Pickup to Arrival')
y = pd.DataFrame(final_features['Time from Pickup to Arrival'])


# Split into 70% training and 30% testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

"""##Scale features"""

from sklearn.preprocessing import MinMaxScaler

mms = MinMaxScaler(feature_range=(0, 1))
mms.fit(X_train)
X_train = mms.transform(X_train)
X_test = mms.transform(X_test)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

"""Change scaling for y (Train & Test)"""

y_train = np.array(y_train).reshape((-1, ))
y_test = np.array(y_test).reshape((-1, ))

"""##Base Models"""

# Function to calculate RMSE
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_pred - y_test)**2))

# Takes in a model, trains the model, and evaluates the model on the test set
def fit_and_evaluate(model):
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions and evalute
    model_pred = model.predict(X_test)
    model_rmse = rmse(y_test, model_pred)
    
    # Return the performance metric
    return model_rmse

#Linear Regression
lr = LinearRegression()
lr_rmse = fit_and_evaluate(lr)

print('Linear Regression RMSE = %0.4f' % lr_rmse)

#Lasso regression
lasso = Lasso(alpha=0.01)
lasso_rmse = fit_and_evaluate(lasso)

print('Lasso Regression RMSE = %0.4f' % lasso_rmse)

#Ridge regression
ridge = Ridge()
ridge_rmse = fit_and_evaluate(ridge)

print('Ridge Regression RMSE = %0.4f' % ridge_rmse)

#Decision tree
dec_tree = DecisionTreeRegressor(max_depth=3)
dec_tree_rmse = fit_and_evaluate(dec_tree)

print('Decision Tree RMSE = %0.4f' % dec_tree_rmse)

#Random Forest
random_forest = RandomForestRegressor(random_state=60)
random_forest_rmse = fit_and_evaluate(random_forest)

print('Random Forest Regression RMSE = %0.4f' % random_forest_rmse)

#Gradient boosted
gradient_boosted = GradientBoostingRegressor(random_state=60)
gradient_boosted_rmse = fit_and_evaluate(gradient_boosted)

print('Gradient Boosted Regression RMSE = %0.4f' % gradient_boosted_rmse)

#Pickle model for use within our API
import pickle
save_path = '../assets/trained-models/team7_sendy_simple_lm_regression.pkl'
print (f"Training completed. Saving model to: {save_path}")
pickle.dump(gradient_boosted, open(save_path,'wb'))