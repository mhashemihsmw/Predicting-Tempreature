# %% [markdown]
# # Predicting Tempratur: Running Machine Learning Experiments in Python

# %% [markdown]
# As the climate changes, predicting the weather becomes ever more important for businesses. Since the weather depends on a lot of different factors, we want to run a lot of experiments to determine what the best approach is to predict the weather.
# 
# - In this project, we will use London Weather data sourced from [Kaggle](https://www.kaggle.com/datasets/emmanuelfwerr/london-weather-data) to try and predict the temperature.
# - The focus of this code-along is running machine learning experiments. We will first do some minor exploratory data analysis, and then use `MLflow` to run experiments on what models and what hyperparameters to use.
# - This is interesting for those of you that have already trained a machine learning model before and want to see how you can speed up the process of finding the best model.

# %% [markdown]
# ### Import libraries
# First, we'll import necessary libraries, including **MLflow**.
# 
# **MLflow** is an open-source platform designed to help manage the end-to-end machine learning lifecycle. It provides a comprehensive set of tools and features to streamline the process of building, training, and deploying machine learning models. Today, we'll be using MLflow for tracking experiments, hyperparameter tuning, model performance evaluation, and comparison and analysis of multiple models.
# 
# To use MLflow, we first need to install the package, since it's not included in the workspace by default. Using the `!`, we can run a bash command to install it.

# %% [markdown]
# After the installation, we can import all the libraries, including MLflow. 
# 
# - `pandas` to import and read, and edit the data
# - `numpy` is used for calculations. 
# - `MLflow` is the library we'll use for structuring our machine learning experiments
# - `seaborn` is used for visualizations
# - `sklearn`, scikit-learn is used for machine learning, with functions such as data preprocessing, model training and prediction

# %%
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# %% [markdown]
# ### Load data
# We will be working with data stored in `london_weather.csv`, which contains the following columns:
# - **date** - recorded date of measurement - (**int**)
# - **cloud_cover** - cloud cover measurement in oktas - (**float**)
# - **sunshine** - sunshine measurement in hours (hrs) - (**float**)
# - **global_radiation** - irradiance measurement in Watt per square meter (W/m2) - (**float**)
# - **max_temp** - maximum temperature recorded in degrees Celsius (°C) - (**float**)
# - **mean_temp** - mean temperature in degrees Celsius (°C) - (**float**)
# - **min_temp** - minimum temperature recorded in degrees Celsius (°C) - (**float**)
# - **precipitation** - precipitation measurement in millimeters (mm) - (**float**)
# - **pressure** - pressure measurement in Pascals (Pa) - (**float**)
# - **snow_depth** - snow depth measurement in centimeters (cm) - (**float**)
# 
# We'll load the dataset using the pandas `read_csv` function.

# %%
df = pd.read_csv('london_weather.csv')
df.head(5)

# %%
df.info()

# %% [markdown]
# ### Exploratory Data Analysis
# Now that we have loaded the dataset, let's perform some exploratory data analysis to understand the data better. This includes handling missing values, feature engineering, and visualizing the data.
# 
# - Use pandas `pd.to_datetime` function to adjust the type of the date column
# - Also add the year and month 
# - Calculate the number of missing values using pandas `isna()`
# - Select the relevant weather columns
# - Groupby year and month and calculate the mean of the relevant metrics
# - Lineplot the mean temperature per month using `seaborn`
# - Barplot the mean sunshine per month
# - Visualize a heatmap to show the correlation of features using seaborn's `heatmap()` function and pandas `.corr()` function.

# %%
# Converting 'date' column to datetime format
df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month

# Check missing values
df.isna().sum()

# %%
# Grouping data by year and month, calculating mean of weather metrics
weather_metrics = ['month', 'cloud_cover', 'sunshine', 'global_radiation', 'max_temp', 'mean_temp', 'min_temp', 'precipitation', 'pressure', 'snow_depth']
df_per_month = df.groupby(['year', 'month'], as_index = False)[weather_metrics].mean()

# Visualizing data
sns.lineplot(x="year", y="mean_temp", data=df_per_month, ci=None)

# %%
sns.barplot(x='month', y='sunshine', data=df)

# %%
sns.heatmap(df[weather_metrics].corr(), annot=True)

# %% [markdown]
# ### Process data into train and test sets
# Next, we have a function to preprocess the dataframe into train and test samples. 
# - In this function we impute and scale the features. Imputing is done to fill in missing values, in this case using the mean, and scaling is used to put all features on the same scale, which often improves the performance.
# - Important to note is that we split the train and test set before we do imputation and scaling, such that there is no data leakage between train and test set.
# - Before running the function, we will also drop rows in which the temperature variable is unknown. Since we need to be able to train and test on the target variable at all times.

# %%
def preprocess_df(df, feature_selection, target_var):
    df = df.dropna(subset=[target_var])
    X = df[feature_selection]    
    y = df[target_var]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
    
    imputer = SimpleImputer(strategy="mean")
    X_train = imputer.fit_transform(X_train)
    X_test  = imputer.transform(X_test)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

feature_selection = ['month', 'cloud_cover', 'sunshine', 'global_radiation', 'snow_depth']
target_var = 'mean_temp'
X_train, X_test, y_train, y_test = preprocess_df(df, feature_selection, target_var)

# %% [markdown]
# ### Running a machine learning experiment
# Now we will move on to the bulk of this code-along, namely, running machine learning experiments. A simple example of training a machine learning model would go as follows:
# 
# 
# `lreg = LinearRegression().fit(X_train, y_train)`
# 
# `y_pred = lreg.predict(x_test)`
# 
# `rmse = np.sqrt(mean_squared_error(y_test, y_pred))`
# 
# 
# We train a linear regression model, predict the target variable and calculate the score of the model. Repeating this for different settings, and possibly another model would result in a lot of duplicate code. You could write functions to repeatedly go over different models and hyperparameters, but you can also use MLflow. MLflow allows you to run different experiments, log the experiments, and also save the model with the best result. Since we are limited to the workspaces, we focus on tracking and logging experiments in this project.

# %%
mlflow.set_experiment('example_experiment_1')

with mlflow.start_run():
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate RMSE
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    # Log hyperparameters and metrics
    mlflow.set_experiment_tag("model", "decision tree")
    mlflow.log_metric("rmse", rmse)

    # Save the model
    mlflow.sklearn.log_model(model, "model")

# %% [markdown]
# ### Predicting the temperature in London
# We will now do this for the london weather dataset. We will use two different regression models, decision tree regression, and random forest regression. We will start with just using the depth as a parameter to experiment with.
# 
# - Start by declaring an `EXPERIMENT_NAME` and `EXPERIMENT_ID`, and use the `create_experiment()` function of MLflow to create an experiment.
# - Then create a `for`-loop to go over different `depth` values
# - Declare a parameters dictionary that takes the depth value from the loop. Include a `random_state` to make sure we can re-run the experiment and get similar results.
# - Use MLflow's `start_run()`, include the `experiment_id` and `run_name` in the function call.
# - Fit the `DecisionTreeRegressor()` and `RandomForestRegressor()` to the training data.
# - Get the RMSE of each model and log the parameters and metrics using MLflow's `log_param()` and `log_metric()` functions. 
# - Save the experiment results using MLflow `search_runs()` function.

# %%
# Define experiment name
EXPERIMENT_NAME = "experiment_1"
EXPERIMENT_ID = mlflow.create_experiment(EXPERIMENT_NAME)

# Setup for loop
for idx, depth in enumerate([1, 2, 5, 10, 20]):
    parameters = {
        'max_depth': depth,
        'random_state': 1
    }
    
    RUN_NAME = f"run_{idx}"
    
    with mlflow.start_run(experiment_id=EXPERIMENT_ID, run_name=RUN_NAME):
        dreg = DecisionTreeRegressor(**parameters).fit(X_train, y_train)
        rfreg = RandomForestRegressor(**parameters).fit(X_train, y_train)
        
        dreg_pred = dreg.predict(X_test)
        rfreg_pred = rfreg.predict(X_test)

        dreg_r2 = r2_score(y_test, dreg_pred)
        rfreg_r2 = r2_score(y_test, rfreg_pred)
        
        mlflow.log_param("max_depth", depth)
                         
        mlflow.log_metric("dreg_r2", dreg_r2)
        mlflow.log_metric("rfreg_r2", rfreg_r2)

# Save experiment results
experiment_result = mlflow.search_runs(experiment_names=[EXPERIMENT_NAME])

# %%
experiment_result


