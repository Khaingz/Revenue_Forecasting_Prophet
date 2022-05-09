# Revenue_Forecasting_Net_Prophet
Challenge11

## Forecasting Net Prophet

### Description 

In this challenge, You’re a growth analyst at MercadoLibre. With over 200 million users, MercadoLibre is the most popular e-commerce site in Latin America. You've been tasked with analyzing the company's financial and user data in clever ways to make the company grow. So, you want to find out if the ability to predict search traffic can translate into the ability to successfully trade the stock.

### Technologies

This project leverages Google Colab to run a .ipynb notebook. The following packages are also used:

- pandas - Data analysis toolkit for Python.

- hvPlot - A high-level plotting API for the PyData ecosystem built on HoloViews.

- holoviews - Data analysis and visualization tool.

- pystan - Python interface to Stan, a package for Bayesian inference.

- fbprophet - Prophet is a procedure for forecasting time series data.

Install and import the required libraries and dependencies

### Install the required libraries
!pip install pystan
!pip install fbprophet
!pip install hvplot
!pip install holoviews

### Import the required libraries and dependencies
import pandas as pd
import holoviews as hv
from fbprophet import Prophet
import numpy as np
import hvplot.pandas
import datetime as dt
%matplotlib inline

This section divides the instructions for this Challenge into four steps and an optional fifth step, as follows:

Step 1: Find unusual patterns in hourly Google search traffic

#### Upload the "google_hourly_search_trends.csv" file into Colab, then store in a Pandas DataFrame
#### Set the "Date" column as the Datetime Index.

from google.colab import files
uploaded = files.upload()

#### Set the date column as the Datetime Index
df_mercado_trends["Date"]=pd.to_datetime(
    df_mercado_trends["Date"],
    infer_datetime_format=True
)
df_mercado_trends

#### Set the Date column as DataFrame index
df_mercado_trends=df_mercado_trends.set_index("Date")
df_mercado_trends.head()

#### Review the data types of the DataFrame using the info function
df_mercado_trends.info()

#### Holoviews extension to render hvPlots in Colab
hv.extension('bokeh')

#### Slice the DataFrame to just the month of May 2020
df_may_2020 = df_mercado_trends.loc["2020-05"]

#### Use hvPlot to visualize the data for May 2020
df_may_2020.hvplot(title= "Mercado Search Trends 2020-05")

####  Calculate the sum of the total search traffic for May 2020
traffic_may_2020 = df_may_2020['Search Trends'].sum()

#### View the traffic_may_2020 value
The total search traffic value for May 2020 is 38181 searches

#### Calcluate the monhtly median search traffic across all months 
#### Group the DataFrame by index year and then index month, chain the sum and then the median functions
#### Define groupby levels
groupby_levels = [df_mercado_trends.index.year, df_mercado_trends.index.month]

#### Calcluate total monthly value for each month
total_monthly_volume = df_mercado_trends.groupby(by=groupby_levels).sum()

#### Calcluate monthly median search traffic
median_monthly_traffic = total_monthly_volume['Search Trends'].median()

#### View the median_monthly_traffic value
The median monthly traffic value is 35172.5 searches.

Question: Did the Google search traffic increase during the month that MercadoLibre released its financial results?

Answer: Yes, The Google search traffic did slightly increase in the month of May duing Mercadolibre released its financial results.

Step 2: Mine the search traffic data for seasonality

#### Holoviews extension to render hvPlots in Colab
hv.extension('bokeh')

#### Group the hourly search data to plot (use hvPlot) the average traffic by the day of week 
df_mercado_trends.groupby(df_mercado_trends.index.dayofweek).mean().hvplot(
    title='Mercado Search Trends by Day of the Week', 
    xlabel = 'Day of Week')

#### Holoviews extension to render hvPlots in Colab
hv.extension('bokeh')

#### Use hvPlot to visualize the hour of the day and day of week search traffic as a heatmap.
df_mercado_trends.hvplot.heatmap(
    x='index.hour',
    y='index.dayofweek',
    C='Search Trends',
    cmap='reds'
).aggregate(function=np.mean)

Question: Does any day-of-week effect that you observe concentrate in just a few hours of that day?

Answer: Most of the concentration appear in the very beginning of the day or end of the day, and happend to be concentrated from Monday to Friday.

#### Holoviews extension to render hvPlots in Colab
hv.extension('bokeh')

#### Group the hourly search data to plot (use hvPlot) the average traffic by the week of the year
df_mercado_trends.groupby(df_mercado_trends.index.isocalendar().week).mean().hvplot()

Question: Does the search traffic tend to increase during the winter holiday period (weeks 40 through 52)?

Answer: Yes, from weeks 40 through weeks 51 period the search traffic tend to increase during the winter holiday, but we can also see on weeks 52 the search traffic is significantly drop down.

Step 3: Relate the search traffic to stock price patterns

#### Upload the "mercado_stock_price.csv" file into Colab, then store in a Pandas DataFrame
#### Set the "date" column as the Datetime Index.
from google.colab import files
uploaded = files.upload()

df_mercado_stock = pd.read_csv(("mercado_stock_price.csv"),
                               index_col="date",
                               parse_dates=True,
                               infer_datetime_format=True)
                               
#### Holoviews extension to render hvPlots in Colab
hv.extension('bokeh')

#### Use hvPlot to visualize the closing price of the df_mercado_stock DataFrame
df_mercado_stock["close"].hvplot(
    title = "Mercado Stock Closing Prices",
    figsize =[20,10])

#### Concatenate the df_mercado_stock DataFrame with the df_mercado_trends DataFrame
#### Concatenate the DataFrame by columns (axis=1), and drop and rows with only one column of data
mercado_stock_trends_df= pd.concat([df_mercado_trends, df_mercado_stock], axis=1).dropna()

#### For the combined dataframe, slice to just the first half of 2020 (2020-01 through 2020-06) 
first_half_2020 = mercado_stock_trends_df.loc["2020-01-01": "2020-06-30"]

#### Holoviews extension to render hvPlots in Colab
hv.extension('bokeh')

#### Use hvPlot to visualize the close and Search Trends data
#### Plot each column on a separate axes using the following syntax
#### `hvplot(shared_axes=False, subplots=True).cols(1)`
first_half_2020.hvplot(shared_axes=False, subplots=True).cols(1)

Question: Do both time series indicate a common trend that’s consistent with this narrative?

Answer: This narrative seems indicate a common trend. Starting in March, the search trends began to drop as well as the closing price was drop. Towards the end of April, the search results and stock price began to increase. This trend seems to indicate a consistend common trend for the remainder of the half-year.

#### Create a new column in the mercado_stock_trends_df DataFrame called Lagged Search Trends
#### This column should shift the Search Trends information by one hour
mercado_stock_trends_df['Lagged Search Trends'] = mercado_stock_trends_df['Search Trends'].shift(1)

#### Create a new column in the mercado_stock_trends_df DataFrame called Stock Volatility
#### This column should calculate the standard deviation of the closing stock price return data over a 4 period rolling window
mercado_stock_trends_df['Stock Volatility'] = mercado_stock_trends_df['close'].pct_change().rolling(window=4).std()

#### Holoviews extension to render hvPlots in Colab
hv.extension('bokeh')

#### Use hvPlot to visualize the stock volatility
mercado_stock_trends_df['Stock Volatility'].hvplot(title="Mercado Stock Volatility")

#### Create a new column in the mercado_stock_trends_df DataFrame called Hourly Stock Return
#### This column should calculate hourly return percentage of the closing price
mercado_stock_trends_df['Hourly Stock Return'] = mercado_stock_trends_df['close'].pct_change()

#### Construct correlation table of Stock Volatility, Lagged Search Trends, and Hourly Stock Return
mercado_stock_trends_df[['Stock Volatility','Lagged Search Trends','Hourly Stock Return']].corr()

Question: Does a predictable relationship exist between the lagged search traffic and the stock volatility or between the lagged search traffic and the stock price returns?

Answer: Yes, a predicatable relationship does exist between the lagged search traffic and the stock volatility also between the lagged search traffic and the stock price returns does exist a predicatable relationship.

Step 4: Create a time series model with Prophet
#### Using the df_mercado_trends DataFrame, reset the index so the date information is no longer the index
mercado_prophet_df = df_mercado_trends.reset_index()

#### Label the columns ds and y so that the syntax is recognized by Prophet
mercado_prophet_df = mercado_prophet_df.rename(columns={'Date':'ds', 'Search Trends': 'y'})

#### Drop an NaN values from the prophet_df DataFrame
mercado_prophet_df = mercado_prophet_df.dropna()

#### Call the Prophet function, store as an object
model_mercado_trends = Prophet()

#### Fit the time-series model.
model_mercado_trends.fit(mercado_prophet_df)

#### Create a future dataframe to hold predictions
#### Make the prediction go out as far as 2000 hours (approx 80 days)
future_mercado_trends = model_mercado_trends.make_future_dataframe(periods=2000, freq='H')

#### Make the predictions for the trend data using the future_mercado_trends DataFrame
forecast_mercado_trends = model_mercado_trends.predict(future_mercado_trends)

#### Plot the Prophet predictions for the Mercado trends data
model_mercado_trends.plot(forecast_mercado_trends)

Question: How's the near-term forecast for the popularity of MercadoLibre?

Answer: The popularity of Mercadolibre is forecasted to be dorp down in this near-term foreacast.

#### Set the index in the forecast_mercado_trends DataFrame to the ds datetime column
forecast_mercado_trends = forecast_mercado_trends.set_index('ds')

#### View the only the yhat,yhat_lower and yhat_upper columns from the DataFrame
display(forecast_mercado_trends[['yhat','yhat_lower','yhat_upper']].head())
display(forecast_mercado_trends[['yhat','yhat_lower','yhat_upper']].tail())

#### Holoviews extension to render hvPlots in Colab
hv.extension('bokeh')

#### From the forecast_mercado_trends DataFrame, use hvPlot to visualize
####  the yhat, yhat_lower, and yhat_upper columns over the last 2000 hours 
forecast_mercado_trends[['yhat','yhat_lower','yhat_upper']].iloc[-2000:,:].plot()

#### Reset the index in the forecast_mercado_trends DataFrame
forecast_mercado_trends = forecast_mercado_trends.reset_index()

#### Use the plot_components function to visualize the forecast results 
#### for the forecast_canada DataFrame 
figures_mercado_trends = model_mercado_trends.plot_components(forecast_mercado_trends)

Question: What time of day exhibits the greatest popularity?

Answer: In the mindnight (00:00:00) appears the greatest popularity.

Question: Which day of week gets the most search traffic?

Answer: Tuesday gets the most search traffic.

Question: What's the lowest point for search traffic in the calendar year?

Answer: # The lowest point for search traffic in the calendar year is in October.

Step 5 (optional): Forecast revenue by using time series models

A few weeks after your initial analysis, the finance group follows up to find out if you can help them solve a different problem. Your fame as a growth analyst in the company continues to grow!

Specifically, the finance group wants a forecast of the total sales for the next quarter. This will dramatically increase their ability to plan budgets and to help guide expectations for the company investors.

To do so, complete the following steps:

1. Read in the daily historical sales (that is, revenue) figures, and then apply a Prophet model to the data. The daily sales figures are quoted in millions of USD dollars.
#### Upload the "mercado_daily_revenue.csv" file into Colab, then store in a Pandas DataFrame
#### Set the "date" column as the DatetimeIndex
#### Sales are quoted in millions of US dollars
from google.colab import files
uploaded = files.upload()

df_mercado_sales = pd.read_csv(
    'mercado_daily_revenue.csv',
    index_col ='date',
    infer_datetime_format= True,
    parse_dates=True)
    
#### Holoviews extension to render hvPlots in Colab
hv.extension('bokeh')

#### Use hvPlot to visualize the daily sales figures 
df_mercado_sales.hvplot(title="Mercado Daily Sales")    
    
#### Apply a Facebook Prophet model to the data.Set up the dataframe in the neccessary format:
#### Reset the index so that date becomes a column in the DataFrame
mercado_sales_prophet_df = df_mercado_sales.reset_index()

#### Adjust the columns names to the Prophet syntax
mercado_sales_prophet_df.columns = ['ds','y']

#### Drop NaN values from the prophet_df DataFrame
mercado_sales_prophet_df = mercado_sales_prophet_df.dropna()

#### Create the model
mercado_sales_prophet_model = Prophet(yearly_seasonality=True)

#### Fit the model
mercado_sales_prophet_model.fit(mercado_sales_prophet_df)

#### Predict sales for 90 days (1 quarter) out into the future.
#### Start by making a future dataframe
mercado_sales_prophet_future = mercado_sales_prophet_model.make_future_dataframe(periods=90 , freq='D')

#### Make predictions for the sales each day over the next quarter
mercado_sales_prophet_forecast = mercado_sales_prophet_model.predict(mercado_sales_prophet_future)

2. Interpret the model output to identify any seasonal patterns in the company's revenue. For example, what are the peak revenue days? (Mondays? Fridays? Something else?)
#### Use the plot_components function to analyze seasonal patterns in the company's revenue
figures = mercado_sales_prophet_model.plot_components(mercado_sales_prophet_forecast)

Question: For example, what are the peak revenue days? (Mondays? Fridays? Something else?)

Answer: # The peak revenue days occurs on Wednesdays.

3. Produce a sales forecast for the finance group. Give them a number for the expected total sales in the next quarter. Include the best- and worst-case scenarios to help them make better plans.
#### Plot the predictions for the Mercado sales
mercado_sales_prophet_model.plot(mercado_sales_prophet_forecast)

#### For the mercado_sales_prophet_forecast DataFrame, set the ds column as the DataFrame Index
mercado_sales_prophet_forecast = mercado_sales_prophet_forecast.set_index('ds')

#### Produce a sales forecast for the finance division giving them a number for expected total sales next quarter.
#### Provide best case (yhat_upper), worst case (yhat_lower), and most likely (yhat) scenarios.
#### Create a forecast_quarter Dataframe for the period 2020-05-14 to 2020-08-12 (Previously 2020-07-01 to 2020-09-30)*
####   *I updated the dates for this section to include 90 full days of the forecasted data.  
####   *The original data ended at 2020-05-14 so is only forecasted out to 2020-08-12 (90 days as indicated above).
#### The DataFrame should include the columns yhat_upper, yhat_lower, and yhat
mercado_sales_forecast_quarter = mercado_sales_prophet_forecast[['yhat', 'yhat_lower', 'yhat_upper']].loc['2020-05-14':'2020-08-12']

#### Update the column names for the forecast_quarter DataFrame
#### to match what the finance division is looking for 
mercado_sales_forecast_quarter.columns = ['Most Likely', 'Worst Case Scenario', 'Best Case Scenario']

#### Displayed the summed values for all the rows in the forecast_quarter DataFrame
mercado_sales_forecast_quarter.sum()

Question : Based on the forecast information generated above, produce a sales forecast for the finance division, giving them a number for expected total sales next quarter. Include best and worst case scenarios, to better help the finance team plan.

Answer: Based on the forecast infromation genearated, the expected total sales for the next quarter(90 days) are defined into 3 categroies those are: Most Likely gaining total revenue, worst case scenario total revenue and best case scenario total revenue.

Most likely total Revenue = $ 1,955.88 million

Worst case scenario Revenue = $ 1,789.56 million

Best case scenario Revenue = $ 2,122.20 million

For the worst case scenario, I would provide business strategies to Sale and Marketing team to put more efforts and focus on product campaign, use effective marketing strategies like (offer discounts and rebates to repeated customers, adding complimentary services or products etc) to invigorate the sales revenues.

For the best case scenario, Once the sales went up, I would provide recommendation to Production, Manufacturing, warehouse and shipping team to focus on materials inverntory, cost control, quality control and budget controlling. The production line need to make sure company operations are run smoothly to achieve the target revenue.
