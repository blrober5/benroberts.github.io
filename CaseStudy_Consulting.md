
# Predicting Net Rate of Bike Renting

### Part of a case study to determine the net change in available bikes per hour at stations across the Bay Area


```python
###Import libraries
import os
import numpy as np
import pandas as pd

pd.set_option('display.max_rows', 10000)
pd.set_option('display.max_columns', 100)
```


```python
###Read in Data

#Set Working Directory
os.chdir('/Users/benroberts/Downloads/bikes_data_only/data')
#Read Tables
station=pd.read_csv('station_data.csv')
trip=pd.read_csv('trip_data.csv')
weather=pd.read_csv('weather_data.csv')
```

## Explore Station Data


```python
print(station.shape)
station.head()
```

    (76, 6)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>Name</th>
      <th>Lat</th>
      <th>Long</th>
      <th>Dock Count</th>
      <th>City</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>San Jose Diridon Caltrain Station</td>
      <td>37.329732</td>
      <td>-121.901782</td>
      <td>27</td>
      <td>San Jose</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>San Jose Civic Center</td>
      <td>37.330698</td>
      <td>-121.888979</td>
      <td>15</td>
      <td>San Jose</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>Santa Clara at Almaden</td>
      <td>37.333988</td>
      <td>-121.894902</td>
      <td>11</td>
      <td>San Jose</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>Adobe on Almaden</td>
      <td>37.331415</td>
      <td>-121.893200</td>
      <td>19</td>
      <td>San Jose</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6</td>
      <td>San Pedro Square</td>
      <td>37.336721</td>
      <td>-121.894074</td>
      <td>15</td>
      <td>San Jose</td>
    </tr>
  </tbody>
</table>
</div>




```python
###Checking Missing Values
station.isna().sum()
```




    Id            0
    Name          0
    Lat           0
    Long          0
    Dock Count    0
    City          0
    dtype: int64




```python
###Count Changed Stations as Original Station ID
def change_Station(id_new,id_orig):
    station['Id']=np.where(station['Id']==id_new, id_orig, station['Id'])
change_Station(85,23)
change_Station(86,25)
change_Station(87,49)
change_Station(88,69)
change_Station(89,72)
change_Station(90,72)
#Remove Duplicate Station Rows
station=station.drop_duplicates('Id')
```

## Exploring Trip Data


```python
print(trip.shape)
trip.head()
```

    (354152, 6)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Trip ID</th>
      <th>Start Date</th>
      <th>Start Station</th>
      <th>End Date</th>
      <th>End Station</th>
      <th>Subscriber Type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>913460</td>
      <td>31/08/2015 23:26</td>
      <td>50</td>
      <td>31/08/2015 23:39</td>
      <td>70</td>
      <td>Subscriber</td>
    </tr>
    <tr>
      <th>1</th>
      <td>913459</td>
      <td>31/08/2015 23:11</td>
      <td>31</td>
      <td>31/08/2015 23:28</td>
      <td>27</td>
      <td>Subscriber</td>
    </tr>
    <tr>
      <th>2</th>
      <td>913455</td>
      <td>31/08/2015 23:13</td>
      <td>47</td>
      <td>31/08/2015 23:18</td>
      <td>64</td>
      <td>Subscriber</td>
    </tr>
    <tr>
      <th>3</th>
      <td>913454</td>
      <td>31/08/2015 23:10</td>
      <td>10</td>
      <td>31/08/2015 23:17</td>
      <td>8</td>
      <td>Subscriber</td>
    </tr>
    <tr>
      <th>4</th>
      <td>913453</td>
      <td>31/08/2015 23:09</td>
      <td>51</td>
      <td>31/08/2015 23:22</td>
      <td>60</td>
      <td>Customer</td>
    </tr>
  </tbody>
</table>
</div>




```python
###Checking Missing Values
trip.isna().sum()
```




    Trip ID            0
    Start Date         0
    Start Station      0
    End Date           0
    End Station        0
    Subscriber Type    0
    dtype: int64



## Merging and Aggregating Station and Trip Tables


```python
#Left Merge Trip to Each Station where The Station is Where the Trip Begins
station_trip_start=pd.merge(station, trip, how='left', left_on='Id', right_on='Start Station')

#Left Merge Trip to Each Station where The Station is Where the Trip Ends
station_trip_end=pd.merge(station, trip, how='left', left_on='Id', right_on='End Station')

#Row Bind the Two Merges Together
station_trip_=pd.concat([station_trip_start, station_trip_end])

#Remove Duplicate Rows From Concatenation
station_trip_=station_trip_.drop_duplicates()
```


```python
###Net Trip Change Variable
#-1 if Trip Started at Station and Ended at Another Station
#0 if Trip Started and Ended At Station
#1 if Trip Began at Another Station and Ended at Station
station_trip_['net_trip_change']=np.where(station_trip_['Id']!=station_trip_['End Station'], -1, 
                                         np.where(station_trip_['End Station']==station_trip_['Start Station'],0,1))

```


```python
###Split Date and Time into Separate Columns
station_trip_['start_date1']=pd.to_datetime(station_trip_['Start Date']).dt.date
station_trip_['start_time']=pd.to_datetime(station_trip_['Start Date']).dt.time

###Start Hour Column
station_trip_['start_hour'] = station_trip_['start_time'].astype(str).str[:2]
```


```python
###Subscriber Dummy Variable
station_trip_['Subscriber']=np.where(station_trip_['Subscriber Type']=='Subscriber',1,0)
```


```python
###Aggregating Station Trip Table
#One row for every station at a given day and start hour
station_trip_agg=station_trip_.groupby(['Id', 'start_date1','start_hour'],as_index=False).agg({'Name':'first', 'Dock Count':'first', 'City':'first', 
                                                 'net_trip_change':'sum', 'Start Date':'count', 'Subscriber':'sum'})
```

## Merging Aggregated Station Trip Table with Weather Data


```python
###Changing Zip Code Values to City Names for Merge
def zip_to_city(zip_,city):
    weather.loc[weather.Zip == zip_, 'Zip'] = city
zip_to_city(94107, 'San Francisco')
zip_to_city(94063, 'Redwood City')
zip_to_city(94301, 'Palo Alto')
zip_to_city(94041, 'Mountain View')
zip_to_city(95113, 'San Jose')
```


```python
#Make Date Column a Datetime Value for Merge
weather['Date']=pd.to_datetime(weather['Date']).dt.date
```


```python
###Merge
station_final=pd.merge(station_trip_agg, weather, how='left',left_on=['start_date1', 'City'], right_on=['Date','Zip'])
#Rename Columns
station_final=station_final.rename(columns={'Start Date':'Trips/Hour'})
#Create New Features
station_final['Subscriper_Prop']=station_final['Subscriber']/station_final['Trips/Hour']
station_final['month']=station_final['start_date1'].astype(str).str[5:7]
#Drop Duplicate Columns
station_final=station_final.drop(['Zip','Date','Subscriber','start_date1'],axis=1)
```


```python
###View Data
station_final.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>start_hour</th>
      <th>Name</th>
      <th>Dock Count</th>
      <th>City</th>
      <th>net_trip_change</th>
      <th>Trips/Hour</th>
      <th>Max TemperatureF</th>
      <th>Mean TemperatureF</th>
      <th>Min TemperatureF</th>
      <th>Max Dew PointF</th>
      <th>MeanDew PointF</th>
      <th>Min DewpointF</th>
      <th>Max Humidity</th>
      <th>Mean Humidity</th>
      <th>Min Humidity</th>
      <th>Max Sea Level PressureIn</th>
      <th>Mean Sea Level PressureIn</th>
      <th>Min Sea Level PressureIn</th>
      <th>Max VisibilityMiles</th>
      <th>Mean VisibilityMiles</th>
      <th>Min VisibilityMiles</th>
      <th>Max Wind SpeedMPH</th>
      <th>Mean Wind SpeedMPH</th>
      <th>Max Gust SpeedMPH</th>
      <th>PrecipitationIn</th>
      <th>CloudCover</th>
      <th>Events</th>
      <th>WindDirDegrees</th>
      <th>Subscriper_Prop</th>
      <th>month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>12</td>
      <td>San Jose Diridon Caltrain Station</td>
      <td>27</td>
      <td>San Jose</td>
      <td>-1</td>
      <td>1</td>
      <td>86.0</td>
      <td>72.0</td>
      <td>58.0</td>
      <td>60.0</td>
      <td>54.0</td>
      <td>50.0</td>
      <td>86.0</td>
      <td>59.0</td>
      <td>31.0</td>
      <td>29.86</td>
      <td>29.81</td>
      <td>29.75</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>17.0</td>
      <td>5.0</td>
      <td>22.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>296.0</td>
      <td>1.0</td>
      <td>01</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>14</td>
      <td>San Jose Diridon Caltrain Station</td>
      <td>27</td>
      <td>San Jose</td>
      <td>1</td>
      <td>1</td>
      <td>86.0</td>
      <td>72.0</td>
      <td>58.0</td>
      <td>60.0</td>
      <td>54.0</td>
      <td>50.0</td>
      <td>86.0</td>
      <td>59.0</td>
      <td>31.0</td>
      <td>29.86</td>
      <td>29.81</td>
      <td>29.75</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>17.0</td>
      <td>5.0</td>
      <td>22.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>296.0</td>
      <td>1.0</td>
      <td>01</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>21</td>
      <td>San Jose Diridon Caltrain Station</td>
      <td>27</td>
      <td>San Jose</td>
      <td>-4</td>
      <td>4</td>
      <td>86.0</td>
      <td>72.0</td>
      <td>58.0</td>
      <td>60.0</td>
      <td>54.0</td>
      <td>50.0</td>
      <td>86.0</td>
      <td>59.0</td>
      <td>31.0</td>
      <td>29.86</td>
      <td>29.81</td>
      <td>29.75</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>17.0</td>
      <td>5.0</td>
      <td>22.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>296.0</td>
      <td>0.0</td>
      <td>01</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>22</td>
      <td>San Jose Diridon Caltrain Station</td>
      <td>27</td>
      <td>San Jose</td>
      <td>-1</td>
      <td>1</td>
      <td>86.0</td>
      <td>72.0</td>
      <td>58.0</td>
      <td>60.0</td>
      <td>54.0</td>
      <td>50.0</td>
      <td>86.0</td>
      <td>59.0</td>
      <td>31.0</td>
      <td>29.86</td>
      <td>29.81</td>
      <td>29.75</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>17.0</td>
      <td>5.0</td>
      <td>22.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>296.0</td>
      <td>0.0</td>
      <td>01</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>06</td>
      <td>San Jose Diridon Caltrain Station</td>
      <td>27</td>
      <td>San Jose</td>
      <td>-1</td>
      <td>1</td>
      <td>85.0</td>
      <td>70.0</td>
      <td>55.0</td>
      <td>56.0</td>
      <td>49.0</td>
      <td>36.0</td>
      <td>93.0</td>
      <td>54.0</td>
      <td>15.0</td>
      <td>29.99</td>
      <td>29.90</td>
      <td>29.85</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>16.0</td>
      <td>6.0</td>
      <td>23.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>300.0</td>
      <td>1.0</td>
      <td>01</td>
    </tr>
  </tbody>
</table>
</div>



## Imputation


```python
###Checking Missing Values
print(station_final.shape)
station_final.isna().sum()
```

    (188634, 31)





    Id                                0
    start_hour                        0
    Name                              0
    Dock Count                        0
    City                              0
    net_trip_change                   0
    Trips/Hour                        0
    Max TemperatureF                 77
    Mean TemperatureF                77
    Min TemperatureF                 77
    Max Dew PointF                  427
    MeanDew PointF                  427
    Min DewpointF                   427
    Max Humidity                    427
    Mean Humidity                   427
    Min Humidity                    427
    Max Sea Level PressureIn         13
    Mean Sea Level PressureIn        13
    Min Sea Level PressureIn         13
    Max VisibilityMiles              64
    Mean VisibilityMiles             64
    Min VisibilityMiles              64
    Max Wind SpeedMPH                13
    Mean Wind SpeedMPH               13
    Max Gust SpeedMPH              8201
    PrecipitationIn                  13
    CloudCover                       13
    Events                       146832
    WindDirDegrees                   13
    Subscriper_Prop                   0
    month                             0
    dtype: int64




```python
###Dropping Events Columns - High Missingness
station_final=station_final.drop(['Events'],axis=1)
```


```python
###Imputing Median
station_final=station_final.fillna(station_final.median())
```


```python
###One Hot Encoding Categorical Variables
station_final1=pd.get_dummies(station_final['City'], drop_first=True)
#station_final1.head()
###Add to DataFrame
station_final2=pd.concat([station_final, station_final1], axis=1)
station_final2.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>start_hour</th>
      <th>Name</th>
      <th>Dock Count</th>
      <th>City</th>
      <th>net_trip_change</th>
      <th>Trips/Hour</th>
      <th>Max TemperatureF</th>
      <th>Mean TemperatureF</th>
      <th>Min TemperatureF</th>
      <th>Max Dew PointF</th>
      <th>MeanDew PointF</th>
      <th>Min DewpointF</th>
      <th>Max Humidity</th>
      <th>Mean Humidity</th>
      <th>Min Humidity</th>
      <th>Max Sea Level PressureIn</th>
      <th>Mean Sea Level PressureIn</th>
      <th>Min Sea Level PressureIn</th>
      <th>Max VisibilityMiles</th>
      <th>Mean VisibilityMiles</th>
      <th>Min VisibilityMiles</th>
      <th>Max Wind SpeedMPH</th>
      <th>Mean Wind SpeedMPH</th>
      <th>Max Gust SpeedMPH</th>
      <th>PrecipitationIn</th>
      <th>CloudCover</th>
      <th>WindDirDegrees</th>
      <th>Subscriper_Prop</th>
      <th>month</th>
      <th>Palo Alto</th>
      <th>Redwood City</th>
      <th>San Francisco</th>
      <th>San Jose</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>12</td>
      <td>San Jose Diridon Caltrain Station</td>
      <td>27</td>
      <td>San Jose</td>
      <td>-1</td>
      <td>1</td>
      <td>86.0</td>
      <td>72.0</td>
      <td>58.0</td>
      <td>60.0</td>
      <td>54.0</td>
      <td>50.0</td>
      <td>86.0</td>
      <td>59.0</td>
      <td>31.0</td>
      <td>29.86</td>
      <td>29.81</td>
      <td>29.75</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>17.0</td>
      <td>5.0</td>
      <td>22.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>296.0</td>
      <td>1.0</td>
      <td>01</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>14</td>
      <td>San Jose Diridon Caltrain Station</td>
      <td>27</td>
      <td>San Jose</td>
      <td>1</td>
      <td>1</td>
      <td>86.0</td>
      <td>72.0</td>
      <td>58.0</td>
      <td>60.0</td>
      <td>54.0</td>
      <td>50.0</td>
      <td>86.0</td>
      <td>59.0</td>
      <td>31.0</td>
      <td>29.86</td>
      <td>29.81</td>
      <td>29.75</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>17.0</td>
      <td>5.0</td>
      <td>22.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>296.0</td>
      <td>1.0</td>
      <td>01</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>21</td>
      <td>San Jose Diridon Caltrain Station</td>
      <td>27</td>
      <td>San Jose</td>
      <td>-4</td>
      <td>4</td>
      <td>86.0</td>
      <td>72.0</td>
      <td>58.0</td>
      <td>60.0</td>
      <td>54.0</td>
      <td>50.0</td>
      <td>86.0</td>
      <td>59.0</td>
      <td>31.0</td>
      <td>29.86</td>
      <td>29.81</td>
      <td>29.75</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>17.0</td>
      <td>5.0</td>
      <td>22.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>296.0</td>
      <td>0.0</td>
      <td>01</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>22</td>
      <td>San Jose Diridon Caltrain Station</td>
      <td>27</td>
      <td>San Jose</td>
      <td>-1</td>
      <td>1</td>
      <td>86.0</td>
      <td>72.0</td>
      <td>58.0</td>
      <td>60.0</td>
      <td>54.0</td>
      <td>50.0</td>
      <td>86.0</td>
      <td>59.0</td>
      <td>31.0</td>
      <td>29.86</td>
      <td>29.81</td>
      <td>29.75</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>17.0</td>
      <td>5.0</td>
      <td>22.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>296.0</td>
      <td>0.0</td>
      <td>01</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>06</td>
      <td>San Jose Diridon Caltrain Station</td>
      <td>27</td>
      <td>San Jose</td>
      <td>-1</td>
      <td>1</td>
      <td>85.0</td>
      <td>70.0</td>
      <td>55.0</td>
      <td>56.0</td>
      <td>49.0</td>
      <td>36.0</td>
      <td>93.0</td>
      <td>54.0</td>
      <td>15.0</td>
      <td>29.99</td>
      <td>29.90</td>
      <td>29.85</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>16.0</td>
      <td>6.0</td>
      <td>23.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>300.0</td>
      <td>1.0</td>
      <td>01</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



## Linear Regression Modeling 


```python
###Splitting into Training and Test Sets
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

#Assigning Input Variables
X=station_final2.drop(['net_trip_change', 'Id','Name','City'],axis=1)
X=sm.add_constant(X)
#Assigning Target
y=station_final2['net_trip_change']

#Splitting Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

    /Users/benroberts/anaconda3/lib/python3.7/site-packages/numpy/core/fromnumeric.py:2389: FutureWarning: Method .ptp is deprecated and will be removed in a future version. Use numpy.ptp instead.
      return ptp(axis=axis, out=out, **kwargs)



```python
###Checking Multicollinearity
from statsmodels.stats.outliers_influence import variance_inflation_factor
multi=X_train._get_numeric_data()
multi.head()
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X_train.astype(float).values, i) for i in range(X_train.shape[1])]
vif["features"] = X_train.columns
vif.round(1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>VIF Factor</th>
      <th>features</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>120439.7</td>
      <td>const</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>start_hour</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.1</td>
      <td>Dock Count</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.1</td>
      <td>Trips/Hour</td>
    </tr>
    <tr>
      <th>4</th>
      <td>211.0</td>
      <td>Max TemperatureF</td>
    </tr>
    <tr>
      <th>5</th>
      <td>573.4</td>
      <td>Mean TemperatureF</td>
    </tr>
    <tr>
      <th>6</th>
      <td>170.6</td>
      <td>Min TemperatureF</td>
    </tr>
    <tr>
      <th>7</th>
      <td>18.9</td>
      <td>Max Dew PointF</td>
    </tr>
    <tr>
      <th>8</th>
      <td>60.0</td>
      <td>MeanDew PointF</td>
    </tr>
    <tr>
      <th>9</th>
      <td>25.9</td>
      <td>Min DewpointF</td>
    </tr>
    <tr>
      <th>10</th>
      <td>27.4</td>
      <td>Max Humidity</td>
    </tr>
    <tr>
      <th>11</th>
      <td>151.7</td>
      <td>Mean Humidity</td>
    </tr>
    <tr>
      <th>12</th>
      <td>90.6</td>
      <td>Min Humidity</td>
    </tr>
    <tr>
      <th>13</th>
      <td>72.8</td>
      <td>Max Sea Level PressureIn</td>
    </tr>
    <tr>
      <th>14</th>
      <td>203.4</td>
      <td>Mean Sea Level PressureIn</td>
    </tr>
    <tr>
      <th>15</th>
      <td>55.1</td>
      <td>Min Sea Level PressureIn</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1.7</td>
      <td>Max VisibilityMiles</td>
    </tr>
    <tr>
      <th>17</th>
      <td>3.9</td>
      <td>Mean VisibilityMiles</td>
    </tr>
    <tr>
      <th>18</th>
      <td>3.7</td>
      <td>Min VisibilityMiles</td>
    </tr>
    <tr>
      <th>19</th>
      <td>3.2</td>
      <td>Max Wind SpeedMPH</td>
    </tr>
    <tr>
      <th>20</th>
      <td>4.2</td>
      <td>Mean Wind SpeedMPH</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2.2</td>
      <td>Max Gust SpeedMPH</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1.5</td>
      <td>PrecipitationIn</td>
    </tr>
    <tr>
      <th>23</th>
      <td>2.6</td>
      <td>CloudCover</td>
    </tr>
    <tr>
      <th>24</th>
      <td>1.3</td>
      <td>WindDirDegrees</td>
    </tr>
    <tr>
      <th>25</th>
      <td>1.0</td>
      <td>Subscriper_Prop</td>
    </tr>
    <tr>
      <th>26</th>
      <td>1.1</td>
      <td>month</td>
    </tr>
    <tr>
      <th>27</th>
      <td>1.9</td>
      <td>Palo Alto</td>
    </tr>
    <tr>
      <th>28</th>
      <td>1.9</td>
      <td>Redwood City</td>
    </tr>
    <tr>
      <th>29</th>
      <td>3.7</td>
      <td>San Francisco</td>
    </tr>
    <tr>
      <th>30</th>
      <td>2.9</td>
      <td>San Jose</td>
    </tr>
  </tbody>
</table>
</div>




```python
###Keeping Only Mean Weather Columns due to Multicollinearity
station_final_test=station_final2.drop(['Max TemperatureF','Min TemperatureF','Max Dew PointF','Min DewpointF',
                                       'Max Humidity', 'Min Humidity', 'Max Sea Level PressureIn', 'Min Sea Level PressureIn',
                                       'Max VisibilityMiles', 'Min VisibilityMiles', 'Max Wind SpeedMPH', 'Max Gust SpeedMPH', 'MeanDew PointF'],axis=1)
```


```python
###Final Splitting
X=station_final_test.drop(['net_trip_change', 'Id','Name','City', 'Trips/Hour'],axis=1)
X=sm.add_constant(X)
y=station_final_test['net_trip_change']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```


```python
### Check Multicollinearity Again
multi=X_train._get_numeric_data()
multi.head()
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X_train.astype(float).values, i) for i in range(X_train.shape[1])]
vif["features"] = X_train.columns
vif.round(1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>VIF Factor</th>
      <th>features</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>105741.0</td>
      <td>const</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>start_hour</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>Dock Count</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.9</td>
      <td>Mean TemperatureF</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.9</td>
      <td>Mean Humidity</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.8</td>
      <td>Mean Sea Level PressureIn</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1.5</td>
      <td>Mean VisibilityMiles</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1.6</td>
      <td>Mean Wind SpeedMPH</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1.3</td>
      <td>PrecipitationIn</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1.8</td>
      <td>CloudCover</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1.2</td>
      <td>WindDirDegrees</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1.0</td>
      <td>Subscriper_Prop</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1.1</td>
      <td>month</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1.4</td>
      <td>Palo Alto</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1.2</td>
      <td>Redwood City</td>
    </tr>
    <tr>
      <th>15</th>
      <td>3.3</td>
      <td>San Francisco</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2.7</td>
      <td>San Jose</td>
    </tr>
  </tbody>
</table>
</div>




```python
###Linear Regression
model = sm.OLS(y_train, X_train.astype(float)).fit() ## sm.OLS(output, input)
#predictions = model.predict(X_test)

# Print out the statistics
model.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>     <td>net_trip_change</td> <th>  R-squared:         </th>  <td>   0.001</td>  
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th>  <td>   0.000</td>  
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>  <td>   4.353</td>  
</tr>
<tr>
  <th>Date:</th>             <td>Mon, 09 Mar 2020</td> <th>  Prob (F-statistic):</th>  <td>1.16e-08</td>  
</tr>
<tr>
  <th>Time:</th>                 <td>17:02:22</td>     <th>  Log-Likelihood:    </th> <td>-3.2321e+05</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>132043</td>      <th>  AIC:               </th>  <td>6.465e+05</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td>132026</td>      <th>  BIC:               </th>  <td>6.466e+05</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>    16</td>      <th>                     </th>      <td> </td>     
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>      <td> </td>     
</tr>
</table>
<table class="simpletable">
<tr>
              <td></td>                 <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>                     <td>    2.0981</td> <td>    2.504</td> <td>    0.838</td> <td> 0.402</td> <td>   -2.810</td> <td>    7.006</td>
</tr>
<tr>
  <th>start_hour</th>                <td>   -0.0121</td> <td>    0.002</td> <td>   -7.779</td> <td> 0.000</td> <td>   -0.015</td> <td>   -0.009</td>
</tr>
<tr>
  <th>Dock Count</th>                <td>    0.0041</td> <td>    0.002</td> <td>    2.169</td> <td> 0.030</td> <td>    0.000</td> <td>    0.008</td>
</tr>
<tr>
  <th>Mean TemperatureF</th>         <td>   -0.0010</td> <td>    0.002</td> <td>   -0.569</td> <td> 0.569</td> <td>   -0.004</td> <td>    0.002</td>
</tr>
<tr>
  <th>Mean Humidity</th>             <td>   -0.0006</td> <td>    0.001</td> <td>   -0.543</td> <td> 0.587</td> <td>   -0.003</td> <td>    0.002</td>
</tr>
<tr>
  <th>Mean Sea Level PressureIn</th> <td>   -0.0622</td> <td>    0.081</td> <td>   -0.770</td> <td> 0.441</td> <td>   -0.221</td> <td>    0.096</td>
</tr>
<tr>
  <th>Mean VisibilityMiles</th>      <td>   -0.0081</td> <td>    0.008</td> <td>   -1.054</td> <td> 0.292</td> <td>   -0.023</td> <td>    0.007</td>
</tr>
<tr>
  <th>Mean Wind SpeedMPH</th>        <td>    0.0010</td> <td>    0.003</td> <td>    0.326</td> <td> 0.744</td> <td>   -0.005</td> <td>    0.007</td>
</tr>
<tr>
  <th>PrecipitationIn</th>           <td>    0.0019</td> <td>    0.053</td> <td>    0.035</td> <td> 0.972</td> <td>   -0.102</td> <td>    0.106</td>
</tr>
<tr>
  <th>CloudCover</th>                <td>    0.0002</td> <td>    0.004</td> <td>    0.053</td> <td> 0.958</td> <td>   -0.009</td> <td>    0.009</td>
</tr>
<tr>
  <th>WindDirDegrees</th>            <td> 8.565e-05</td> <td>    0.000</td> <td>    0.749</td> <td> 0.454</td> <td>   -0.000</td> <td>    0.000</td>
</tr>
<tr>
  <th>Subscriper_Prop</th>           <td>   -0.0076</td> <td>    0.026</td> <td>   -0.291</td> <td> 0.771</td> <td>   -0.059</td> <td>    0.043</td>
</tr>
<tr>
  <th>month</th>                     <td>    0.0026</td> <td>    0.002</td> <td>    1.127</td> <td> 0.260</td> <td>   -0.002</td> <td>    0.007</td>
</tr>
<tr>
  <th>Palo Alto</th>                 <td>    0.0075</td> <td>    0.062</td> <td>    0.120</td> <td> 0.905</td> <td>   -0.115</td> <td>    0.130</td>
</tr>
<tr>
  <th>Redwood City</th>              <td>   -0.0553</td> <td>    0.078</td> <td>   -0.707</td> <td> 0.479</td> <td>   -0.209</td> <td>    0.098</td>
</tr>
<tr>
  <th>San Francisco</th>             <td>   -0.0129</td> <td>    0.035</td> <td>   -0.369</td> <td> 0.712</td> <td>   -0.081</td> <td>    0.055</td>
</tr>
<tr>
  <th>San Jose</th>                  <td>   -0.0017</td> <td>    0.040</td> <td>   -0.043</td> <td> 0.966</td> <td>   -0.080</td> <td>    0.077</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>63069.306</td> <th>  Durbin-Watson:     </th>  <td>   1.997</td>  
</tr>
<tr>
  <th>Prob(Omnibus):</th>  <td> 0.000</td>   <th>  Jarque-Bera (JB):  </th> <td>3490382.765</td>
</tr>
<tr>
  <th>Skew:</th>           <td> 1.527</td>   <th>  Prob(JB):          </th>  <td>    0.00</td>  
</tr>
<tr>
  <th>Kurtosis:</th>       <td>28.002</td>   <th>  Cond. No.          </th>  <td>9.25e+04</td>  
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 9.25e+04. This might indicate that there are<br/>strong multicollinearity or other numerical problems.



## Model Performance


```python
###Calculating Root Mean Squared Error on the Test Data
predictions = model.predict(X_test.astype(float))

from sklearn.metrics import mean_squared_error
from math import sqrt

rms = sqrt(mean_squared_error(y_test, predictions))
print(rms)
```

    2.8081904524912047



```python
station_final2['prediction']=predictions
station_results = station_final2[station_final2['prediction'].isna()==False]
station_results.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>start_hour</th>
      <th>Name</th>
      <th>Dock Count</th>
      <th>City</th>
      <th>net_trip_change</th>
      <th>Trips/Hour</th>
      <th>Max TemperatureF</th>
      <th>Mean TemperatureF</th>
      <th>Min TemperatureF</th>
      <th>Max Dew PointF</th>
      <th>MeanDew PointF</th>
      <th>Min DewpointF</th>
      <th>Max Humidity</th>
      <th>Mean Humidity</th>
      <th>Min Humidity</th>
      <th>Max Sea Level PressureIn</th>
      <th>Mean Sea Level PressureIn</th>
      <th>Min Sea Level PressureIn</th>
      <th>Max VisibilityMiles</th>
      <th>Mean VisibilityMiles</th>
      <th>Min VisibilityMiles</th>
      <th>Max Wind SpeedMPH</th>
      <th>Mean Wind SpeedMPH</th>
      <th>Max Gust SpeedMPH</th>
      <th>PrecipitationIn</th>
      <th>CloudCover</th>
      <th>WindDirDegrees</th>
      <th>Subscriper_Prop</th>
      <th>month</th>
      <th>Palo Alto</th>
      <th>Redwood City</th>
      <th>San Francisco</th>
      <th>San Jose</th>
      <th>prediction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>12</td>
      <td>San Jose Diridon Caltrain Station</td>
      <td>27</td>
      <td>San Jose</td>
      <td>-1</td>
      <td>1</td>
      <td>86.0</td>
      <td>72.0</td>
      <td>58.0</td>
      <td>60.0</td>
      <td>54.0</td>
      <td>50.0</td>
      <td>86.0</td>
      <td>59.0</td>
      <td>31.0</td>
      <td>29.86</td>
      <td>29.81</td>
      <td>29.75</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>17.0</td>
      <td>5.0</td>
      <td>22.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>296.0</td>
      <td>1.00</td>
      <td>01</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.045955</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2</td>
      <td>08</td>
      <td>San Jose Diridon Caltrain Station</td>
      <td>27</td>
      <td>San Jose</td>
      <td>-4</td>
      <td>4</td>
      <td>85.0</td>
      <td>70.0</td>
      <td>55.0</td>
      <td>56.0</td>
      <td>49.0</td>
      <td>36.0</td>
      <td>93.0</td>
      <td>54.0</td>
      <td>15.0</td>
      <td>29.99</td>
      <td>29.90</td>
      <td>29.85</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>16.0</td>
      <td>6.0</td>
      <td>23.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>300.0</td>
      <td>1.00</td>
      <td>01</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.094933</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2</td>
      <td>16</td>
      <td>San Jose Diridon Caltrain Station</td>
      <td>27</td>
      <td>San Jose</td>
      <td>6</td>
      <td>6</td>
      <td>85.0</td>
      <td>70.0</td>
      <td>55.0</td>
      <td>56.0</td>
      <td>49.0</td>
      <td>36.0</td>
      <td>93.0</td>
      <td>54.0</td>
      <td>15.0</td>
      <td>29.99</td>
      <td>29.90</td>
      <td>29.85</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>16.0</td>
      <td>6.0</td>
      <td>23.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>300.0</td>
      <td>1.00</td>
      <td>01</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>-0.001544</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2</td>
      <td>17</td>
      <td>San Jose Diridon Caltrain Station</td>
      <td>27</td>
      <td>San Jose</td>
      <td>0</td>
      <td>4</td>
      <td>85.0</td>
      <td>70.0</td>
      <td>55.0</td>
      <td>56.0</td>
      <td>49.0</td>
      <td>36.0</td>
      <td>93.0</td>
      <td>54.0</td>
      <td>15.0</td>
      <td>29.99</td>
      <td>29.90</td>
      <td>29.85</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>16.0</td>
      <td>6.0</td>
      <td>23.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>300.0</td>
      <td>0.75</td>
      <td>01</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>-0.011710</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2</td>
      <td>14</td>
      <td>San Jose Diridon Caltrain Station</td>
      <td>27</td>
      <td>San Jose</td>
      <td>-1</td>
      <td>1</td>
      <td>63.0</td>
      <td>56.0</td>
      <td>48.0</td>
      <td>52.0</td>
      <td>48.0</td>
      <td>44.0</td>
      <td>100.0</td>
      <td>78.0</td>
      <td>55.0</td>
      <td>30.05</td>
      <td>29.90</td>
      <td>29.76</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>7.0</td>
      <td>18.0</td>
      <td>5.0</td>
      <td>22.0</td>
      <td>0.02</td>
      <td>4.0</td>
      <td>145.0</td>
      <td>1.00</td>
      <td>01</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.007690</td>
    </tr>
  </tbody>
</table>
</div>




```python
###Low R-Squared on Training, High Root Mean Squared Error
```

## Potential Improvements

#### To improve model performance and provide better predictions of net rate bike renting, I would try the following:

Using a stepwise variable selection technique with linear regression to tease out the significant variables

Checking the other assumptions of the linear regression (linearity, normality and constant variance of the errors, independence)

Using a regularized regression technique such as Ridge regression to better deal with multicollinearity or LASSO regression for less biased variable selection
         
Engineering more features

Using an Ensembled Tree Based Approach (Random Forest, XGBoost)

## Conclusion
I used a linear regression model to predict the net change in bikes at each station at a given hour. In the model,
I used variables such as proportion of subscribers renting, the hour of the day, the month of the year, the station city, how many bikes the station could hold and various weather factors. Overall, the model predicted net change with a root mean squared error of 2.81 on the validation data. 


```python

```
