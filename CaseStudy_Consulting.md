
# Predicting Net Rate of Bike Renting


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
#Missing Values
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
#Missing Values
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
station_final1=pd.get_dummies(station_final['Name'])
station_final1.head()
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
      <th>2nd at Folsom</th>
      <th>2nd at South Park</th>
      <th>2nd at Townsend</th>
      <th>5th at Howard</th>
      <th>Adobe on Almaden</th>
      <th>Arena Green / SAP Center</th>
      <th>Beale at Market</th>
      <th>Broadway St at Battery St</th>
      <th>Broadway at Main</th>
      <th>California Ave Caltrain Station</th>
      <th>Castro Street and El Camino Real</th>
      <th>Civic Center BART (7th at Market)</th>
      <th>Clay at Battery</th>
      <th>Commercial at Montgomery</th>
      <th>Cowper at University</th>
      <th>Davis at Jackson</th>
      <th>Embarcadero at Bryant</th>
      <th>Embarcadero at Folsom</th>
      <th>Embarcadero at Sansome</th>
      <th>Embarcadero at Vallejo</th>
      <th>Evelyn Park and Ride</th>
      <th>Franklin at Maple</th>
      <th>Golden Gate at Polk</th>
      <th>Grant Avenue at Columbus Avenue</th>
      <th>Harry Bridges Plaza (Ferry Building)</th>
      <th>Howard at 2nd</th>
      <th>Japantown</th>
      <th>MLK Library</th>
      <th>Market at 10th</th>
      <th>Market at 4th</th>
      <th>Market at Sansome</th>
      <th>Mechanics Plaza (Market at Battery)</th>
      <th>Mezes Park</th>
      <th>Mountain View Caltrain Station</th>
      <th>Mountain View City Hall</th>
      <th>Palo Alto Caltrain Station</th>
      <th>Park at Olive</th>
      <th>Paseo de San Antonio</th>
      <th>Post at Kearney</th>
      <th>Powell Street BART</th>
      <th>Powell at Post (Union Square)</th>
      <th>Redwood City Caltrain Station</th>
      <th>Redwood City Medical Center</th>
      <th>Redwood City Public Library</th>
      <th>Rengstorff Avenue / California Street</th>
      <th>Ryland Park</th>
      <th>SJSU - San Salvador at 9th</th>
      <th>SJSU 4th at San Carlos</th>
      <th>San Antonio Caltrain Station</th>
      <th>San Antonio Shopping Center</th>
      <th>San Francisco Caltrain (Townsend at 4th)</th>
      <th>San Francisco Caltrain 2 (330 Townsend)</th>
      <th>San Francisco City Hall</th>
      <th>San Jose City Hall</th>
      <th>San Jose Civic Center</th>
      <th>San Jose Diridon Caltrain Station</th>
      <th>San Mateo County Center</th>
      <th>San Pedro Square</th>
      <th>San Salvador at 1st</th>
      <th>Santa Clara County Civic Center</th>
      <th>Santa Clara at Almaden</th>
      <th>South Van Ness at Market</th>
      <th>Spear at Folsom</th>
      <th>St James Park</th>
      <th>Steuart at Market</th>
      <th>Temporary Transbay Terminal (Howard at Beale)</th>
      <th>Townsend at 7th</th>
      <th>University and Emerson</th>
      <th>Washington at Kearney</th>
      <th>Yerba Buena Center of the Arts (3rd @ Howard)</th>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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

    /Users/benroberts/anaconda3/lib/python3.7/site-packages/statsmodels/regression/linear_model.py:1543: RuntimeWarning: divide by zero encountered in double_scalars
      return 1 - self.ssr/self.centered_tss
    /Users/benroberts/anaconda3/lib/python3.7/site-packages/statsmodels/stats/outliers_influence.py:181: RuntimeWarning: divide by zero encountered in double_scalars
      vif = 1. / (1. - r_squared_i)





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
      <td>0.0</td>
      <td>const</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>start_hour</td>
    </tr>
    <tr>
      <th>2</th>
      <td>inf</td>
      <td>Dock Count</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.3</td>
      <td>Trips/Hour</td>
    </tr>
    <tr>
      <th>4</th>
      <td>211.1</td>
      <td>Max TemperatureF</td>
    </tr>
    <tr>
      <th>5</th>
      <td>573.7</td>
      <td>Mean TemperatureF</td>
    </tr>
    <tr>
      <th>6</th>
      <td>170.7</td>
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
      <td>27.5</td>
      <td>Max Humidity</td>
    </tr>
    <tr>
      <th>11</th>
      <td>152.4</td>
      <td>Mean Humidity</td>
    </tr>
    <tr>
      <th>12</th>
      <td>91.0</td>
      <td>Min Humidity</td>
    </tr>
    <tr>
      <th>13</th>
      <td>72.8</td>
      <td>Max Sea Level PressureIn</td>
    </tr>
    <tr>
      <th>14</th>
      <td>203.6</td>
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
      <td>4.3</td>
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
      <td>2.7</td>
      <td>CloudCover</td>
    </tr>
    <tr>
      <th>24</th>
      <td>1.3</td>
      <td>WindDirDegrees</td>
    </tr>
    <tr>
      <th>25</th>
      <td>1.1</td>
      <td>Subscriper_Prop</td>
    </tr>
    <tr>
      <th>26</th>
      <td>1.1</td>
      <td>month</td>
    </tr>
    <tr>
      <th>27</th>
      <td>inf</td>
      <td>2nd at Folsom</td>
    </tr>
    <tr>
      <th>28</th>
      <td>inf</td>
      <td>2nd at South Park</td>
    </tr>
    <tr>
      <th>29</th>
      <td>inf</td>
      <td>2nd at Townsend</td>
    </tr>
    <tr>
      <th>30</th>
      <td>inf</td>
      <td>5th at Howard</td>
    </tr>
    <tr>
      <th>31</th>
      <td>inf</td>
      <td>Adobe on Almaden</td>
    </tr>
    <tr>
      <th>32</th>
      <td>inf</td>
      <td>Arena Green / SAP Center</td>
    </tr>
    <tr>
      <th>33</th>
      <td>inf</td>
      <td>Beale at Market</td>
    </tr>
    <tr>
      <th>34</th>
      <td>inf</td>
      <td>Broadway St at Battery St</td>
    </tr>
    <tr>
      <th>35</th>
      <td>inf</td>
      <td>Broadway at Main</td>
    </tr>
    <tr>
      <th>36</th>
      <td>inf</td>
      <td>California Ave Caltrain Station</td>
    </tr>
    <tr>
      <th>37</th>
      <td>inf</td>
      <td>Castro Street and El Camino Real</td>
    </tr>
    <tr>
      <th>38</th>
      <td>inf</td>
      <td>Civic Center BART (7th at Market)</td>
    </tr>
    <tr>
      <th>39</th>
      <td>inf</td>
      <td>Clay at Battery</td>
    </tr>
    <tr>
      <th>40</th>
      <td>inf</td>
      <td>Commercial at Montgomery</td>
    </tr>
    <tr>
      <th>41</th>
      <td>inf</td>
      <td>Cowper at University</td>
    </tr>
    <tr>
      <th>42</th>
      <td>inf</td>
      <td>Davis at Jackson</td>
    </tr>
    <tr>
      <th>43</th>
      <td>inf</td>
      <td>Embarcadero at Bryant</td>
    </tr>
    <tr>
      <th>44</th>
      <td>inf</td>
      <td>Embarcadero at Folsom</td>
    </tr>
    <tr>
      <th>45</th>
      <td>inf</td>
      <td>Embarcadero at Sansome</td>
    </tr>
    <tr>
      <th>46</th>
      <td>inf</td>
      <td>Embarcadero at Vallejo</td>
    </tr>
    <tr>
      <th>47</th>
      <td>inf</td>
      <td>Evelyn Park and Ride</td>
    </tr>
    <tr>
      <th>48</th>
      <td>inf</td>
      <td>Franklin at Maple</td>
    </tr>
    <tr>
      <th>49</th>
      <td>inf</td>
      <td>Golden Gate at Polk</td>
    </tr>
    <tr>
      <th>50</th>
      <td>inf</td>
      <td>Grant Avenue at Columbus Avenue</td>
    </tr>
    <tr>
      <th>51</th>
      <td>inf</td>
      <td>Harry Bridges Plaza (Ferry Building)</td>
    </tr>
    <tr>
      <th>52</th>
      <td>inf</td>
      <td>Howard at 2nd</td>
    </tr>
    <tr>
      <th>53</th>
      <td>inf</td>
      <td>Japantown</td>
    </tr>
    <tr>
      <th>54</th>
      <td>inf</td>
      <td>MLK Library</td>
    </tr>
    <tr>
      <th>55</th>
      <td>inf</td>
      <td>Market at 10th</td>
    </tr>
    <tr>
      <th>56</th>
      <td>inf</td>
      <td>Market at 4th</td>
    </tr>
    <tr>
      <th>57</th>
      <td>inf</td>
      <td>Market at Sansome</td>
    </tr>
    <tr>
      <th>58</th>
      <td>inf</td>
      <td>Mechanics Plaza (Market at Battery)</td>
    </tr>
    <tr>
      <th>59</th>
      <td>inf</td>
      <td>Mezes Park</td>
    </tr>
    <tr>
      <th>60</th>
      <td>inf</td>
      <td>Mountain View Caltrain Station</td>
    </tr>
    <tr>
      <th>61</th>
      <td>inf</td>
      <td>Mountain View City Hall</td>
    </tr>
    <tr>
      <th>62</th>
      <td>inf</td>
      <td>Palo Alto Caltrain Station</td>
    </tr>
    <tr>
      <th>63</th>
      <td>inf</td>
      <td>Park at Olive</td>
    </tr>
    <tr>
      <th>64</th>
      <td>inf</td>
      <td>Paseo de San Antonio</td>
    </tr>
    <tr>
      <th>65</th>
      <td>inf</td>
      <td>Post at Kearney</td>
    </tr>
    <tr>
      <th>66</th>
      <td>inf</td>
      <td>Powell Street BART</td>
    </tr>
    <tr>
      <th>67</th>
      <td>inf</td>
      <td>Powell at Post (Union Square)</td>
    </tr>
    <tr>
      <th>68</th>
      <td>inf</td>
      <td>Redwood City Caltrain Station</td>
    </tr>
    <tr>
      <th>69</th>
      <td>inf</td>
      <td>Redwood City Medical Center</td>
    </tr>
    <tr>
      <th>70</th>
      <td>inf</td>
      <td>Redwood City Public Library</td>
    </tr>
    <tr>
      <th>71</th>
      <td>inf</td>
      <td>Rengstorff Avenue / California Street</td>
    </tr>
    <tr>
      <th>72</th>
      <td>inf</td>
      <td>Ryland Park</td>
    </tr>
    <tr>
      <th>73</th>
      <td>inf</td>
      <td>SJSU - San Salvador at 9th</td>
    </tr>
    <tr>
      <th>74</th>
      <td>inf</td>
      <td>SJSU 4th at San Carlos</td>
    </tr>
    <tr>
      <th>75</th>
      <td>inf</td>
      <td>San Antonio Caltrain Station</td>
    </tr>
    <tr>
      <th>76</th>
      <td>inf</td>
      <td>San Antonio Shopping Center</td>
    </tr>
    <tr>
      <th>77</th>
      <td>inf</td>
      <td>San Francisco Caltrain (Townsend at 4th)</td>
    </tr>
    <tr>
      <th>78</th>
      <td>inf</td>
      <td>San Francisco Caltrain 2 (330 Townsend)</td>
    </tr>
    <tr>
      <th>79</th>
      <td>inf</td>
      <td>San Francisco City Hall</td>
    </tr>
    <tr>
      <th>80</th>
      <td>inf</td>
      <td>San Jose City Hall</td>
    </tr>
    <tr>
      <th>81</th>
      <td>inf</td>
      <td>San Jose Civic Center</td>
    </tr>
    <tr>
      <th>82</th>
      <td>inf</td>
      <td>San Jose Diridon Caltrain Station</td>
    </tr>
    <tr>
      <th>83</th>
      <td>inf</td>
      <td>San Mateo County Center</td>
    </tr>
    <tr>
      <th>84</th>
      <td>inf</td>
      <td>San Pedro Square</td>
    </tr>
    <tr>
      <th>85</th>
      <td>inf</td>
      <td>San Salvador at 1st</td>
    </tr>
    <tr>
      <th>86</th>
      <td>inf</td>
      <td>Santa Clara County Civic Center</td>
    </tr>
    <tr>
      <th>87</th>
      <td>inf</td>
      <td>Santa Clara at Almaden</td>
    </tr>
    <tr>
      <th>88</th>
      <td>inf</td>
      <td>South Van Ness at Market</td>
    </tr>
    <tr>
      <th>89</th>
      <td>inf</td>
      <td>Spear at Folsom</td>
    </tr>
    <tr>
      <th>90</th>
      <td>inf</td>
      <td>St James Park</td>
    </tr>
    <tr>
      <th>91</th>
      <td>inf</td>
      <td>Steuart at Market</td>
    </tr>
    <tr>
      <th>92</th>
      <td>inf</td>
      <td>Temporary Transbay Terminal (Howard at Beale)</td>
    </tr>
    <tr>
      <th>93</th>
      <td>inf</td>
      <td>Townsend at 7th</td>
    </tr>
    <tr>
      <th>94</th>
      <td>inf</td>
      <td>University and Emerson</td>
    </tr>
    <tr>
      <th>95</th>
      <td>inf</td>
      <td>Washington at Kearney</td>
    </tr>
    <tr>
      <th>96</th>
      <td>inf</td>
      <td>Yerba Buena Center of the Arts (3rd @ Howard)</td>
    </tr>
  </tbody>
</table>
</div>




```python
###Keeping Only Mean Weather Columns due to Multicollinearity
station_final_test=station_final2.drop(['Max TemperatureF','Min TemperatureF','Max Dew PointF','Min DewpointF',
                                       'Max Humidity', 'Min Humidity', 'Max Sea Level PressureIn', 'Min Sea Level PressureIn',
                                       'Max VisibilityMiles', 'Min VisibilityMiles', 'Max Wind SpeedMPH', 'Max Gust SpeedMPH'],axis=1)
```


```python

```


```python
###Final Splitting
X=station_final_test.drop(['net_trip_change', 'Id','Name','City', 'Trips/Hour'],axis=1)
X=sm.add_constant(X)
y=station_final_test['net_trip_change']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```


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
  <th>Dep. Variable:</th>     <td>net_trip_change</td> <th>  R-squared:         </th>  <td>   0.019</td>  
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th>  <td>   0.019</td>  
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>  <td>   31.96</td>  
</tr>
<tr>
  <th>Date:</th>             <td>Sat, 01 Feb 2020</td> <th>  Prob (F-statistic):</th>   <td>  0.00</td>   
</tr>
<tr>
  <th>Time:</th>                 <td>15:17:23</td>     <th>  Log-Likelihood:    </th> <td>-3.2197e+05</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>132043</td>      <th>  AIC:               </th>  <td>6.441e+05</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td>131961</td>      <th>  BIC:               </th>  <td>6.449e+05</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>    81</td>      <th>                     </th>      <td> </td>     
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>      <td> </td>     
</tr>
</table>
<table class="simpletable">
<tr>
                        <td></td>                           <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>                                         <td>    2.0645</td> <td>    1.930</td> <td>    1.070</td> <td> 0.285</td> <td>   -1.718</td> <td>    5.847</td>
</tr>
<tr>
  <th>start_hour</th>                                    <td>   -0.0135</td> <td>    0.002</td> <td>   -8.778</td> <td> 0.000</td> <td>   -0.017</td> <td>   -0.010</td>
</tr>
<tr>
  <th>Dock Count</th>                                    <td>    0.0311</td> <td>    0.031</td> <td>    1.009</td> <td> 0.313</td> <td>   -0.029</td> <td>    0.091</td>
</tr>
<tr>
  <th>Mean TemperatureF</th>                             <td>   -0.0038</td> <td>    0.004</td> <td>   -0.859</td> <td> 0.390</td> <td>   -0.012</td> <td>    0.005</td>
</tr>
<tr>
  <th>MeanDew PointF</th>                                <td>    0.0031</td> <td>    0.005</td> <td>    0.681</td> <td> 0.496</td> <td>   -0.006</td> <td>    0.012</td>
</tr>
<tr>
  <th>Mean Humidity</th>                                 <td>   -0.0021</td> <td>    0.002</td> <td>   -0.896</td> <td> 0.370</td> <td>   -0.007</td> <td>    0.003</td>
</tr>
<tr>
  <th>Mean Sea Level PressureIn</th>                     <td>   -0.0730</td> <td>    0.080</td> <td>   -0.910</td> <td> 0.363</td> <td>   -0.230</td> <td>    0.084</td>
</tr>
<tr>
  <th>Mean VisibilityMiles</th>                          <td>   -0.0101</td> <td>    0.008</td> <td>   -1.295</td> <td> 0.195</td> <td>   -0.025</td> <td>    0.005</td>
</tr>
<tr>
  <th>Mean Wind SpeedMPH</th>                            <td>    0.0009</td> <td>    0.003</td> <td>    0.304</td> <td> 0.761</td> <td>   -0.005</td> <td>    0.007</td>
</tr>
<tr>
  <th>PrecipitationIn</th>                               <td>   -0.0097</td> <td>    0.053</td> <td>   -0.184</td> <td> 0.854</td> <td>   -0.113</td> <td>    0.093</td>
</tr>
<tr>
  <th>CloudCover</th>                                    <td>    0.0002</td> <td>    0.004</td> <td>    0.034</td> <td> 0.973</td> <td>   -0.009</td> <td>    0.009</td>
</tr>
<tr>
  <th>WindDirDegrees</th>                                <td> 7.318e-05</td> <td>    0.000</td> <td>    0.641</td> <td> 0.521</td> <td>   -0.000</td> <td>    0.000</td>
</tr>
<tr>
  <th>Subscriper_Prop</th>                               <td>   -0.0020</td> <td>    0.026</td> <td>   -0.075</td> <td> 0.941</td> <td>   -0.054</td> <td>    0.050</td>
</tr>
<tr>
  <th>month</th>                                         <td>    0.0024</td> <td>    0.002</td> <td>    1.048</td> <td> 0.295</td> <td>   -0.002</td> <td>    0.007</td>
</tr>
<tr>
  <th>2nd at Folsom</th>                                 <td>   -0.8130</td> <td>    0.054</td> <td>  -15.067</td> <td> 0.000</td> <td>   -0.919</td> <td>   -0.707</td>
</tr>
<tr>
  <th>2nd at South Park</th>                             <td>   -0.1503</td> <td>    0.122</td> <td>   -1.236</td> <td> 0.217</td> <td>   -0.389</td> <td>    0.088</td>
</tr>
<tr>
  <th>2nd at Townsend</th>                               <td>   -0.0101</td> <td>    0.263</td> <td>   -0.039</td> <td> 0.969</td> <td>   -0.525</td> <td>    0.505</td>
</tr>
<tr>
  <th>5th at Howard</th>                                 <td>    0.2280</td> <td>    0.121</td> <td>    1.886</td> <td> 0.059</td> <td>   -0.009</td> <td>    0.465</td>
</tr>
<tr>
  <th>Adobe on Almaden</th>                              <td>   -0.1247</td> <td>    0.114</td> <td>   -1.089</td> <td> 0.276</td> <td>   -0.349</td> <td>    0.100</td>
</tr>
<tr>
  <th>Arena Green / SAP Center</th>                      <td>   -0.0401</td> <td>    0.112</td> <td>   -0.359</td> <td> 0.720</td> <td>   -0.259</td> <td>    0.179</td>
</tr>
<tr>
  <th>Beale at Market</th>                               <td>   -0.4645</td> <td>    0.053</td> <td>   -8.706</td> <td> 0.000</td> <td>   -0.569</td> <td>   -0.360</td>
</tr>
<tr>
  <th>Broadway St at Battery St</th>                     <td>   -0.0139</td> <td>    0.122</td> <td>   -0.114</td> <td> 0.909</td> <td>   -0.252</td> <td>    0.225</td>
</tr>
<tr>
  <th>Broadway at Main</th>                              <td>    0.4694</td> <td>    0.535</td> <td>    0.877</td> <td> 0.380</td> <td>   -0.580</td> <td>    1.518</td>
</tr>
<tr>
  <th>California Ave Caltrain Station</th>               <td>    0.2565</td> <td>    0.173</td> <td>    1.485</td> <td> 0.137</td> <td>   -0.082</td> <td>    0.595</td>
</tr>
<tr>
  <th>Castro Street and El Camino Real</th>              <td>    0.1558</td> <td>    0.247</td> <td>    0.630</td> <td> 0.529</td> <td>   -0.329</td> <td>    0.641</td>
</tr>
<tr>
  <th>Civic Center BART (7th at Market)</th>             <td>   -0.0497</td> <td>    0.155</td> <td>   -0.320</td> <td> 0.749</td> <td>   -0.354</td> <td>    0.255</td>
</tr>
<tr>
  <th>Clay at Battery</th>                               <td>    0.1459</td> <td>    0.123</td> <td>    1.181</td> <td> 0.237</td> <td>   -0.096</td> <td>    0.388</td>
</tr>
<tr>
  <th>Commercial at Montgomery</th>                      <td>    0.0841</td> <td>    0.122</td> <td>    0.689</td> <td> 0.491</td> <td>   -0.155</td> <td>    0.323</td>
</tr>
<tr>
  <th>Cowper at University</th>                          <td>    0.2561</td> <td>    0.258</td> <td>    0.991</td> <td> 0.322</td> <td>   -0.250</td> <td>    0.763</td>
</tr>
<tr>
  <th>Davis at Jackson</th>                              <td>    0.1508</td> <td>    0.123</td> <td>    1.225</td> <td> 0.220</td> <td>   -0.090</td> <td>    0.392</td>
</tr>
<tr>
  <th>Embarcadero at Bryant</th>                         <td>   -0.0198</td> <td>    0.121</td> <td>   -0.164</td> <td> 0.870</td> <td>   -0.256</td> <td>    0.217</td>
</tr>
<tr>
  <th>Embarcadero at Folsom</th>                         <td>   -0.0861</td> <td>    0.055</td> <td>   -1.554</td> <td> 0.120</td> <td>   -0.195</td> <td>    0.022</td>
</tr>
<tr>
  <th>Embarcadero at Sansome</th>                        <td>    0.2969</td> <td>    0.119</td> <td>    2.486</td> <td> 0.013</td> <td>    0.063</td> <td>    0.531</td>
</tr>
<tr>
  <th>Embarcadero at Vallejo</th>                        <td>    0.4748</td> <td>    0.124</td> <td>    3.844</td> <td> 0.000</td> <td>    0.233</td> <td>    0.717</td>
</tr>
<tr>
  <th>Evelyn Park and Ride</th>                          <td>   -0.1191</td> <td>    0.148</td> <td>   -0.806</td> <td> 0.420</td> <td>   -0.409</td> <td>    0.171</td>
</tr>
<tr>
  <th>Franklin at Maple</th>                             <td>    0.2151</td> <td>    0.284</td> <td>    0.757</td> <td> 0.449</td> <td>   -0.342</td> <td>    0.772</td>
</tr>
<tr>
  <th>Golden Gate at Polk</th>                           <td>   -0.3595</td> <td>    0.148</td> <td>   -2.429</td> <td> 0.015</td> <td>   -0.649</td> <td>   -0.069</td>
</tr>
<tr>
  <th>Grant Avenue at Columbus Avenue</th>               <td>   -0.7598</td> <td>    0.121</td> <td>   -6.289</td> <td> 0.000</td> <td>   -0.997</td> <td>   -0.523</td>
</tr>
<tr>
  <th>Harry Bridges Plaza (Ferry Building)</th>          <td>   -0.0766</td> <td>    0.144</td> <td>   -0.534</td> <td> 0.593</td> <td>   -0.358</td> <td>    0.205</td>
</tr>
<tr>
  <th>Howard at 2nd</th>                                 <td>    0.0805</td> <td>    0.053</td> <td>    1.509</td> <td> 0.131</td> <td>   -0.024</td> <td>    0.185</td>
</tr>
<tr>
  <th>Japantown</th>                                     <td>    0.1814</td> <td>    0.140</td> <td>    1.292</td> <td> 0.197</td> <td>   -0.094</td> <td>    0.457</td>
</tr>
<tr>
  <th>MLK Library</th>                                   <td>   -0.1176</td> <td>    0.091</td> <td>   -1.300</td> <td> 0.194</td> <td>   -0.295</td> <td>    0.060</td>
</tr>
<tr>
  <th>Market at 10th</th>                                <td>   -0.5561</td> <td>    0.263</td> <td>   -2.116</td> <td> 0.034</td> <td>   -1.071</td> <td>   -0.041</td>
</tr>
<tr>
  <th>Market at 4th</th>                                 <td>   -0.0430</td> <td>    0.049</td> <td>   -0.870</td> <td> 0.384</td> <td>   -0.140</td> <td>    0.054</td>
</tr>
<tr>
  <th>Market at Sansome</th>                             <td>    0.1720</td> <td>    0.263</td> <td>    0.655</td> <td> 0.512</td> <td>   -0.343</td> <td>    0.687</td>
</tr>
<tr>
  <th>Mechanics Plaza (Market at Battery)</th>           <td>    0.0308</td> <td>    0.055</td> <td>    0.554</td> <td> 0.579</td> <td>   -0.078</td> <td>    0.140</td>
</tr>
<tr>
  <th>Mezes Park</th>                                    <td>   -0.1224</td> <td>    0.220</td> <td>   -0.557</td> <td> 0.577</td> <td>   -0.553</td> <td>    0.308</td>
</tr>
<tr>
  <th>Mountain View Caltrain Station</th>                <td>   -0.0470</td> <td>    0.150</td> <td>   -0.313</td> <td> 0.754</td> <td>   -0.341</td> <td>    0.247</td>
</tr>
<tr>
  <th>Mountain View City Hall</th>                       <td>    0.2196</td> <td>    0.135</td> <td>    1.629</td> <td> 0.103</td> <td>   -0.045</td> <td>    0.484</td>
</tr>
<tr>
  <th>Palo Alto Caltrain Station</th>                    <td>   -0.2996</td> <td>    0.164</td> <td>   -1.826</td> <td> 0.068</td> <td>   -0.621</td> <td>    0.022</td>
</tr>
<tr>
  <th>Park at Olive</th>                                 <td>    0.1597</td> <td>    0.172</td> <td>    0.931</td> <td> 0.352</td> <td>   -0.177</td> <td>    0.496</td>
</tr>
<tr>
  <th>Paseo de San Antonio</th>                          <td>    0.2829</td> <td>    0.141</td> <td>    2.012</td> <td> 0.044</td> <td>    0.007</td> <td>    0.558</td>
</tr>
<tr>
  <th>Post at Kearney</th>                               <td>    0.0319</td> <td>    0.055</td> <td>    0.581</td> <td> 0.561</td> <td>   -0.076</td> <td>    0.140</td>
</tr>
<tr>
  <th>Powell Street BART</th>                            <td>    0.0671</td> <td>    0.048</td> <td>    1.389</td> <td> 0.165</td> <td>   -0.028</td> <td>    0.162</td>
</tr>
<tr>
  <th>Powell at Post (Union Square)</th>                 <td>   -0.5399</td> <td>    0.054</td> <td>  -10.087</td> <td> 0.000</td> <td>   -0.645</td> <td>   -0.435</td>
</tr>
<tr>
  <th>Redwood City Caltrain Station</th>                 <td>   -0.2934</td> <td>    0.218</td> <td>   -1.344</td> <td> 0.179</td> <td>   -0.721</td> <td>    0.134</td>
</tr>
<tr>
  <th>Redwood City Medical Center</th>                   <td>    0.4035</td> <td>    0.204</td> <td>    1.977</td> <td> 0.048</td> <td>    0.003</td> <td>    0.804</td>
</tr>
<tr>
  <th>Redwood City Public Library</th>                   <td>    0.0352</td> <td>    0.270</td> <td>    0.130</td> <td> 0.896</td> <td>   -0.494</td> <td>    0.564</td>
</tr>
<tr>
  <th>Rengstorff Avenue / California Street</th>         <td>    0.0992</td> <td>    0.166</td> <td>    0.597</td> <td> 0.550</td> <td>   -0.226</td> <td>    0.425</td>
</tr>
<tr>
  <th>Ryland Park</th>                                   <td>   -0.0610</td> <td>    0.141</td> <td>   -0.432</td> <td> 0.666</td> <td>   -0.338</td> <td>    0.216</td>
</tr>
<tr>
  <th>SJSU - San Salvador at 9th</th>                    <td>    0.0308</td> <td>    0.163</td> <td>    0.189</td> <td> 0.850</td> <td>   -0.288</td> <td>    0.350</td>
</tr>
<tr>
  <th>SJSU 4th at San Carlos</th>                        <td>    0.2014</td> <td>    0.110</td> <td>    1.831</td> <td> 0.067</td> <td>   -0.014</td> <td>    0.417</td>
</tr>
<tr>
  <th>San Antonio Caltrain Station</th>                  <td>   -0.1331</td> <td>    0.160</td> <td>   -0.831</td> <td> 0.406</td> <td>   -0.447</td> <td>    0.181</td>
</tr>
<tr>
  <th>San Antonio Shopping Center</th>                   <td>    0.1479</td> <td>    0.140</td> <td>    1.059</td> <td> 0.290</td> <td>   -0.126</td> <td>    0.422</td>
</tr>
<tr>
  <th>San Francisco Caltrain (Townsend at 4th)</th>      <td>    1.5477</td> <td>    0.048</td> <td>   32.306</td> <td> 0.000</td> <td>    1.454</td> <td>    1.642</td>
</tr>
<tr>
  <th>San Francisco Caltrain 2 (330 Townsend)</th>       <td>   -0.2638</td> <td>    0.150</td> <td>   -1.762</td> <td> 0.078</td> <td>   -0.557</td> <td>    0.030</td>
</tr>
<tr>
  <th>San Francisco City Hall</th>                       <td>   -0.1910</td> <td>    0.072</td> <td>   -2.668</td> <td> 0.008</td> <td>   -0.331</td> <td>   -0.051</td>
</tr>
<tr>
  <th>San Jose City Hall</th>                            <td>   -0.0290</td> <td>    0.147</td> <td>   -0.197</td> <td> 0.844</td> <td>   -0.317</td> <td>    0.259</td>
</tr>
<tr>
  <th>San Jose Civic Center</th>                         <td>    0.2015</td> <td>    0.147</td> <td>    1.368</td> <td> 0.171</td> <td>   -0.087</td> <td>    0.490</td>
</tr>
<tr>
  <th>San Jose Diridon Caltrain Station</th>             <td>   -0.2274</td> <td>    0.266</td> <td>   -0.855</td> <td> 0.393</td> <td>   -0.749</td> <td>    0.294</td>
</tr>
<tr>
  <th>San Mateo County Center</th>                       <td>    0.1088</td> <td>    0.689</td> <td>    0.158</td> <td> 0.875</td> <td>   -1.242</td> <td>    1.460</td>
</tr>
<tr>
  <th>San Pedro Square</th>                              <td>    0.2088</td> <td>    0.134</td> <td>    1.557</td> <td> 0.119</td> <td>   -0.054</td> <td>    0.472</td>
</tr>
<tr>
  <th>San Salvador at 1st</th>                           <td>    0.2159</td> <td>    0.161</td> <td>    1.344</td> <td> 0.179</td> <td>   -0.099</td> <td>    0.531</td>
</tr>
<tr>
  <th>Santa Clara County Civic Center</th>               <td>    0.0785</td> <td>    0.160</td> <td>    0.490</td> <td> 0.624</td> <td>   -0.235</td> <td>    0.392</td>
</tr>
<tr>
  <th>Santa Clara at Almaden</th>                        <td>    0.2259</td> <td>    0.242</td> <td>    0.934</td> <td> 0.350</td> <td>   -0.248</td> <td>    0.700</td>
</tr>
<tr>
  <th>South Van Ness at Market</th>                      <td>   -0.2585</td> <td>    0.054</td> <td>   -4.756</td> <td> 0.000</td> <td>   -0.365</td> <td>   -0.152</td>
</tr>
<tr>
  <th>Spear at Folsom</th>                               <td>    0.0125</td> <td>    0.080</td> <td>    0.156</td> <td> 0.876</td> <td>   -0.144</td> <td>    0.169</td>
</tr>
<tr>
  <th>St James Park</th>                                 <td>    0.0476</td> <td>    0.146</td> <td>    0.326</td> <td> 0.745</td> <td>   -0.239</td> <td>    0.334</td>
</tr>
<tr>
  <th>Steuart at Market</th>                             <td>   -0.0907</td> <td>    0.144</td> <td>   -0.629</td> <td> 0.529</td> <td>   -0.373</td> <td>    0.192</td>
</tr>
<tr>
  <th>Temporary Transbay Terminal (Howard at Beale)</th> <td>   -0.4708</td> <td>    0.145</td> <td>   -3.256</td> <td> 0.001</td> <td>   -0.754</td> <td>   -0.187</td>
</tr>
<tr>
  <th>Townsend at 7th</th>                               <td>    0.4297</td> <td>    0.120</td> <td>    3.580</td> <td> 0.000</td> <td>    0.194</td> <td>    0.665</td>
</tr>
<tr>
  <th>University and Emerson</th>                        <td>    0.3834</td> <td>    0.264</td> <td>    1.454</td> <td> 0.146</td> <td>   -0.133</td> <td>    0.900</td>
</tr>
<tr>
  <th>Washington at Kearney</th>                         <td>    0.3853</td> <td>    0.125</td> <td>    3.076</td> <td> 0.002</td> <td>    0.140</td> <td>    0.631</td>
</tr>
<tr>
  <th>Yerba Buena Center of the Arts (3rd @ Howard)</th> <td>    0.1733</td> <td>    0.053</td> <td>    3.262</td> <td> 0.001</td> <td>    0.069</td> <td>    0.277</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>52195.952</td> <th>  Durbin-Watson:     </th>  <td>   1.998</td>  
</tr>
<tr>
  <th>Prob(Omnibus):</th>  <td> 0.000</td>   <th>  Jarque-Bera (JB):  </th> <td>2862194.801</td>
</tr>
<tr>
  <th>Skew:</th>           <td> 1.122</td>   <th>  Prob(JB):          </th>  <td>    0.00</td>  
</tr>
<tr>
  <th>Kurtosis:</th>       <td>25.698</td>   <th>  Cond. No.          </th>  <td>3.91e+16</td>  
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The smallest eigenvalue is 7.18e-24. This might indicate that there are<br/>strong multicollinearity problems or that the design matrix is singular.



## Model Performance


```python
###Calculating Root Mean Squared Error on the Test Data
predictions = model.predict(X_test.astype(float))

from sklearn.metrics import mean_squared_error
from math import sqrt

rms = sqrt(mean_squared_error(y_test, predictions))
```


```python
rms
```




    2.7857651919436424




```python

```


```python
###LASSO - Didn't Use
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
lasso=Lasso()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-4,1e-5,1e-2,1,5,10,20]}
lasso_regressor=GridSearchCV(lasso,parameters,scoring='neg_mean_squared_error',cv=5)
lasso_regressor.fit(X,y)
print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)
```

## Potential Improvements

#### To improve model performance and provide better predictions of net rate bike renting, I would try the following:

Using a stepwise variable selection technique with linear regression to tease out the significant variables

Checking the other assumptions of the linear regression (linearity, normality and constant variance of the errors, independence)

Using a regularized regression technique such as Ridge regression to better deal with multicollinearity or LASSO regression for less biased variable selection
         
Engineering more features

## Conclusion
I used a linear regression model to predict the net change in bikes at each station at a given hour. In the model,
I used variables such as proportion of subscribers renting, the hour of the day, the month of the year, the station name, how many bikes the station could hold and various weather factors. Overall, the model predicted net change with a root mean squared error of 2.78 on the validation data. 


```python

```
