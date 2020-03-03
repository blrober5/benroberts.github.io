# Welcome to my Data Science Portfolio
## Benjamin Roberts
### blrober5@ncsu.edu
### Institute for Advanced Analytics

## About Me
I am a creative and detailed-oriented data scientist with experience conducting end-to-end, value-added analyses on large, complex datasets. My passion for data comes from my undergraduate studies in Economics and Political Science, where I focused my research on the quantitative relationship between policies and economic outcomes. I am currently completing my MS in Analytics at the Institute for Advanced Analytics at North Carolina State University and will graudate in May.

## Project 1
```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```


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

