# Welcome to my Data Science Portfolio
## Benjamin Roberts
- blrober5@ncsu.edu
- Institute for Advanced Analytics

## About Me
I am a creative and detailed-oriented data scientist with experience conducting end-to-end, value-added analyses on large, complex datasets. My passion for data comes from my undergraduate studies in Economics and Political Science, where I focused my research on the quantitative relationship between policies and economic outcomes. For my honors thesis, I utilized various statistical modeling techniques to examine the causal relationship between state-level reproductive policies and female labor supply. I am currently completing my MS in Analytics at the Institute for Advanced Analytics at North Carolina State University and will graduate in May. During my studies at the IAA, my practicum team has served as consultants to Trillium Health Resources, one of North Carolina's largest Medicaid Managed Care Organizations. Over these 8 months, we have led the Prediction as a Path to Prevention project to identify Medicaid members at high-risk for suicide or self-harm incidents.

## Projects
### Machine Learning Translation
My most recent project has involved the use of Gated Recurrent Units (GRU) to translate English input text to Spanish. To train the network, I utilized the EuroParl parallel corpus from NLTK, which contains text from speeches in the European Parliament dictated in every European language. For this project, I focused on the English and Spanish texts, training the model on over 1.5 million unique sentences from this corpus. The vocabulary of each language was restricted to the 30,000 most common words. You can test out some translations here!
The code for preprocessing the data and training the model can also be found here.

### Hyperparameter Tuning, Evaluating Performance and Intepreting LightGBM Models

### Network Analysis: Examining Fake News Echo Chambers Across Political Facebook Groups

### Case Study (Insurance/Consulting)





[Project 1](CaseStudy_Consulting.md)
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

