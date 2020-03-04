# Welcome to my Data Science Portfolio
## Benjamin Roberts
- blrober5@ncsu.edu
- Institute for Advanced Analytics

## About Me
I am a creative and detailed-oriented data scientist with experience conducting end-to-end, value-added analyses on large, complex datasets. My passion for data comes from my undergraduate studies in Economics and Political Science, where I focused my research on the quantitative relationship between policies and economic outcomes. For my honors thesis, I utilized various statistical modeling techniques to examine the causal relationship between state-level reproductive policies and female labor supply. I am currently completing my MS in Analytics at the Institute for Advanced Analytics (IAA) at North Carolina State University and will graduate in May. During my studies at the IAA, my practicum team has served as consultants to Trillium Health Resources, one of North Carolina's largest Medicaid Managed Care Organizations. Over these 8 months, we have led the Prediction as a Path to Prevention project to identify Medicaid members at high-risk for suicide or self-harm incidents.

## Projects
### Machine Learning Translation
My most recent project has involved the use of Gated Recurrent Units (GRU) using Keras in Python to translate English input text to Spanish. To train the network, I utilized the EuroParl parallel corpus from NLTK, which contains text from speeches in the European Parliament dictated in every European language. For this project, I focused on the English and Spanish texts, training the model on over 1.5 million unique sentences from this corpus. The vocabulary of each language was restricted to the 30,000 most common words. You can test out some translations here! The code for preprocessing the data and training the model can also be found here.

### Bayesian Hyperparameter Tuning, Evaluating Performance and Intepreting LightGBM Models
During our machine learning class at the IAA, my team was tasked with producing a model that balanced Sensitivty (True Positive Rate) and Precision (Positived Predicted Value) on out-of-sample data for a provided dataset. As part of this project, I utilized Bayesian HyperParameter tuning to train a LightGBM model, with the goal being to efficiently identify optimal hyperparameter values on the large dataset. In this notebook, I created functions in Python using Pandas and Scikit-Learn for preprocessing the data (oversampling, imputations, encoding categorical variables), conducting Bayesian HyperParamter tuning to maximize AUC on out-of-sample data, and building and evaluating the model for performance (balance of TPR and PPV) on out-of-sample data. These functions can be applied to any dataset. I also provide code for determining variable importance and evaluating the relationship between key variables and the target through SHAP values. The SHAP values show how predictions for the target change across different levels of the variables, as well as how each variable contributes to the predicted probabilities for each observation.

### Text Analytics: Topic Modeling Descriptions of UFO Sightings Using Latent Semantic Analysis
As part of our text analytics class, my team used a Kaggle dataset of described UFO sightings to determine common words and themes for these sightings. After preprocessing the data, including removing stop words from and porter stemming the descriptions, we used Latent Semantic Analysis (SVD) to reduce the data from a vocabulary of 10,000 unique terms to 15 topics/concepts. We then summarized these concepts by examining the highest weighted terms for each topic. By determining the scores of the comments within each topic (multiplying the terms in the comment by their weights for each concept), we were able to identify the comments that were most related to each topic. This provided a more complete summary of each topic. For example, Topic 1's highest scored comments dealt with 'Bright light moving in the sky', while in Topic 3 the most related comments described 'Fire in the sky'. From this, we assumed most of these people were seeing planes or fireworks, especially due to the fact that most of the comments occurred on July 4th. My Python notebook for this project can be found here.

### Network Analysis: Examining Fake News Echo Chambers Across Political Facebook Groups
During my time studying abroad in Helsinki, Finland, I took a class on Network Analysis using R. For my final project, I wanted to use this technique to explore how the profileration of fake news during the recent 2016 presidential election might have exacerabted echo chambers on social media. Using a Facebook API, I pulled shared news links from 14 distinct politcal Facebook groups, gathering information from 7 Liberal/Left-Leaning pages ('imwithhillary2016', 'OccupyDemocrats') and 7 Conservative/Right-Leaning pages ('DonaldTrump4President', 'conservativedaily'). Using these news links, I created a network with the political pages as the nodes and the number of shared links between pages as the weighted edges. This network of all shared news sources exhibited high connectivity between pages of all ideologies, showing no evidence for echo chambers amongst partisan Facebook pages in terms of general domains. However, I then created a network with only news domains determined by academic sources to be fake news. This network demonstrated much greater polarization, with two major clusters forming in the networ separating the distinct liberal and conversative communities. Thus, the analysis indicated that the sharing of fake news on partisan Facebook pages not only contributed to echo chambers within liberal and conservative communities on the social media site, it created echo chambers amongst communities where they ostensibly didn’t exist before. The network of political Facebook groups and fake news domains with clustered communities is shown below.
[Rplot07.pdf](https://github.com/blrober5/benroberts.github.io/files/4284333/Rplot07.pdf)
The code for the network analysis can be found here and the research paper detailing the methodology and results can be found here.

### Case Study (Insurance/Consulting)

### Thesis 




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

