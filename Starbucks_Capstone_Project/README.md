# The Starbucks Capstone Project for Udacity's Data Scientist Nanodegree Program
## Predict Whether Someone Will Complete an Offer Based on Their Income

### Introduction
This data set contains simulated data that mimics customer behavior on the Starbucks rewards mobile app. Once every few days, Starbucks sends out an offer to users of the mobile app. An offer can be merely an advertisement for a drink or an actual offer such as a discount or BOGO (buy one get one free). Some users might not receive any offer during certain weeks. Not all users receive the same offer, and that is the challenge to solve with this data set.
In this project I want to combine transaction, demographic and offer data to determine which demographic groups respond best to which offer type. This data set is a simplified version of the real Starbucks app because the underlying simulator only has one product whereas Starbucks actually sells dozens of products.

### Project Overview
I choose the starbucks capstone project to build a machine learning model that Predict Whether Someone Will Complete an Offer Based on Their Income. I'll combine the portfolio, profile, and transaction and make some machine learning models. For evaluating the models I chose accuracy and F1-score metric. 
I choose accuracy as a matrix because it will be used as a reference for algorithm performance if the data set has a very close number of False Negatives and False Positives. However, if the numbers are not close, then I should use the F1 Score as a reference.

### Data Sets Description
The data is contained in three files:
portfolio.json - containing offer ids and meta data about each offer (duration, type, etc.)
profile.json - demographic data for each customer
transcript.json - records for transactions, offers received, offers viewed, and offers completed

Here is the schema and explanation of each variable in the files:
1. portfolio.json
id (string) - offer id
offer_type (string) - type of offer ie BOGO, discount, informational
difficulty (int) - minimum required spend to complete an offer
reward (int) - reward given for completing an offer
duration (int) - time for offer to be open, in days
channels (list of strings)

2. profile.json
age (int) - age of the customer
became_member_on (int) - date when customer created an app account
gender (str) - gender of the customer (note some entries contain 'O' for other rather than M or F)
id (str) - customer id
income (float) - customer's income

3. transcript.json
event (str) - record description (ie transaction, offer received, offer viewed, etc.)
person (str) - customer id
time (int) - time in hours since start of test. The data begins at time t=0
value - (dict of strings) - either an offer id or transaction amount depending on the record

### Conclusion
In this project I do several things such as cleaning data, exploring data, combining data, building models and evaluating models.
Cleaning data is quite complicated in this project because there are data columns that must be separated into several data columns. 
There are a few things I did to make machine learning models such as remove data that has empty values, and change some data types to make it easier to create models. In data exploration, I only focus on things related to customer income.
In the section on making machine learning models, by comparing several models such as Decision Tree Classifier, Random Forest Classifier and KNeighbors Classifier. The results of these predictions Decision Tree Classifier has better accuracy and F1 scores than Random Forest Classifier and KNeighbors Classifier. 
Decision Tree Classifier has training and testing accuracy: 0.622 and 0.623 and training and testing F1 score: 0.593 and 0.597.
However, this model does not yet have an optimal accuracy and F1 score. We should do improvement or try another model that has accuracy and the F1 score is higher than this model.
