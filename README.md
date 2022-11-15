# Airbnb-Rating-Prediction

## Problem Statement
We are the quality assurance team at Airbnb. At Airbnb, our company values dictate that we consider any review less than five stars to indicate major issues with the property and/or host. 
#### In order to inform our division and provide feedback to hosts, we want to build classification models to predict--based off of either property features or prior reviews--whether a property is poorly or highly rated.  To uphold our stringent expectations for quality, we define poorly rated as less than 4.5 stars and highly rated as greater than 4.9 stars.


## Table of Contents 
1. 01_Executive_Summary
2. 02_EDA
3. 03_Modeling
4. 04_Neural Network Models

The data used in this project was sourced from Airbnb. The dataset gives information on various features of Airbnb listings as well as reviews left for listings. The data was collected from listings in Athens, Greece.

Link to data: http://insideairbnb.com/get-the-data/

From this link, the files the project uses are listings.csv.gz and reviews.csv.gz. These can be found by scrolling to the heading 'Athens, Attica, Greece' and selecting the first and third linked files.

Data was downloaded as a CSV file from the Airbnb website. It was read into datasets using Pandas software. The data files were larger than the limit GitHub allows to be uploaded, so Git Large File Storage was used to create pointer files that could be uploaded and used as a map to find the large files. In order to download the files and run the notebooks, users need to have Git Large File Storage installed on their computer. Instructions for installing Git LFS can be found here: https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage

Data cleaning for the listings dataset involved selecting relevant variables, dropping null values in our target variable, removing outliers in the price column, creating dummy variables for categorical data, and using linear imputation to impute missing values. Data cleaning for the reviews data set involved removing rows with missing reviews, removing non-alphanumeric characters from the reviews, and lemmatizing the text data. For both datasets we took the continuous variable of ratings and used it to create two classes, poorly rated listings and highly rated listings, which we then used as our target variable for our classification models to predict. 

## Software Requirements
This analysis uses Pandas and Numpy to work with dataframe data. For visualizations, this analysis uses MatPlotLib and Seaborn. Preprocessing and modeling was done using packages from Sklearn, Spacy, Natural Language Toolkit, and TensorFlow.
