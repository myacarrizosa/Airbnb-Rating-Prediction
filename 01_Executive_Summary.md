# Executive Summary

## Problem Statement

We are the quality assurance team at Airbnb. At Airbnb, our company values dictate that we consider any review Less than five stars to indicate major issues with the property and/or host. 

#### In order to inform our division and provide feedback to hosts, we want to build classification models to predict--based off of either property features or prior reviews--whether a property is poorly or highly rated.  To uphold our stringent expectations for quality, we define poorly rated as less than 4.5 stars and highly rated as greater than 4.9 stars.


Originally, we had planned to build prediction models using a subset of the data from one neighborhood and predicting the property ratings as a continuous variable. However, after building initial models and examining the spread of our target variable, we realized that a substantial majority of the reviews were clustered at 4.8 or above, with no reviews for that neighborhood below 3.5, meaning that the variables we selected to explain changes in average property rating would be mostly predicting very small changes with comparatively crude measures like number of bedrooms and bathrooms. This extreme right skew was reflected in the larger data set as well. Because of this, we were seeing very weak performance from our models and knew we would have to take a different approach.

 We decided to go back to the full data set and take a subsection of that data set, selecting reviews under 4.5 stars and an equal number of randomly selected reviews above 4.9 stars to get the extreme ends of the data set. We then classified the reviews as either bad reviews or good reviews and built classification models to predict the type of review, based on the logic that there is some aspect that makes these categories different that a model could pick up on. To reiterate our final problem statement:
 
 #### In order to inform our division and provide feedback to hosts, we want to build classification models to predict--based off of either property features or prior reviews--whether a property is poorly or highly rated.  To uphold our stringent expectations for quality, we define poorly rated as less than 4.5 stars and highly rated as greater than 4.9 stars.
 
 
 
## Description of Data
The data used in this project comes from two datasets provided by Airbnb with information on properties in Athens, Greece. We 


text data consisting of Reddit submissions (title and text) posted on the r/AskPhilosophy and r/Religion subreddits. The AskPhilosophy subreddit was chosen instead of the Philosophy subreddit because the Philosophy subreddit mainly consisted of links to external pages and videos, which wouldn't provide the text data needed to compare and classify the pages. After data cleaning, there were 4945 data records.
 
 ## Data Dictionary

|Feature|Type|Dataset|Description|
|---|---|---|---|
|neighbourhood|object|listings_subset| Neighborhood in Athens, Attica, Greece| 
|host_response_time|object|listings_subset| Either within an hour, within a few hours, within a day, or a few days or more. Dummy variables were made for each category|
|host_total_listings_count|float|listings_subset| The number of listings the host has (per Airbnb calculations) |
|host_identity_verified|object|listings_subset| 0 if the host is not verified, 1 if the host is verified|
|property_type|object|listings_subset| Self selected property type. Hotels, bed and breakfasts, and private residences are described as such by their hosts in this field|
|room_type|object|listings_subset| Either entire home/apt, private room, shared room, or hotel. Dummy variables were made for each category|
|accomodates|integer|listings_subset| How many guests the listing accodomodates |
|bathrooms_text|object|listings_subset|How many bathrooms in the listing. Dummy variables were made for each of the categories |
|bedrooms|float|listings_subset| Number of bedrooms |
|beds|float|listings_subset| Number of beds |
|price|float|listings_subset| Daily price|
|minimum_nights|integer|listings_subset|Minimum number of nights the listing can be booked for |
|maximum_nights|integar|listings_subset|Maximum number of nights the listing can be booked for |
|number_of_reviews|integer|listings_subset|The number of reviews the listings has|
|instant_bookable|object|listings_subset|0 if the booking must be approved by the host, 1 if the listing can be instantly booked |
|calculated_host_listings_count|integer|listings_subset|How many listings the host has with Airbnb |
|calculated_host_listings_count|integer|listings_subset|How many listings the host has with Airbnb |
|reviews_scores_rating|float|listings_subset & reviews_subset|Average rating of the listing|
|type|integer|listings_subset & reviews_subset|Highly rated (defined as reviews_scores_rating > 4.9) denoted as 0, poorly rated (defined as reviews_scores_rating < 4.5) denoted as 1 |

 
 
 
 
 
 
 
 http://insideairbnb.com/get-the-data/
 
 
 *logreg l:  Train accuracy: 0.7003188097768331, Test accuracy: 0.7038216560509554
 knn l: Train accuracy: 0.9989373007438895, Test accuracy: 0.6624203821656051
 decision tree l: Train accuracy: 0.8257173219978746, Test accuracy: 0.6815286624203821
 *random forest l: Train accuracy: 0.7837407013815091, Test accuracy: 0.7245222929936306
 AdaBoost l:  Train accuracy: 0.7587672688629118, Test accuracy: 0.6942675159235668
 Gradient Boost l:  Train accuracy: 0.7587672688629118, Test accuracy: 0.6942675159235668
 Bagging l: Train accuracy: 0.997874601487779, Test accuracy: 0.7070063694267515


logreg r: Train accuracy: 0.8643874178940067, Test accuracy: 0.7713866111859013
knn r:  Train accuracy: 0.7563542899106523, Test accuracy: 0.6545098519153102
decision tree r: Train accuracy: 0.6898535351474848, Test accuracy: 0.6543874678741892
random forest r: Train accuracy: 0.7340377789563869, Test accuracy: 0.704442540692693
*adaboost r: Train accuracy: 0.7594549385989964, Test accuracy: 0.7496022518663567
gradient boost r:  Train accuracy: 0.8064542450328424, Test accuracy: 0.7486231795373883
bagging r: 
nn1: Train Accuracy: 0.8076781630516052, Test Accuracy: 0.5368987917900085
nn2: Train Accuracy: 0.8428052663803101, Test Accuracy: 0.5323705673217773
nn3: Train Accuracy: 0.6062992215156555, Test Accuracy: 0.5532982349395752