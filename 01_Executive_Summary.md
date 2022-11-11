

We are the quality assurance team at Airbnb. At Airbnb, our company values dictate that we consider any review Less than five stars to indicate major issues with the property and/or host. 

##### In order to inform our division and provide feedback to hosts, we want to build classification models to predict--based off of either property features or prior reviews--whether a property is poorly or highly rated.  To uphold our stringent expectations for quality, we define poorly rated as less than 4.5 stars and highly rated as greater than 4.9 stars.








Originally, we had planned to build prediction models using a subset of the data from one neighborhood and predicting the property ratings as a continuous variable. However, after building initial models and examining the spread of our target variable, we realized that a substantial majority of the reviews were clustered at 4.8 or above, with no reviews for that neighborhood below 3.5, meaning that the variables we selected to explain changes in average property rating would be mostly predicting very small changes with comparatively crude measures like number of bedrooms and bathrooms. This extreme right skew was reflected in the larger data set as well. Because of this, we were seeing very weak performance from our models and knew we would have to take a different approach.

 We decided to go back to the full data set and take a subsection of that data set, selecting reviews under 3 stars and an equal number of randomly selected reviews above 4.9 stars to get the extreme ends of the data set. We then classified the reviews as either bad reviews or good reviews and built classification models to predict the type of review, hoping that we could build models that could differentiate between more extreme differences. This led to a revision of our problem statement.