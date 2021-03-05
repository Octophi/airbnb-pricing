# Airbnb Pricing

A machine learning project looking to perform mutli-class classification task predicting price (classified as 1, 2, 3, or 4) of Airbnb listing based on features such as number of bedrooms, number of bathrooms, etc. This project was completed as the final project for the course Duke CS671: Theory and Algorithms of Machine Learning.

## Description of Dataset

The training dataset is a set of approximately 10,000 observations, with 24 features (including number_of_reviews, availability_365, etc) as well as an output variable price which takes on an integer value within the range 1-4 inclusive, with 1 representing the lowest cost bracket and 4 representing the highest cost bracket. The dataset was provided as part of a Kaggle competition.

## Exploratory Analysis

First, I checked whether the dataset was balanced - it was. Following, I made violin plots plotting each variable against the price to get some basic intuition regarding which variables might be more predictive. I also made a few other plots, including a PCA embedding of the numberical data, which was unilluminating, and some scatter plots of variables against price.

While inspecting the data, it was found that the variable is_business_travel_ready only had one unique feature, and thus this feature was discarded entirely.

Not all of the features were readily available to be fed into machine learning techniques. Certain date features such as host_since were processed to compute new features such as num_months_since_host, which computed the number of months since the time a user became a host to the final time in which someone became host in the dataset. Other categorical features such as neighbourhood were encoded using an ordinal encoder. 

I also tried engineering some new features based on the existing ones, such as bed_room_quotient, which is the quotient between the number of beds and bedrooms, as well as bed_bath_quotient. These turned out to not be very effective in our test models, and thus were not included in the final model.

## Models 

I tried a few different models, including random forest, boosted decision stumps, and a k-nearest neighbors model. Ultimately, the random forest and k-nearest neighbors model were submitted as the final models for the project.

Random forest was my first choice because it is naturally adaptable to a multiclass classification problem and can deal with categorical data upon performing ordinal encoding. It can learn nonlinearities well, and is a powerful out of the box method - in fact, though it was the first model I tried, it turned out to be the most effective out of all my models.

For the k-nearest neighbors model, I used a metric called the Gower distance to measure distance, which is a metric on mixed data, that is, data including both categorical and numerical features. It uses the Hamming distance on categorical features and scaled Manhattan distance on numerical features, generating the overall distance betwqeen points as a linear combination of distances between feature values. This was my first time using the Gower distance and it was an interesting experiment because it seemed potentially well-suited to deal with variables which perhaps are not well-encoded by an ordinal encoder, such as neighbourhood. 

## Training 

For random forest, I used the standard sklearn implementation of random forest, which does decision tree fitting using the CART algorithm. For k-nearest neighbors, I used a precomputed distance matrix computed using the gower Python package, and this distance matrix was fed into a standard KNeighborsClassifier.

## Hyperparameter Selection

For random forest, the main hyperparameters to tune were max_features and n_estimators. I began with a randomized grid search, considering different possible values for splitting criterion, number of estimators, max depth of tree, and so on. After narrowing down on what ranges of feature values seemed most effective, I performed a more systematized grid search to finalize the values of max_features and n_estimators. Throughout, I used 5-fold cross validation with the same fixed folds to evaluate the performance of any given set of hyperparameters. 

For k-nearest neighbors, the main hyperparameter to tune was the value of k. I evaluated 5-fold cross validation accuracy of the model for 12 different values of k ranging from 3 to 89, not all equally spaced, and chose k=21 since this appeared to be an inflection point. The cross validation was performed using the same splits as above.

## Results

The random forest model consistently achieved a mean 5-fold cross validation score of approximately 0.55 while the k-nearest neighbors apporach achieved a 5-fold cross validation score of 0.43. These were both well above the 0.25 one would expect from random guessing, but were not as high as I had hoped.

## Notes

For more details, including references, see the included PDF writeup.
