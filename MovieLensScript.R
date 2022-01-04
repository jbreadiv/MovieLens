#Rscript for Capstone Report - MovieLens
# J.B. Read
# 01/02/2022
###

# Install packages 

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

# Load packages in library

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

# Download MovieLens file

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

# Create variable for ratings; takes a long time to unzip

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))), col.names = c("userId", "movieId", "rating", "timestamp"))

# Create variable for movies; takes a long time to unzip

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 4.0 or later:

movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId), title = as.character(title), genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") 
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

# Building on previous course lessons, build a first model (Chp. 33.7.4)
mu_hat <- mean(edx$rating)
mu_hat

# Unknown ratings are then created and predicted
naive_rmse <- RMSE(validation$rating, mu_hat)
naive_rmse

# Building on previous course lessons, model movie effect (Chp. 33.7.5)
# Create b_i for average movie rank
fit <- edx %>%
  group_by(movieId) %>%
  summarize(fit = mean(rating - mu_hat))
fit
# Predict ratings for mu_hat and fit
ratingpredict <- validation %>% 
  left_join(fit, by='movieId') %>%
  mutate(pred = mu_hat + fit) %>%
  pull(pred)

# RMSE is calculated
RMSE(validation$rating, ratingpredict)

# Plot the distribution
qplot(fit, data = fit, bins = 20, color = I("black"))

# Then Compute user effects (chp. 33.7.6)
user_avgs <- edx %>% 
  left_join(fit, by='movieId') %>%
  group_by(userId) %>%
  summarize(user_avgs = mean(rating - mu_hat - fit))
user_avgs

# Generate ratings including bias
ratingpredict <- validation %>% 
  left_join(fit, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu_hat + fit + user_avgs) %>%
  pull(pred)

# RMSE of movie effect
RMSE(ratingpredict, validation$rating)

# Regularisation of the data
lambdas <- seq(from=0, to=10, by=0.25)

# Calculating RMSE with each lambda, this take a long time to run (choosing the penalty term chp.33.9.3)
rmses <- sapply(lambdas, function(l){
  mu_hat <- mean(edx$rating)
  fit <- edx %>% 
    group_by(movieId) %>%
    summarize(fit = sum(rating - mu_hat)/(n()+l))
  user_avgs <- edx %>% 
    left_join(fit, by="movieId") %>%
    group_by(userId) %>%
    summarize(user_avgs = sum(rating - fit - mu_hat)/(n()+l))
  ratingpredict <- validation %>% 
    left_join(fit, by = "movieId") %>%
    left_join(user_avgs, by = "userId") %>%
    mutate(pred = mu_hat + fit + user_avgs) %>%
    pull(pred)
  return(RMSE(ratingpredict, validation$rating))
})

qplot(lambdas, rmses)

min(rmses)

#Create final model with movie effect and user effect with regularisation
l <- lambdas[which.min(rmses)]

fit1 <- edx %>% 
  group_by(movieId) %>%
  summarize(fit1 = sum(rating - mu_hat)/(n()+l))

user_avgs1 <- edx %>% 
  left_join(fit1, by="movieId") %>%
  group_by(userId) %>%
  summarize(user_avgs1 = sum(rating - fit1 - mu_hat)/(n()+l))

ratingpredict1 <- 
  validation %>% 
  left_join(fit1, by = "movieId") %>%
  left_join(user_avgs1, by = "userId") %>%
  mutate(pred = mu_hat + fit1 + user_avgs1) %>%
  pull(pred)

RMSE(ratingpredict1, validation$rating)



