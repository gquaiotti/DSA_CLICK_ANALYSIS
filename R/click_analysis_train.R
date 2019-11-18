# DATA SCIENCE ACADEMY 
# Big Data Analytics com R e Microsoft Azure Machine Learning
#
# Fraud Detection on Click Traffic in Mobile Application Ads
# Predict if an user will download an application after clicking in Mobile Application Ads
#
# Gabriel Quaiotti
# Jul 2019
#
# Read the train_smote.csv file
# Generate the trained model
#

setwd('D:/Github/DSA_CLICK_ANALYSIS')

v_c_file_train <- 'dataset/train_smote.csv'
# v_c_file_test <- 'dataset/test_split.csv'

library(data.table)
library(e1071)
# library(caret)

# Load dataset
train <- data.table::fread(file = v_c_file_train,
                           colClasses = rep(x = "factor", 10))

remove(v_c_file_train)

# TRAIN
# Naive-Bayes
gc()
model <- e1071::naiveBayes(is_attributed ~ ., train)
saveRDS(model, "model/model_naivebayes.rds")

remove(train)

# Test
# test <- data.table::fread(file = v_c_file_test,
#                          colClasses = rep(x = "factor", 10))

# gc()
# predict <- predict(object = model, newdata = test)
# cm <- confusionMatrix(test[, is_attributed], predict, positive = "1")
# print(cm)

remove(model)

# remove(predict)
# remove(cm)
# remove(test)