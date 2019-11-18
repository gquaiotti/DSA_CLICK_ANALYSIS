# DATA SCIENCE ACADEMY 
# Big Data Analytics com R e Microsoft Azure Machine Learning
#
# Fraud Detection on Click Traffic in Mobile Application Ads
# Predict if an user will download an application after clicking in Mobile Application Ads
#
# Gabriel Quaiotti
# Nov 2019
#
# Read the train_complete.csv file that contains the transformed training dataset
# Check if the dataset is unbalanced
# If it is umbalanced, SMOTE it and generate the train_smote.csv file

setwd('D:/Github/DSA_CLICK_ANALYSIS')

library(data.table)
library(DMwR)

v_c_file_train <- 'dataset/train_complete.csv'
v_c_file_smote <- 'dataset/train_smote.csv'

# Load dataset
train <- data.table::fread(file = v_c_file_train,
                           colClasses = rep(x = "factor", 10))

gc()

# DATASET BALANCE
t <- table(train[, is_attributed])

# If the dataset is unbalanced (more than 30%), SMOTE it!
if (t[2] / t[1] <= 0.30) {
  train <- SMOTE(form = is_attributed ~ ., data = train)
}

data.table::fwrite(file = v_c_file_smote, x = train)

remove(train)
remove(v_c_file_train)
remove(v_c_file_smote)
