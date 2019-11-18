# DATA SCIENCE ACADEMY 
# Big Data Analytics com R e Microsoft Azure Machine Learning
#
# Fraud Detection on Click Traffic in Mobile Application Ads
# Predict if an user will download an application after clicking in Mobile Application Ads
#
# Gabriel Quaiotti
# Nov 2019
#
# Read the transformed test dataset
# Recover the trained model
# Predict the target variable
# Generate the predict.csv file with the result

setwd('D:/Github/DSA_CLICK_ANALYSIS')

v_c_file_test <- 'dataset/test_complete.csv'
v_c_file_submit <- 'prediction/predict.csv'

library(data.table)
library(e1071)

# Load dataset
test <- data.table::fread(file = v_c_file_test,
                          colClasses = c(click_id = "integer",
                                         app = "factor", 
                                         device = "factor",
                                         os = "factor",
                                         channel = "factor",
                                         click_order_class = "factor",
                                         last_click_time_class = "factor",
                                         next_click_time_class = "factor",
                                         ip_click_count = "factor",
                                         hour = "factor"))

remove(v_c_file_test)

model <- readRDS(file = "model/model_naivebayes.rds")

gc()
p <- predict(object = model, newdata = test)

test[, is_attributed := p[.I]]

data.table::fwrite(file = v_c_file_submit, x = test[,c("click_id", "is_attributed")])

remove(model)
remove(p)
remove(v_c_file_submit)
remove(test)