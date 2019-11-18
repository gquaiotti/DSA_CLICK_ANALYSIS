# DATA SCIENCE ACADEMY 
# Big Data Analytics com R e Microsoft Azure Machine Learning
#
# Fraud Detection on Click Traffic in Mobile Application Ads
# Predict if an user will download an application after clicking in Mobile Application Ads
#
# Gabriel Quaiotti
# Nov 2019
#
# Read and transform the test dataset (test_supplement.csv file)
# Generate the test_complete.csv file according to the trained model
#

setwd('D:/Github/DSA_CLICK_ANALYSIS')

v_c_file_test <- 'dataset/test_supplement.csv'

library(data.table)
library(fasttime)
library(e1071)

# Load dataset
test <- data.table::fread(file = v_c_file_test,
                          colClasses = c(click_id = "integer",
                                         ip = "character", 
                                         app = "factor", 
                                         device = "factor", 
                                         os = "factor",
                                         channel = "factor", 
                                         click_time = "character"))

remove(v_c_file_test)

str(test)

# TRANSFORM
# Order by IP and Click Time
data.table::setorder(x = test, ip, click_time)

# Get click_order by IP
test[, click_order := frank(x = .I), by = ip]

# Categorize by click_order
test[click_order == 1, click_order_class := 1] 
test[click_order > 1 & click_order <= 10, click_order_class := 2] 
test[click_order > 10 & click_order <= 20, click_order_class := 3] 
test[click_order > 20 & click_order <= 30, click_order_class := 4] 
test[click_order > 30 & click_order <= 40, click_order_class := 5] 
test[click_order > 40 & click_order <= 50, click_order_class := 6] 
test[click_order > 50 & click_order <= 100, click_order_class := 7] 
test[click_order > 100, click_order_class := 8]

# Remove unused column
test[, click_order := NULL]

gc()

# Get last click from the same IP
# Get diff between last click time and click time
# Categorize last click timeand click time difference
test[, last_click_time := ifelse(ip == data.table::shift(ip), data.table::shift(click_time), NA)]
test[!is.na(last_click_time), last_click_time_diff := difftime(time1 = fastPOSIXct(click_time), time2 = fastPOSIXct(last_click_time), units = ("mins"))]
test[, last_click_time := NULL]

gc()

test[is.na(last_click_time_diff), last_click_time_class := 1]
test[last_click_time_diff < 1, last_click_time_class := 2]
test[last_click_time_diff >= 1 & last_click_time_diff < 60, last_click_time_class := 3]
test[last_click_time_diff >= 60 & last_click_time_diff < 1440, last_click_time_class := 4]
test[last_click_time_diff >= 1440 & last_click_time_diff < 43200, last_click_time_class := 5]
test[last_click_time_diff >= 43200 & last_click_time_diff < 518400, last_click_time_class := 6]
test[last_click_time_diff >= 518400, last_click_time_class := 7]

test[, last_click_time_diff := NULL]

gc()

# Get next click from the same IP
# Get diff between click time and next click time
# Categorize click time and last click time difference
test[, next_click_time := ifelse(ip == data.table::shift(ip, type = "lead"), data.table::shift(click_time, type = "lead"), NA)]
test[!is.na(next_click_time), next_click_time_diff := difftime(time1 = fastPOSIXct(next_click_time), time2 = fastPOSIXct(click_time), units = ("mins"))]
test[, next_click_time := NULL]

gc()


test[is.na(next_click_time_diff), next_click_time_class := 1]
test[next_click_time_diff < 1, next_click_time_class := 2]
test[next_click_time_diff >= 1 & next_click_time_diff < 60, next_click_time_class := 3]
test[next_click_time_diff >= 60 & next_click_time_diff < 1440, next_click_time_class := 4]
test[next_click_time_diff >= 1440 & next_click_time_diff < 43200, next_click_time_class := 5]
test[next_click_time_diff >= 43200 & next_click_time_diff < 518400, next_click_time_class := 6]
test[next_click_time_diff >= 518400, next_click_time_class := 7]
test[, next_click_time_diff := NULL]

gc()

test[, click_count := (.N), by = ip]

test[click_count == 1, ip_click_count := 1] 
test[click_count > 1 & click_count <= 10, ip_click_count := 2] 
test[click_count > 10 & click_count <= 20, ip_click_count := 3] 
test[click_count > 20 & click_count <= 30, ip_click_count := 4] 
test[click_count > 30 & click_count <= 40, ip_click_count := 5] 
test[click_count > 40 & click_count <= 50, ip_click_count := 6] 
test[click_count > 50 & click_count <= 100, ip_click_count := 7] 
test[click_count > 100, ip_click_count := 8]

test[, click_count := NULL]
test[, ip := NULL]

gc()

# Get the HOUR of click_time
test[, hour := as.factor(lubridate::hour(fastPOSIXct(click_time, tz = "GMT")))]

test[, click_time := NULL]

gc()

fwrite(x = test, file = 'dataset/test_complete.csv')

remove(test)