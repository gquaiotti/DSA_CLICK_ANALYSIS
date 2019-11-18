# DATA SCIENCE ACADEMY 
# Big Data Analytics com R e Microsoft Azure Machine Learning
#
# Fraud Detection on Click Traffic in Mobile Application Ads
# Predict if an user will download an application after clicking in Mobile Application Ads
#
# Gabriel Quaiotti
# Jul 2019
#
# CLICK_ANALISYS_TRANSFORM
# 
# Read the train.csv file, transform and writes three files: 
# train_complete.csv :: transformed training dataset
# train_split.csv :: splited training dataset for sample training
# test_split.csv :: splited testing dataset for sample evaluation
#

setwd('D:/Github/DSA_CLICK_ANALYSIS')

v_c_file <- 'dataset/train.csv'

library(data.table)
library(fasttime)
library(lubridate)
library(caTools)

# Load dataset

# I was loading the columns as factors but it appears that loading as integers and transforming
# to factor with lapply got a better performance

gc()

train <- data.table::fread(file = v_c_file,
                           drop = c("attributed_time"),
                           colClasses = c(ip = "character", 
                                          app = "integer", 
                                          device = "device",
                                          os = "os",
                                          channel = "integer",
                                          click_time = "character", 
                                          is_attributed = "integer"))

remove(v_c_file)

# ip: ip address of click.
# app: app id for marketing.
# device: device type id of user mobile phone (e.g., iphone 6 plus, iphone 7, huawei mate 7, etc.)
# os: os version id of user mobile phone
# channel: channel id of mobile ad publisher
# click_time: timestamp of click (UTC)
# attributed_time: if user download the app for after clicking an ad, this is the time of the app download
# is_attributed: the target that is to be predicted, indicating the app was downloaded

# TRANSFORM
# Order by IP and Click Time
data.table::setorder(x = train, ip, click_time)

# Get click_order by IP
train[, click_order := frank(x = .I), by = ip]

# Categorize by click_order
train[click_order == 1, click_order_class := 1] 
train[click_order > 1 & click_order <= 10, click_order_class := 2] 
train[click_order > 10 & click_order <= 20, click_order_class := 3] 
train[click_order > 20 & click_order <= 30, click_order_class := 4] 
train[click_order > 30 & click_order <= 40, click_order_class := 5] 
train[click_order > 40 & click_order <= 50, click_order_class := 6] 
train[click_order > 50 & click_order <= 100, click_order_class := 7] 
train[click_order > 100, click_order_class := 8]

# Remove unused column
train[, click_order := NULL]

gc()

# Get last click from the same IP
# Get diff between last click time and click time
# Categorize last click timeand click time difference
train[, last_click_time := ifelse(ip == data.table::shift(ip), data.table::shift(click_time), NA)]
train[!is.na(last_click_time), last_click_time_diff := difftime(time1 = fastPOSIXct(click_time), time2 = fastPOSIXct(last_click_time), units = ("mins"))]
train[, last_click_time := NULL]

gc()

train[is.na(last_click_time_diff), last_click_time_class := 1]
train[last_click_time_diff < 1, last_click_time_class := 2]
train[last_click_time_diff >= 1 & last_click_time_diff < 60, last_click_time_class := 3]
train[last_click_time_diff >= 60 & last_click_time_diff < 1440, last_click_time_class := 4]
train[last_click_time_diff >= 1440 & last_click_time_diff < 43200, last_click_time_class := 5]
train[last_click_time_diff >= 43200 & last_click_time_diff < 518400, last_click_time_class := 6]
train[last_click_time_diff >= 518400, last_click_time_class := 7]

train[, last_click_time_diff := NULL]

gc()

# Get next click from the same IP
# Get diff between click time and next click time
# Categorize click time and last click time difference
train[, next_click_time := ifelse(ip == data.table::shift(ip, type = "lead"), data.table::shift(click_time, type = "lead"), NA)]
train[!is.na(next_click_time), next_click_time_diff := difftime(time1 = fastPOSIXct(next_click_time), time2 = fastPOSIXct(click_time), units = ("mins"))]
train[, next_click_time := NULL]

gc()


train[is.na(next_click_time_diff), next_click_time_class := 1]
train[next_click_time_diff < 1, next_click_time_class := 2]
train[next_click_time_diff >= 1 & next_click_time_diff < 60, next_click_time_class := 3]
train[next_click_time_diff >= 60 & next_click_time_diff < 1440, next_click_time_class := 4]
train[next_click_time_diff >= 1440 & next_click_time_diff < 43200, next_click_time_class := 5]
train[next_click_time_diff >= 43200 & next_click_time_diff < 518400, next_click_time_class := 6]
train[next_click_time_diff >= 518400, next_click_time_class := 7]
train[, next_click_time_diff := NULL]

gc()

train[, click_count := (.N), by = ip]

train[click_count == 1, ip_click_count := 1] 
train[click_count > 1 & click_count <= 10, ip_click_count := 2] 
train[click_count > 10 & click_count <= 20, ip_click_count := 3] 
train[click_count > 20 & click_count <= 30, ip_click_count := 4] 
train[click_count > 30 & click_count <= 40, ip_click_count := 5] 
train[click_count > 40 & click_count <= 50, ip_click_count := 6] 
train[click_count > 50 & click_count <= 100, ip_click_count := 7] 
train[click_count > 100, ip_click_count := 8]

train[, click_count := NULL]
train[, ip := NULL]

gc()

# Get the HOUR of click_time
train[, hour := as.factor(lubridate::hour(fastPOSIXct(click_time, tz = "GMT")))]

train[, click_time := NULL]

gc()

# CLEAR
# Verify missing values
# any(is.na(train))
# There are no missing values
# train <- na.omit(train)

# Remove duplicated lines
train <- unique(train)

train <- train[ , lapply(X = .SD, FUN = as.factor)]

gc()

fwrite(x = train, file = 'dataset/train_complete.csv')

# SPLIT
split = sample.split(Y = train[, is_attributed], SplitRatio = 0.7)
test <- subset(train, split == FALSE)
train <- subset(train, split == TRUE)
remove(split)

gc()

fwrite(x = train, file = 'dataset/train_split.csv')
fwrite(x = test, file = 'dataset/test_split.csv')

remove(test)
remove(train)