library(tidyverse)
library(caret)
library(lubridate)

# Utility functions
collapse.factor <- function(x) {
  x <- fct_lump_n(x, n=5)
  x <- fct_explicit_na(x, na_level = "Other")
  return(x)
}

# Function to plot missing value percentages in each column in a dataframe
get.missing.perc <- function(df) {
  na.perc <- colMeans(is.na(df)) * 100
  data.frame(perc=na.perc[na.perc > 0], name=names(na.perc[na.perc > 0])) %>%
    arrange(perc) %>%
    mutate(name=factor(name, levels = name)) %>%
    ggplot(aes(y=name, x=perc, fill=perc<50)) +
    geom_bar(stat = "identity") +
    labs(title="Percentage of missing values", y="Variable", x="Percentage")
}


# Importing the test and train Datasets
train.data <- read.csv("data/Train.csv", na.strings = c("", "NA"))
test.data <- read.csv("data/Test.csv", na.strings = c("", "NA"))

# Getting the Unique customer IDs from both Train and Test data
custIds.train <- unique(train.data$custId)
custIds.test <- unique(test.data$custId)

# Checking if there are any duplicate customers in Test data and Train data
which(custIds.test %in% custIds.train) # Shows 0 common customer IDs

# Getting the Max and Min data seen in the Train dataset
maxdate = max(ymd(train.data$date))
mindate = min(ymd(train.data$date))

# Combining both Train and Test to do some Pre-Processing on some Important columns
# Then grouping the data using custId
full.data <- train.data %>% dplyr::select(-revenue) %>%
  rbind(test.data) %>%
  mutate(date=ymd(date), 
         across(.cols=where(is.character), .fns=collapse.factor)) %>%
  group_by(custId) %>%
  summarise(channelGrouping = max(ifelse(is.na(channelGrouping) == TRUE, -9999, channelGrouping)),
            first_ses_from_the_period_start = min(date) - mindate,
            last_ses_from_the_period_end = maxdate - max(date),
            interval_dates = max(date) - min(date),
            unique_date_num = length(unique(date)),
            maxVisitNum = max(visitNumber, na.rm = TRUE),
            browser = as.factor(first(ifelse(is.na(browser), -9999, browser))),
            operatingSystem = as.factor(first(ifelse(is.na(operatingSystem) == TRUE, -9999, operatingSystem))),
            deviceCategory = as.factor(first(ifelse(is.na(deviceCategory) == TRUE, -9999, deviceCategory))),
            subContinent = as.factor(first(ifelse(is.na(subContinent) == TRUE, -9999, subContinent))),
            country = as.factor(first(ifelse(is.na(country) == TRUE, -9999, country))),
            region = as.factor(first(ifelse(is.na(region) == TRUE, -9999, region))),
            networkDomain = as.factor(first(ifelse(is.na(networkDomain) == TRUE, -9999, networkDomain))),
            source = as.factor(first(ifelse(is.na(source) == TRUE, -9999, source))),
            medium = as.factor(first(ifelse(is.na(medium) == TRUE, -9999, medium))),
            isVideoAd_mean = mean(ifelse(is.na(adwordsClickInfo.isVideoAd) == TRUE, 0, 1)),
            isMobile = mean(ifelse(isMobile == TRUE, 1 , 0)),
            isTrueDirect = mean(ifelse(is.na(isTrueDirect) == TRUE, 0, 1)),
            bounce_sessions = sum(ifelse(is.na(bounces) == TRUE, 0, 1)),
            pageviews_sum = sum(pageviews, na.rm = TRUE),
            pageviews_mean = mean(ifelse(is.na(pageviews), 0, pageviews)),
            pageviews_min = min(ifelse(is.na(pageviews), 0, pageviews)),
            pageviews_max = max(ifelse(is.na(pageviews), 0, pageviews)),
            pageviews_median = median(ifelse(is.na(pageviews), 0, pageviews)),
            session_cnt = NROW(visitStartTime)) %>%
  mutate(pctBounce = bounce_sessions/session_cnt, 
         allBounce = as.numeric(bounce_sessions >= session_cnt),
         bounceViewRatio = bounce_sessions/(pageviews_sum+1))

# Splitting the full.data into train and test data again
train.data <- train.data %>%
  group_by(custId) %>%
  summarise(revenue=sum(revenue)) %>%
  mutate(revenue=log(revenue+1)) %>%
  inner_join(full.data %>% filter(custId %in% custIds.train), by="custId")

test.data <- full.data %>% filter(custId %in% custIds.test)

rm(full.data)
  

# Modelling
# Classifying if a customer will spend money or not using LDA classifier
fitControl <- trainControl(method = "cv")

fit.lda <- train(as.factor(revenue>0) ~ channelGrouping + 
                   allBounce  + region + 
                   browser + operatingSystem + country  + 
                   bounce_sessions + pageviews_sum, 
                 data=train.data, trControl=fitControl,
                 method="lda")
fit.lda
# Linear Discriminant Analysis 
# 
# 47249 samples
# 8 predictor
# 2 classes: 'FALSE', 'TRUE' 
# 
# No pre-processing
# Resampling: Cross-Validated (10 fold) 
# Summary of sample sizes: 42525, 42524, 42523, 42523, 42524, 42525, ... 
# Resampling results:
#   
#   Accuracy   Kappa    
# 0.9452899  0.6641091

table(train.data$revenue>0, predict(fit.lda, train.data))
#       FALSE  TRUE
# FALSE 41757   408
# TRUE   2178  2906

# Adding revenue>0 probability to training data
train.data$posterior <- predict(fit.lda, train.data, type="prob")[, "TRUE"]


# Fitting MARS model to predict the amount of LOG revenue a customer will generate
fitControl <- trainControl(method = "cv")
tuneGrid <- expand.grid(nprune=seq(20, 30, by=1), degree= 1:3)

fit.mars <- train(revenue ~ bounceViewRatio + channelGrouping + posterior +
                   allBounce + as.numeric(first_ses_from_the_period_start) + 
                   as.numeric(last_ses_from_the_period_end) + pctBounce + region + 
                   source + browser + operatingSystem + country + networkDomain + 
                   log(bounce_sessions + 1) + bounce_sessions + pageviews_sum + 
                   log(pageviews_sum + 1) + log(pageviews_mean + 1) + pageviews_max + 
                   pageviews_median + log(session_cnt), 
                 data=train.data, trControl=fitControl, tuneGrid=tuneGrid,
                 method="earth")
fit.mars
# Multivariate Adaptive Regression Spline 
# 
# 47249 samples
# 19 predictor
# 
# No pre-processing
# Resampling: Cross-Validated (10 fold) 
# Summary of sample sizes: 42524, 42524, 42524, 42524, 42524, 42525, ... 
# Resampling results across tuning parameters:
#   
#   degree  nprune  RMSE       Rsquared   MAE      
# 1       20      0.7236465  0.6944663  0.2931321
# 1       21      0.7236465  0.6944663  0.2931321
# 1       22      0.7236465  0.6944663  0.2931321
# 1       23      0.7236465  0.6944663  0.2931321
# 1       24      0.7236465  0.6944663  0.2931321
# 1       25      0.7236465  0.6944663  0.2931321
# 1       26      0.7236465  0.6944663  0.2931321
# 1       27      0.7236465  0.6944663  0.2931321
# 1       28      0.7236465  0.6944663  0.2931321
# 1       29      0.7236465  0.6944663  0.2931321
# 1       30      0.7236465  0.6944663  0.2931321
# 2       20      0.6945988  0.7185354  0.2398561
# 2       21      0.6947592  0.7184175  0.2398997
# 2       22      0.6945226  0.7185992  0.2401436
# 2       23      0.6943829  0.7187026  0.2399386
# 2       24      0.6943724  0.7187125  0.2400128
# 2       25      0.6942832  0.7187846  0.2400029
# 2       26      0.6942832  0.7187846  0.2400029
# 2       27      0.6942832  0.7187846  0.2400029
# 2       28      0.6942832  0.7187846  0.2400029
# 2       29      0.6942832  0.7187846  0.2400029
# 2       30      0.6942832  0.7187846  0.2400029
# 3       20      0.6896715  0.7225496  0.2325857
# 3       21      0.6896411  0.7225770  0.2324190
# 3       22      0.6896647  0.7225595  0.2323578
# 3       23      0.6896298  0.7225877  0.2323725
# 3       24      0.6896298  0.7225877  0.2323725
# 3       25      0.6896298  0.7225877  0.2323725
# 3       26      0.6896298  0.7225877  0.2323725
# 3       27      0.6896298  0.7225877  0.2323725
# 3       28      0.6896298  0.7225877  0.2323725
# 3       29      0.6896298  0.7225877  0.2323725
# 3       30      0.6896298  0.7225877  0.2323725
# 
# RMSE was used to select the optimal model using the smallest value.
# The final values used for the model were nprune = 23 and degree = 3.

plot(fit.mars)

# Finally, preparing the test data
# Getting the LDA probabilities
test.data$posterior <- predict(fit.lda, test.data, type="prob")[, "TRUE"]

# Predicting test data using MARS
preds.mars <- predict(fit.mars, newdata=test.data)
preds.mars[preds.mars < 0] <- 0

tibble(custId = test.data$custId, predRevenue = preds.mars[,"y"]) %>%
  write.csv("Submission.csv", row.names = F)






  



