# Title: NLP HW 2, Part 1 Exploration
# NAME: Alexey Yushkin
# Date: Mar 11 2021
# Note: Some chunks of the code are commented because running takes much time.
#       Instead, the script uses prepared data saved in csv files.


# Setting working directory
setwd("~/Downloads/TEXT_ANALYTICS_AND_NLP/Hult_NLP/personal/HW2")

# Do not re-encode strings, turn off scientific notation
options(stringsAsFactors = FALSE, scipen = 999)


#################### Importing libraries and data #################### 

# Uploading libraries
library(tm)
library(dplyr)
library(rtweet)
library(mgsub)


# Loading training data
txt_raw_data <- read.csv("student_tm_case_training_data.csv", header = TRUE)

# Loading validating data
score_raw_data <- read.csv("student_tm_case_score_data.csv", header = TRUE)


#################### Defining constants and custom functions #################### 

# Vector of stop words
stops <- stopwords("SMART")


# Function for changing case to lower
tryTolower <- function(x){
  # return NA when there is an error
  y = NA
  # tryCatch error
  try_error = tryCatch(tolower(x), error = function(e) e)
  # if not an error
  if (!inherits(try_error, "error"))
    y = tolower(x)
  return(y)
}

# Cleaning function
tweets_clean <- function(text, stopwords){
  # Converting data to ASCII
  text <- iconv(text, "latin1", "ASCII", sub = "")
  # Replacing dashes with spaces
  text <- mgsub(text, "-", " ")
  # Replacing emojis with text
  text <- mgsub(text, emojis$code, paste0(" ", emojis$description," "))
  # Removing URLs
  text <- qdapRegex::rm_url(text)
  # Replacing abbreviation with text
  text <- qdap::replace_abbreviation(text) 
  # Removing ampersands
  text <- gsub("&amp", "", text)
  # Removing retweet headers
  text <- gsub("(RT|via)((?:\\b\\W*@\\w+)+)", "", text)
  # Removing usernames
  text <- mgsub(text, "@[a-z,A-Z]*", "")
  # Making lower case
  text <- tryTolower(text)
  # Removing punctuation
  text <- removePunctuation(text)
  # Removing numbers
  text <- removeNumbers(text)
  # Removing stop words
  text <- removeWords(text, stopwords)
  # Removing unknown emojis
  text <- mgsub(text, "�", "")
  # Removing extra whitespace
  text <- stripWhitespace(text)
  # Removing leading and trailing spaces
  text <- trimws(text, which = "both")
  return(text)
}


#################### EDA I #################### 

# Checking dimensions, column names, types of data and the first records
glimpse(txt_raw_data)
glimpse(score_raw_data)


# Checking the basic statistics
summary(txt_raw_data)
summary(score_raw_data)


# Checking the first records
head(txt_raw_data, 1)
head(score_raw_data, 1)


#################### Cleaning the text data #################### 

# # Cleaning up training data
# txt_data <- txt_raw_data
# txt_data$rawText <- tweets_clean(txt_data$rawText, stops)
# 
# # Renaming the rawText column
# names(txt_data)[names(txt_data) == "rawText"] <- "text"
# 
# # # Removing records with empty text values
# # txt_data <- txt_data %>%
# #   filter(text != "")
# 
# # Writing the result into a file
# write.csv(txt_data, "clean_txt_data.csv", row.names = FALSE)

# Reading data from the file
txt_data <- read.csv("clean_txt_data.csv", header = TRUE)


# # Cleaning up scoring data
# score_data <- score_raw_data
# score_data$rawText <- tweets_clean(score_data$rawText, stops)
# 
# # Renaming the rawText column
# names(score_data)[names(score_data) == "rawText"] <- "text"
# 
# # # Removing records with empty text values
# # score_data <- score_data %>%
# #   filter(text != "")
# 
# # Writing the result into a file
# write.csv(score_data, "clean_score_data.csv", row.names = FALSE)

# Reading data from the file
score_data <- read.csv("clean_score_data.csv", header = TRUE)


#################### EDA II #################### 

# Word frequencies 
# - for training data
freq_train <- qdap::freq_terms(txt_data$text, 
                               stopwords = c(stops, "im", "dont", "youre", "day", "month", "time"))
plot(freq_train)

# - for scoring data
freq_score <- qdap::freq_terms(score_data$text, 
                               stopwords = c(stops, "im", "dont", "youre", "day", "month", "time"))
plot(freq_score)

### The most frequent words are almost the same, but not all of them, and 
### relative frequencies are different. Probably this means that the model trained 
### on the training dataset might not work very well on the scoring data. 


# Polarity 
# - for training data by groups
txt_data$label <- as.factor(txt_data$label)
pol_train <- qdap::polarity(txt_data$text, txt_data$label)
barplot(pol_train$group$ave.polarity, 
        names.arg = pol_train$group$label, 
        main = "Polarity by groups",
        adj = 0.5) 

# # - for all training data
# (pol_train <- qdap::polarity(txt_data$text))
# #   all total.sentences total.words ave.polarity sd.polarity stan.mean.polarity
# # 1 all            2000       14911       -0.088       0.443             -0.199

# # - for all scoring data 
# (pol_score <- qdap::polarity(score_data$text))
# #   all total.sentences total.words ave.polarity sd.polarity stan.mean.polarity
# # 1 all            1000        7396       -0.078       0.435             -0.179

## Both datasets have close slightly negative average polarity values


# End