# Title: NLP HW 2, Part 2 Model construction
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
library(caret)
library(text2vec)
library(glmnet)


# Loading training data
txt_data <- read.csv("clean_txt_data.csv", header = TRUE)

# Loading validating data
score_raw_data <- read.csv("student_tm_case_score_data.csv", header = TRUE)
score_data <- read.csv("clean_score_data.csv", header = TRUE)


#################### Train-validation split #################### 

# Creating indexes for train dataset
set.seed(102)
idx <- createDataPartition(txt_data$label, p = 0.7, list = FALSE)

# Creating a train dataset
train <- as.data.frame(txt_data[idx,])

# Creating a validation dataset
valid <- as.data.frame(txt_data[-idx,])


# Checking proportions of labels in both datasets
round(as.numeric(table(train$label)[1]) / as.numeric(table(train$label)[2]), 2)
round(as.numeric(table(valid$label)[1]) / as.numeric(table(valid$label)[2]), 2)


#################### Data preparation #################### 

# Initial iterator to make a vocabulary
iter_maker_train <- itoken(train$text,
                           progressbar = TRUE)

# Making a vocabulary
text_vocab <- create_vocabulary(iter_maker_train, 
                                c(ngram_min = 1L, ngram_max = 2L), 
                                sep_ngram = "_")


# Making a pruned vocabulary to make DTM smaller
pruned_text_vocab <- prune_vocabulary(text_vocab,
                                      term_count_min = 3, 
                                      doc_proportion_max = 0.1,
                                      doc_proportion_min = 0.001)

# Quantity of terms in the full vocabulary
nrow(text_vocab)

# Quantity of terms in the pruned vocabulary
nrow(pruned_text_vocab)


# Using the pruned vocabulary to declare the DTM vectors 
vectorizer <- vocab_vectorizer(pruned_text_vocab)

# Taking the vocabulary lexicon and the pruned text function to make a DTM 
tweets_dtm <- create_dtm(iter_maker_train, vectorizer)


#################### Creating a model #################### 

# Training a model
model_fit <- cv.glmnet(tweets_dtm,
                       y = as.factor(train$label),
                       alpha = 0.9,
                       family = 'binomial',
                       type.measure = 'auc',
                       nfolds = 5,
                       intercept = FALSE)


# Subset to impacting terms
best_terms <- subset(as.matrix(coefficients(model_fit)), 
                     as.matrix(coefficients(model_fit)) != 0)

# Checking some terms which have an impact
head(best_terms, 10)

# Quantity of terms which have an impact
nrow(best_terms)


# Counting the probabilities for predictions on the train dataset
train_probs <- predict(model_fit,
                       tweets_dtm,
                       type = "response",
                       s = c("lambda.1se", "lambda.min"))

# Checking different thresholds and making predictions on the train dataset
train_preds <- as.numeric(sapply(train_probs, function(i) i > 0.5))


# Checking the Confusion Matrix
confusionMatrix(as.factor(train_preds),
                as.factor(train$label))$table

# Checking the Accuracy score
round(confusionMatrix(as.factor(train_preds),
                      as.factor(train$label))$overall[1], 3)
# 0.918 


# Checking the Sensitivity score
round(confusionMatrix(as.factor(train_preds),
                      as.factor(train$label))$byClass[1], 3)
# 0.991
## If we don't want to miss target tweets, the higher Sensitivity, the better


#################### Validating the Model on the score dataset #################### 

# Creating an iterator for the valid dataset
iter_maker_valid <- itoken(valid$text, 
                           progressbar = TRUE)

# Making a DTM for the valid dataset
val_dtm <- create_dtm(iter_maker_valid, vectorizer)

# Making predictions on the valid dataset
val_preds <- predict(model_fit,
                     val_dtm,
                     type = "class",
                     s = c("lambda.1se", "lambda.min"))


# Checking the Confusion Matrix
confusionMatrix(as.factor(val_preds),
                as.factor(valid$label))$table

# Checking the Accuracy score
round(confusionMatrix(as.factor(val_preds),
                      as.factor(valid$label))$overall[1], 3)
# 0.878


# Checking the Sensitivity score
round(confusionMatrix(as.factor(val_preds),
                      as.factor(valid$label))$byClass[1], 3)
# 0.963


#################### Making predictions on the scoring dataset #################### 

# Creating an iterator for the scoring dataset
iter_maker_score <- itoken(score_data$text, 
                           progressbar = TRUE)

# Making a DTM for the scoring dataset
score_dtm <- create_dtm(iter_maker_score, vectorizer)


# Counting the probabilities for predictions on the valid dataset
score_probs <- predict(model_fit,
                       score_dtm,
                       type = "response",
                       s = c("lambda.1se", "lambda.min"))

# Making predictions on the scoring dataset
score_preds <- as.numeric(sapply(score_probs, function(i) i > 0.5))

# Adding predictions and probabilities to the scoring dataset
result_1 <- cbind(score_raw_data, score_preds, score_probs)

# Renaming columns
names(result_1)[names(result_1) == "docID"] <- "doc_id"
names(result_1)[names(result_1) == "rawText"] <- "text"
names(result_1)[names(result_1) == "score_preds"] <- "label"
names(result_1)[names(result_1) == "1"] <- "probability"


# Writing predictions and probabilities into the file
write.csv(result_1, "Yushkin_TM_scores.csv", row.names = FALSE)


#####
## The following code shows different approach -
## creating a common vocabulary for training and scoring data.
## 
## However, in production, if we need to classify
## tweets right after they posted, such approach would require
## much more computational power because we can't know in advance
## which words and phrases will be used in the future tweets,
## so will have to create the new vocabulary each time
## classifying new tweet(-s).
## 
## Moreover, in this case the previous approach allows to identify
## 9 (around 4.8%) more target tweets. Perhaps this is because
## the score text data blurs some important terms.


#################### Creating a joint dataset #################### 

# Removing the target variable column from training data
txt_data_wo_labels <- txt_data 
txt_data_wo_labels$label <- NULL

# Combining training and scoring data
data <- rbind(txt_data_wo_labels, score_data)


#################### Data preparation #################### 

# Initial iterator to make a vocabulary
iter_maker_joint <- itoken(data$text,
                           progressbar = TRUE)

# Making a vocabulary
text_vocab <- create_vocabulary(iter_maker_joint, 
                                c(ngram_min = 1L, ngram_max = 2L), 
                                sep_ngram = "_")


# Prune vocabulary to make DTM smaller
pruned_text_vocab <- prune_vocabulary(text_vocab,
                                      term_count_min = 3, 
                                      doc_proportion_max = 0.1,
                                      doc_proportion_min = 0.001)

# Quantity of terms in the full vocabulary
nrow(text_vocab)

# Quantity of terms in the pruned vocabulary
nrow(pruned_text_vocab)


# Using the pruned vocabulary to declare the DTM vectors 
vectorizer <- vocab_vectorizer(pruned_text_vocab)

# Taking the vocabulary lexicon and the pruned text function to make a DTM 
tweets_dtm <- create_dtm(iter_maker_train, vectorizer)


#################### Train-validation split #################### 

# Creating indexes for train dataset
set.seed(102)
idx <- createDataPartition(txt_data$label, p = 0.7, list = FALSE)

# Creating a train dataset
train <- as.data.frame(txt_data[idx,])

# Creating a validation dataset
valid <- as.data.frame(txt_data[-idx,])


#################### Training a model #################### 

# Training a model
model_fit <- cv.glmnet(tweets_dtm,
                       y = as.factor(train$label),
                       alpha = 0.9,
                       family = 'binomial',
                       type.measure = 'auc',
                       nfolds = 5,
                       intercept = FALSE)


# Subset to impacting terms
best_terms <- subset(as.matrix(coefficients(model_fit)), 
                     as.matrix(coefficients(model_fit)) != 0)

# Checking some terms which have an impact
head(best_terms, 10)

# Quantity of terms which have an impact
nrow(best_terms)


# Counting the probabilities for predictions on the train dataset
train_probs <- predict(model_fit,
                       tweets_dtm,
                       type = "response",
                       s = c("lambda.1se", "lambda.min"))

# Checking different thresholds and making predictions on the train dataset
train_preds <- as.numeric(sapply(train_probs, function(i) i > 0.5))


# Checking the Confusion Matrix
confusionMatrix(as.factor(train_preds),
                as.factor(train$label))$table

# Checking the Accuracy score
round(confusionMatrix(as.factor(train_preds),
                      as.factor(train$label))$overall[1], 3)
# 0.904

# Checking the Sensitivity score
round(confusionMatrix(as.factor(train_preds),
                      as.factor(train$label))$byClass[1], 3)
# 0.991


#################### Validating the Model on the score dataset #################### 

# Creating an iterator for the valid dataset
iter_maker_valid <- itoken(valid$text, 
                           progressbar = TRUE)

# Making a DTM for the valid dataset
val_dtm <- create_dtm(iter_maker_valid, vectorizer)


# Counting the probabilities for predictions on the valid dataset
val_probs <- predict(model_fit,
                     val_dtm,
                     type = "response",
                     s = c("lambda.1se", "lambda.min"))

# Making predictions on the valid dataset
val_preds <- as.numeric(sapply(val_probs, function(i) i > 0.5))


# Checking the Confusion Matrix
confusionMatrix(as.factor(val_preds),
                as.factor(valid$label))$table

# Checking the Accuracy score
round(confusionMatrix(as.factor(val_preds),
                      as.factor(valid$label))$overall[1], 3)
# 0.88


# Checking the Sensitivity score
round(confusionMatrix(as.factor(val_preds),
                      as.factor(valid$label))$byClass[1], 3)
# 0.974


#################### Making predictions on the scoring dataset #################### 

# Creating an iterator for the scoring dataset
iter_maker_score <- itoken(score_data$text, 
                           progressbar = TRUE)

# Making a DTM for the scoring dataset
score_dtm <- create_dtm(iter_maker_score, vectorizer)


# Counting the probabilities for predictions on the valid dataset
score_probs <- predict(model_fit,
                       score_dtm,
                       type = "response",
                       s = c("lambda.1se", "lambda.min"))

# Making predictions on the scoring dataset
score_preds <- as.numeric(sapply(score_probs, function(i) i > 0.5))

# Adding predictions and probabilities to the scoring dataset
result_2 <- cbind(score_raw_data, score_preds, score_probs)


# Joining results 
results <- cbind(score_raw_data, 
                 label_1 = result_1$label, 
                 label_2 = result_2$score_preds, 
                 probability_1 = result_1$probability,
                 probability_2 = result_2[,1])

# Comparing results
(diff <- results %>% 
  filter(label_1 != label_2) %>% 
  select(rawText, label_1, label_2))

sum(results$label_1)
sum(results$label_2)

## The score is 196:187 in favor of the first approach 
## (not 198:187 because 2 out of 13 probably are false positive)


# End