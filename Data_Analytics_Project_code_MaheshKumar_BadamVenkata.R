############### Prediction of possibility of revenue generation in Online Shopping ###########
# Importing all the packages required for running this code.

install.packages("randomForest")
install.packages("xgboost")
install.packages("mlr")
install.packages("arulesCBA")
library(tidyverse)
library(e1071)
library(caret)
library(kernlab)
library(arulesViz)
library(rpart)
library(rpart.plot)
library(randomForest)
library(data.table)
library(mlr)
library(xgboost)
library(parallel)
library(parallelMap)
parallelStartSocket(cpus = detectCores())
library(arulesCBA)

# Loading the dataset from local library
#getwd()
#setwd('/Users/mahi/Downloads')
df <- read_csv('online_shoppers_intention.csv')
#df <- read_csv(file.choose())

#Pre-processing
# The dataset has month of June listed as June, replacing it with Jun would ease 
# the process of factorization of the column
table(df$Month)
df$Month[df$Month=='June'] <- 'Jun'

# converting all the columns to factors with levels
df_1 <- df %>% mutate(
  Month = factor(Month, levels = month.abb),
  VisitorType = factor(VisitorType),
  Weekend = factor(Weekend),
  Revenue = factor(Revenue),
  SpecialDay = factor(SpecialDay),
  OperatingSystems = factor(OperatingSystems),
  Browser = factor(Browser),
  Region = factor(Region),
  TrafficType = factor(TrafficType)
)

# Summary of data
summary(df_1)
# Type of data in each column
str(df_1)
# Proving no NAs in the dataframe
which(is.na(df_1))

# removing outliers using Interquartile range rule
# the function finds the up and down values of the column and replaces outliers
# with appropriate values and returns the column
outlier_mng <- function(x){
  Q <- quantile(x, probs = c(.25, .75), na.rm = FALSE)
  iqr <- IQR(x)
  up <- unname(Q[2]) + 1.5*iqr
  down <- unname(Q[1]) - 1.5*iqr
  x[x > up] <- up
  x[x < down] <- down
  return(x)
}

# the outlier handling is done for "Administrative", "Administrative_Duration",
# "ProductRelated", "ProductRelated_Duration", "BounceRates", "ExitRates" 

for (i in c(names(df_1)[1:2],names(df_1)[5:8])){
  df_1[, c(i)] <- lapply(select(df, i), outlier_mng)
}


#Data Visualization
# percentage ratio of shoppers with revenue generation from them
c <- ggplot(data = df)
c + geom_bar(aes(Revenue, 
                 ..prop.., 
                 group=1), 
             stat = 'count')+
  scale_y_continuous(labels = scales::percent_format())+
  ggtitle("Proportion of users in Revenue generation(T/F)")

# Plot of monthly visitors and if revenue generated from them
c + geom_bar(mapping = aes(
  x=Month,fill = Revenue
), position = 'dodge')+
  ggtitle("Monthly visitors and if Revenue generated")

# Plot of PageValues versus Revenue
c + geom_jitter(aes(PageValues, Revenue))+
  ggtitle("How PageValues related to Revenue from shopper")

# Plot of ExitRates versus Revenue
c + geom_jitter(aes(ExitRates, Revenue))+
  ggtitle("How ExitRates related to Revenue from shopper")

# Defining function for Accuracy to find from confusion matrix
Accuracy <- function(x){
  100-(100*(x[3]+x[2])/sum(x))
}
# Defining function for Precision to find from confusion matrix

Precision <- function(x){
  100-(100*x[1]/(x[1]+x[2]))
}
# Defining function for Recall to find from confusion matrix

Recall <- function(x){
  100-(100*x[1]/(x[1]+x[3]))
}

# Model 1 - Naive Bayes

set.seed(10)

# Splitting into training and testing data
split_values <- sample(nrow(df), floor(0.7*nrow(df)))
train_data <- df_1[split_values,]
test_data <- df_1[-split_values,]

# showing the proporting of true and false in revenue which is same like main dataset
table(df$Revenue)/nrow(df)

# building the model
model_nb <- naiveBayes(train_data, 
                       train_data$Revenue, 
                       laplace = 1)

# making the predictions
predicted_nb <- predict(model_nb, 
                        test_data)

# building the confusion matrix
conf_matrix_nb <- table(predicted_nb, 
                        test_data$Revenue)

# finding accuracy, precision, recall, F-measure for Naive Bayes
Accuracy_nb <- Accuracy(table(predicted_nb, test_data$Revenue))

Precision_nb <- Precision(conf_matrix_nb)

Recall_nb <- Recall(conf_matrix_nb)

F_measure_nb <- (2*Precision_nb*Recall_nb)/(Precision_nb + Recall_nb)

# Model 2 - KSVM

# scaling the data using minmax scalar
prePr <- preProcess(df[,1:17], 
                    method='scale')

# generating training and testing data
predict(prePr, train_data)
predict(prePr, test_data)

# building the model
model_svm <- ksvm(Revenue~., 
                  data = train_data, 
                  kernel='rbfdot')

# making predictions
predicted_svm <- predict(model_svm, 
                         test_data)

# builsing the confusion matrix
conf_matrix_svm <- table(predicted_svm, 
                         test_data$Revenue)

# finding the accuracy
Accuracy(conf_matrix_svm)


#Model 3 : Association Rules - Apriori

# Preprocessing for apriori

str(df_1)
summary(df_1)
# supervised discretization for the 9 continuous columns to create bins automatically
# that have high predictability

for (j in names(df_1)[0:9]){
  df_1[,c(j)] <- discretizeDF.supervised(Revenue ~., 
                                         df_1[, c(j, "Revenue")],
                                         method = 'mdlp')[,1]
}

# converting the dataset into transactions
tData <- as(df_1, 'transactions')

# dataframe which shows the values in the transactions data
x <- data.frame(sort(table(unlist(LIST(tData))), decreasing=TRUE))

# plot showing the high to low frequent bin in the data
ggplot(data=x, aes(x=factor(Var1), y=Freq)) + 
  geom_col() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  ylab('Frequency') +
  xlab('Items')

# generating rules for Revenue=True with min support of 0.01 and confidence of 0.9

rules <- apriori(tData, parameter = list(supp = 0.01,
                                         minlen = 3,
                                         maxlen = 5,
                                         conf = 0.9))
# converting rules to a dataframe
rules_df <- DATAFRAME(rules, setStart = '', setEnd = '', separate = TRUE)
head(rules_df)
# summary of rules
summary(rules)
# rules which we are interested in 
imp_rules <- subset(rules, subset = rhs %pin% "Revenue=TRUE")
imp_rules
# inspecting the important rules
inspect(sort(imp_rules, by = 'lift'))
imp_rules_df <- DATAFRAME(sort(imp_rules, by = 'lift'), setStart = '', setEnd = '', separate = TRUE)
view(head(imp_rules_df))

# visualising the important rules with features
plot(head(imp_rules), method="graph", control=list(type="items"))
plot(head(imp_rules), method="paracoord", control=list(reorder=TRUE))


# generating rules for Revenue=False with min support of 0.5 and confidence of 0.9
rules <- apriori(tData, parameter = list(supp = 0.5,
                                         minlen = 3,
                                         maxlen = 5,
                                         conf = 0.9))
# converting rules to a dataframe
rules_df <- DATAFRAME(rules, setStart = '', setEnd = '', separate = TRUE)
head(rules_df)
# summary of rules
summary(rules)
# rules which we are interested in 
imp_rules <- subset(rules, subset = rhs %pin% "Revenue")
imp_rules
# inspecting the important rules
inspect(sort(imp_rules, by = 'lift'))
imp_rules_df <- DATAFRAME(sort(imp_rules, by = 'lift'), setStart = '', setEnd = '', separate = TRUE)
view(head(imp_rules_df))
# visualising the important rules with features
plot(head(imp_rules), method="graph", control=list(type="items"))
plot(head(imp_rules), method="paracoord", control=list(reorder=TRUE))

# Model 4:
# Decision Trees:
# creating the decision tree
tree_1 <- rpart(Revenue ~ .,
                data = df_1, 
                control = rpart.control(cp = 0.0001))
# finding the one with min xerror
min_xerror <- tree_1$cptable[,"xerror"] %>% 
  which.min()
# finding the CP wth min xerror
bestcp <- tree_1$cptable[min_xerror,"CP"]
# creating the pruned tree
tree.pruned <- prune(tree_1, cp = bestcp)
# visualizing the pruned tree
prp(tree.pruned, 
    faclen = 0, 
    cex = 0.8, 
    extra = 1)  


# Model 5:

# Random Forest:

# building the random forest model
rf_model <- randomForest(Revenue ~ ., train_data, 
                      ntree = 500, mtry=4)
plot(rf_model)

rf_model
# making predictions
predictions <- predict(rf_model, test_data)
# confusion matrix
conf_matrix_rf <- table(test_data$Revenue, predictions)

# Accuracy for the model
accuracy_RF <- 100*mean(test_data$Revenue == predictions)
#precision
Precision_rf <- Precision(conf_matrix_rf)
#recall
Recall_rf <- Recall(conf_matrix_rf)
#F-measure
F_measure_rf <- (2*Precision_rf*Recall_rf)/(Precision_rf + Recall_rf)
#Feature importances
sort(importance(rf_model), decreasing = T)
varImpPlot(rf_model)


# Model 6:
# XGBoost:
# data for training the XGBoost model
train_xg_labels <- train_data$Revenue
train_xg <- model.matrix(~.+0, data = train_data[, 1:17,with = F])
# data for testing the XGBoost model
test_xg_labels <- test_data$Revenue
test_xg <- model.matrix(~.+0, data = test_data[, 1:17,with = F])
# numeric labels of the data
train_xg_labels <- as.numeric(train_xg_labels)-1
test_xg_labels <- as.numeric(test_xg_labels)-1
#creating the matrix to be loaded for XGBoost model
dtrain <- xgb.DMatrix(data = train_xg, label = train_xg_labels)
dtest <- xgb.DMatrix(data = test_xg, label=test_xg_labels)
# cross validation
params <- list(booster = 'gbtree', objective = 'binary:logistic',
               eta = 0.3, gamma =0 , max_depth=6, min_child_weight=1,
               subsample=1, colsample_bytree=1)
xgbcv <- xgb.cv(params = params, data = dtrain, nrounds = 100,
                nfold = 5, showsd = T, stratified = T, print_every_n = 10,
                early_stopping_rounds = 20, maximize = F)
# min interations to be processed
min(xgbcv$best_iteration)
#training the model
xgb1 <- xgb.train(params = params, data = dtrain,
                  nrounds = 17, watchlist = list(val=dtest,
                                                 train=dtrain),
                  print_every_n = 10, early_stopping_rounds = 10,
                  maximize = F, eval_metric = "error")
#predicting using the model
xgb_pred <- predict(xgb1, dtest)
xgb_pred <- ifelse(xgb_pred > 0.5,1,0)
xgb_pred
# confusion matrix
conf_matrix_xg <- table(xgb_pred, test_xg_labels)
# accuracy
accuracy_xg <- mean(xgb_pred == test_xg_labels)*100
#precision
Precision_xg <- Precision(conf_matrix_xg)
#recall
Recall_xg <- Recall(conf_matrix_xg)
# F-measure
F_measure_xg <- (2*Precision_xg*Recall_xg)/(Precision_xg + Recall_xg)
# Feature importances from XGBoost
mat <- xgb.importance(feature_names = colnames(train_xg),
                      model = xgb1)
xgb.plot.importance(importance_matrix = mat[1:20])


# Model 7:
# Clustering:
# data processing for clustering
df_1 <- df %>% mutate(
  Month = factor(Month, levels = month.abb),
  VisitorType = factor(VisitorType),
  Weekend = factor(Weekend),
  Revenue = factor(Revenue),
  SpecialDay = factor(SpecialDay),
  OperatingSystems = factor(OperatingSystems),
  Browser = factor(Browser),
  Region = factor(Region),
  TrafficType = factor(TrafficType)
)
df_2 <- df_1 %>% mutate(
  Month = as.numeric(Month),
  VisitorType = as.numeric(factor(VisitorType)),
  Weekend = as.numeric(factor(Weekend)),
  Revenue = as.numeric(factor(Revenue)),
  SpecialDay = as.numeric(factor(SpecialDay)),
  OperatingSystems = as.numeric(factor(OperatingSystems)),
  Browser = as.numeric(factor(Browser)),
  Region = as.numeric(factor(Region)),
  TrafficType = as.numeric(factor(TrafficType))
)
for (i in c(names(df_1)[1:2],names(df_1)[5:8])){
  df_1[, c(i)] <- lapply(select(df, i), outlier_mng)
}
str(df_2)
summary(df_2)
# normalizing the data
which(is.na(df_2$Month))
normalize(df_2$Month)
normalize <- function(x){
  return ((x-min(x))/(max(x)-min(x)))
}
scaled_wd <- as.data.frame(lapply(df_2[1:17], normalize))
#no NA values after scaling
which(is.na(scaled_wd))
#training and testing data
train_data <- df_2[split_values,]
test_data <- df_2[-split_values,]
#heirarchial clustering based on euclidean distance
d<- dist(scaled_wd, method = 'euclidean')
h_clust <- hclust(d, method = "ward.D2")
plot(h_clust,labels = df_2$Revenue)
rect.hclust(h_clust, k=4)
# grouped data
groups <- cutree(h_clust, k=4)
groups

# Model 8: 
# PCA
# generating principal components
pcmp <- princomp(scaled_wd, cor = T, covmat = NULL, scores = T)

pred_pc <- predict(pcmp, newdata=scaled_wd)[,1:2]
#clustering the observations based on groups generated above
comp_dt <- cbind(as.data.table(pred_pc),
                               cluster = as.factor(groups),
                               labels=df_2$Revenue)
#viewing the principal components and the clusters they are in
ggplot(comp_dt, aes(Comp.1, Comp.2))+
  geom_point(aes(color= cluster), size=3, alpha = 0.5)

# Model 9:
# KMeans
# clustering the PCA using KMeans
k_clust <- kmeans(scaled_wd, 
                  centers = 4, 
                  iter.max = 100)
# viewing the clusters
ggplot(comp_dt, aes(Comp.1, Comp.2))+
  geom_point(aes(color = as.factor(k_clust$cluster)),
             size = 3, alpha = 1/2)





