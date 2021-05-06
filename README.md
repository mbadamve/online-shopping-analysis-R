# Analysis of e-commerce shopping patterns

## Project Description:

The goal of this project is to determine the possibility of an user making a purchase at an online shopping website by using the features from the Google Analytics data. I have experimented with many of the models and compared to give us an idea of which model is good and also the cost of achieving that performance.

## Dataset:

The dataset is downloaded from Kaggle and it is available [here](https://www.kaggle.com/roshansharma/online-shoppers-intention)

## Dependencies:

1. RStudio with R installed first. More information is found [here](https://cran.r-project.org/)
2. tidyverse
3. e1071
4. caret
5. kernlab
6. arulesViz
7. rpart
8. rpart.plot
9. randomForest
10. data.table
11. mlr
12. xgboost
13. parallel
14. parallelMap
15. arulesCBA

All the above are R packages, they can be installed by using the command install.packages('<package_name>') inside RStudio. For example, to install tidyverse, the command would be install.package('tidyverse')

More information aboout these packages can be found [here](https://www.rdocumentation.org/)

## Files in the repo:

There is one code file (R) and data(.csv) file in this repo. Download both of them in a single directory and run the file to see the model results and predictions.

## Running the file:

It is easy to run the file in RStudio, please make sure all the packages are installed first before running the file.

## Results:

XGBoost performs better and it comes at a higher training time. Arules packages generated very important rules similar to decision trees with which we can interpret the models for understanding the user behavior.

