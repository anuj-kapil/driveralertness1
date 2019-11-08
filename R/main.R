##################################################################################
#
# Course:       32513 - Advanced Data Analytics Algorithms
# Week:
# data source:  Kaggle
# dataset:      Stay Alert! Ford Challenge
# Filters:
#
# Comments:
# From:
#
##################################################################################
library(caret)
library(mlbench)
library(fscaret)
library(dplyr)
library(psych)
library(corrplot)
library(xgboost)
library(ROCR)
library(pROC)
library(gridExtra)
library(gbm)
library(e1071)


#' Main Analysis
#'
#' @return nothing
#' @export
#'
#' @examples
#' nothing
main_analysis <- function(){
#######################################################################
#                                                                     #
#                          1 - LOADING DATA                           #
#                                                                     #
#######################################################################

##### Reading the file #####
data <- read.csv("Data/fordTrain.csv",
                 header=TRUE, stringsAsFactors=FALSE, na.strings = c("NA", ""),
                 strip.white = TRUE, blank.lines.skip=TRUE, skip=0)

validatedata <- read.csv("Data/fordTest.csv",
                         header=TRUE, stringsAsFactors=FALSE, na.strings = c("NA", ""),
                         strip.white = TRUE, blank.lines.skip=TRUE, skip=0)

summary(data)
summary(validatedata)

#######################################################################
#                                                                     #
#                        2 - DATA UNDERSTANDING                       #
#                                                                     #
#######################################################################

#To ensure steps are repeatable
set.seed(131)

#No of records
nrow(data)

#No of attributes
ncol(data)

#No missing values
summary(data)

#Formatting Dataset
dput(names(data))
data<-data[c("TrialID", "ObsNum", "P1", "P2", "P3", "P4", "P5",
             "P6", "P7", "P8", "E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8",
             "E9", "E10", "E11", "V1", "V2", "V3", "V4", "V5", "V6", "V7",
             "V8", "V9", "V10", "V11", "IsAlert")]

#Descriptive Statistics
psych::describe(data, fast = FALSE )


##### Printing the distribution of Result #####
counts <- table(data$IsAlert)
countsframe<-as.data.frame(counts)
png(height=500, width=700, pointsize=20, file="barplot_IsAlert.png")
ggplot(countsframe, aes(x = Var1, y = Freq)) +
  geom_bar(stat = "identity", fill = '#000099') +
  geom_text(aes(label = sprintf("%.2f%%", Freq/sum(Freq) * 100))  ,
            vjust = -.5)+
  scale_size_area() +
  ggtitle("Alertness Distribution") + xlab("IsAlert") + ylab("Frequency")
dev.off()

##### Histogram and Box Plots #####

variables <- c("P1", "P2", "P3", "P4", "P5",
               "P6", "P7", "P8", "E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8",
               "E9", "E10", "E11", "V1", "V2", "V3", "V4", "V5", "V6", "V7",
               "V8", "V9", "V10", "V11")

layout_matrix <- matrix(c(1:4), nrow = 2, ncol = 2, byrow = T)
layout(mat = layout_matrix)
lapply(variables, hist_box_plots, dat=data)


##### Overlaid Histograms #####

data_alert<-data[data$IsAlert == '1',]
data_not_alert<-data[data$IsAlert == '0',]

variables <- c("P5", "P6", "E3", "E7", "E8", "E11", "V4", "V11")

layout_matrix <- matrix(c(1:4), nrow = 2, ncol = 2, byrow = T)
layout(mat = layout_matrix)
lapply(variables, stacked_bar_plots, dat_a=data_alert, dat_b=data_not_alert)

########### Repeated Measure Analysis ################
hist_data <- data %>% count(TrialID)
dev.off()
hist(hist_data$n, main = "Histogram of repeated measures for each driver",
     xlab="Number of Repeated Measures by Driver", ylab="Frequency", col = "blue"
     , breaks=seq(1160,1220,by=10))

############ Correlation Analysis #####################
cor_data_full <- data[,!(names(data) %in% c("IsAlert","TrialID","ObsNum","P8","V7","V9"))]

##### Computing the correlation matrix ######
cor_mat_full <- cor(cor_data_full, use="complete.obs")

##### Computing the correlation matrix ######
dev.off()
corrplot.mixed(cor_mat_full, lower="number", upper="circle")

#######################################################################
#                                                                     #
#                         3 - DATA PREPARATION                        #
#                                                                     #
#######################################################################

# Convert any potential factors in the data through heuristic.
# If number of unique values in dataset is less than specified threshold
# then treat as categorical data
auto_convert_factors <- function(data, cat_threshold=10, cols_ignore=list()) {

  for (col in names(data)) {
    if (!is.factor(data[[col]]) &&
        length(unique(data[[col]])) <= cat_threshold &&
        !is.element(col, cols_ignore)) {
      data[[col]] <- as.factor(data[[col]])
      cat(col, " converted to factor\n")
    }
  }
  data
}

isAlertData <- auto_convert_factors(data, 10, cols_ignore = list('IsAlert'))

#Dropping P8, V7, V9 as they are all zeroes. Dropping TrailID and ObsNum as they are unique ids.

isAlertData<-isAlertData[c("P1", "P2", "P3", "P4", "P5",
                           "P6", "P7", "E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8",
                           "E9", "E10", "E11", "V1", "V2", "V3", "V4","V5", "V6", "V8",
                           "V10", "V11", "IsAlert")]

datadummy<-dummyVars("~.",data=isAlertData,fullRank = F)

datatemp<-as.data.frame(predict(datadummy,isAlertData))

head(datatemp)

summary(datatemp)

#Partitioning dataset in to training and test dataset.

splitIndexMulti <- createDataPartition(datatemp$IsAlert, p=.001, list = FALSE, times = 1)

trainDataset <- datatemp[splitIndexMulti,]
testDataset <- datatemp[-splitIndexMulti,]
dim(datatemp)
dim(trainDataset)
dim(testDataset)

summary(trainDataset)
trainDataset[c(1:60000),]
#Variable Importance

fsModels2 <- c("glm","gbm","treebag","ridge","lasso","rf","xgbLinear")
myFs2<-fscaret(trainDataset, testDataset, myTimeLimit = 40, preprocessData = TRUE,
               Used.funcRegPred = fsModels2, with.labels = TRUE,
               supress.output=FALSE, no.cores = 2, installReqPckg = TRUE)

names(myFs2)
myFs2$VarImp
myFs2$PPlabels
myFs2$VarImp$matrixVarImp.MSE

results <- myFs2$VarImp$matrixVarImp.MSE
results$Input_no <- as.numeric(results$Input_no)
results <- results[c("SUM","SUM%","ImpGrad","Input_no")]
myFs2$PPlabels$Input_no <-  as.numeric(rownames(myFs2$PPlabels))
results <- merge(x=results, y=myFs2$PPlabels, by="Input_no", all.x=T)
results <- results[c('Labels', 'SUM')]
results <- subset(results,results$SUM !=0)
results <- results[order(-results$SUM),]
print(results)


#######################################################################
#                                                                     #
#                         4 - MODELLING                               #
#                                                                     #
#######################################################################

# Uses caret library, doing Automatic grid search (possible to do manual one as well)

modelTrain<-trainDataset
modelTrain$IsAlert <- as.factor(ifelse(modelTrain$IsAlert == 1,'Y','N'))


modelTest<-testDataset
modelTest$IsAlert <- as.factor(ifelse(modelTest$IsAlert == 1,'Y','N'))

#Defining training control
control <- trainControl(
  method          = "repeatedcv",
  number          = 5,
  repeats         = 2,
  search          = "grid",
  classProbs      = TRUE,
  summaryFunction = twoClassSummary, #ROC AUC
  verboseIter     = TRUE
)

## Before variable removal

model1 <- train(IsAlert ~ .,
                data = modelTrain[,!(names(modelTrain) %in% c("TrialID","ObsNum","P8","V7","V9"))],
                method = "gbm",
                metric = "ROC",
                na.action = na.pass,
                preProcess = c("center", "scale", "medianImpute"),
                trControl = control)


model2 <- train(IsAlert ~ .,
                data = modelTrain[,!(names(modelTrain) %in% c("TrialID","ObsNum","P8","V7","V9"))],
                method = "xgbLinear",
                metric = "ROC",
                na.action = na.pass,
                preProcess = c("center", "scale", "medianImpute"),
                trControl = control)

## After variable removal (Highly correlated: P4 and V1 removed)
model3 <- train(IsAlert ~ .,
                data = modelTrain[,!(names(modelTrain) %in% c("TrialID","ObsNum","P8","V7","V9","P4","V1"))],
                method = "gbm",
                metric = "ROC",
                na.action = na.pass,
                preProcess = c("center", "scale", "medianImpute"),
                trControl = control)


model4 <- train(IsAlert ~ .,
                data = modelTrain[,!(names(modelTrain) %in% c("TrialID","ObsNum","P8","V7","V9","P4","V1"))],
                method = "xgbLinear",
                metric = "ROC",
                na.action = na.pass,
                preProcess = c("center", "scale", "medianImpute"),
                trControl = control)

# Check out the hyperparameters
print(model1)

# Look at ROC results versus hyperparameter grid
plot(model1)

# Check out the hyperparameters
print(model2)

# Look at ROC results versus hyperparameter grid
plot(model2)

# Check out the hyperparameters
print(model3)

# Look at ROC results versus hyperparameter grid
plot(model3)

# Check out the hyperparameters
print(model4)

# Look at ROC results versus hyperparameter grid
plot(model4)


######## Results and Interpretation ####################

# Model 1
# Get predictions
test_results1 <- predict(model1,
                         modelTest,
                         na.action = na.pass,
                         type = "prob")

test_results1$obs <- modelTest$IsAlert
test_results1$pred <- predict(model1,modelTest, na.action = na.pass)

# Confusion Matrix
confusionMatrix(test_results1$pred,test_results1$obs, positive="Y")

# AUROC - This is what we compare with other models and select the best one
twoClassSummary(test_results1, lev = c("Y","N"))

# Plot ROC Curve
roc_results1 <- roc(modelTest$IsAlert,
                    predict(model1, modelTest, type="prob")[,1],
                    levels = rev(levels(modelTest$IsAlert)))

roc_results1
plot(roc_results1, print.thres = c(.5), type = "S",
     print.thres.pattern = "%.3f (Spec = %.3f, Sens = %.3f)",
     print.thres.cex = .8,
     legacy.axes = TRUE)
legend("topright", c("GBM Full"))


# Model 2
# Get predictions
test_results2 <- predict(model2,
                         modelTest,
                         na.action = na.pass,
                         type = "prob")

test_results2$obs <- modelTest$IsAlert
test_results2$pred <- predict(model2,modelTest, na.action = na.pass)

# Confusion Matrix
confusionMatrix(test_results2$pred,test_results2$obs, positive="Y")

# AUROC - This is what we compare with other models and select the best one
twoClassSummary(test_results2, lev = c("Y","N"))

# Plot ROC Curve
roc_results2 <- roc(modelTest$IsAlert,
                    predict(model2, modelTest, type="prob")[,1],
                    levels = rev(levels(modelTest$IsAlert)))
legend("topright", c("XGBoost Full"))

roc_results2
plot(roc_results2, print.thres = c(.5), type = "S",
     print.thres.pattern = "%.3f (Spec = %.3f, Sens = %.3f)",
     print.thres.cex = .8,
     legacy.axes = TRUE)

# Model 3
# Get predictions
test_results3 <- predict(model3,
                         modelTest,
                         na.action = na.pass,
                         type = "prob")

test_results3$obs <- modelTest$IsAlert
test_results3$pred <- predict(model3,modelTest, na.action = na.pass)

# Confusion Matrix
confusionMatrix(test_results3$pred,test_results3$obs, positive="Y")

# AUROC - This is what we compare with other models and select the best one
twoClassSummary(test_results3, lev = c("Y","N"))

# Plot ROC Curve
roc_results3 <- roc(modelTest$IsAlert,
                    predict(model3, modelTest, type="prob")[,1],
                    levels = rev(levels(modelTest$IsAlert)))

roc_results3
plot(roc_results3, print.thres = c(.5), type = "S",
     print.thres.pattern = "%.3f (Spec = %.3f, Sens = %.3f)",
     print.thres.cex = .8,
     legacy.axes = TRUE)
legend("topright", c("GBM Reduced"))

# Model 4
# Get predictions
test_results4 <- predict(model4,
                         modelTest,
                         na.action = na.pass,
                         type = "prob")

test_results4$obs <- modelTest$IsAlert
test_results4$pred <- predict(model4,modelTest, na.action = na.pass)

# Confusion Matrix
confusionMatrix(test_results4$pred,test_results4$obs, positive="Y")

# AUROC - This is what we compare with other models and select the best one
twoClassSummary(test_results4, lev = c("Y","N"))

# Plot ROC Curve
roc_results4 <- roc(modelTest$IsAlert,
                    predict(model4, modelTest, type="prob")[,1],
                    levels = rev(levels(modelTest$IsAlert)))

roc_results4
plot(roc_results4, print.thres = c(.5), type = "S",
     print.thres.pattern = "%.3f (Spec = %.3f, Sens = %.3f)",
     print.thres.cex = .8,
     legacy.axes = TRUE)
legend("topright", c("XGBoost Reduced"))

dev.off()
# P8  converted to factor
# E3  converted to factor
# E8  converted to factor
# E9  converted to factor
# V5  converted to factor
# V7  converted to factor
# V9  converted to factor
# V10  converted to factor


##### Preparing the validation dataset : Begin #######
isAlertDataTest <- auto_convert_factors(validatedata, 10, cols_ignore = list('IsAlert'))

#Dropping P8, V7, V9 as they are all zeroes. Dropping TrailID and ObsNum as they are unique ids.

isAlertDataTest<-isAlertDataTest[c("P1", "P2", "P3", "P4", "P5",
                                   "P6", "P7", "E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8",
                                   "E9", "E10", "E11", "V1", "V2", "V3", "V4","V5", "V6", "V8",
                                   "V10", "V11", "IsAlert")]



summary(isAlertDataTest)
datadummyTest<-dummyVars("~.", data=isAlertDataTest,fullRank = F)

datatempTest<-as.data.frame(predict(datadummyTest,isAlertDataTest))

##### Preparing the validation dataset : End #######

modelValidate<-datatempTest
modelValidate$IsAlert <- as.factor(ifelse(modelValidate$IsAlert == 1,'Y','N'))


########### The real test  #################

# Model 1
# Get predictions
test_results5 <- predict(model1,
                         modelValidate,
                         na.action = na.pass,
                         type = "prob")

test_results5$obs <- modelValidate$IsAlert
test_results5$pred <- predict(model1,modelValidate, na.action = na.pass)

# Confusion Matrix
confusionMatrix(test_results5$pred,test_results5$obs, positive="Y")

# AUROC - This is what we compare with other models and select the best one
twoClassSummary(test_results5, lev = c("Y","N"))

# Plot ROC Curve
roc_results5 <- roc(modelValidate$IsAlert,
                    predict(model1, modelValidate, type="prob")[,1],
                    levels = rev(levels(modelValidate$IsAlert)))

roc_results5
plot(roc_results5, print.thres = c(.5), type = "S",
     print.thres.pattern = "%.3f (Spec = %.3f, Sens = %.3f)",
     print.thres.cex = .8,
     legacy.axes = TRUE)
legend("topright", c("GBM Full"))


# Model 2
# Get predictions
test_results6 <- predict(model2,
                         modelValidate,
                         na.action = na.pass,
                         type = "prob")

test_results6$obs <- modelValidate$IsAlert
test_results6$pred <- predict(model2,modelValidate, na.action = na.pass)

# Confusion Matrix
confusionMatrix(test_results6$pred,test_results6$obs, positive="Y")

# AUROC - This is what we compare with other models and select the best one
twoClassSummary(test_results6, lev = c("Y","N"))

# Plot ROC Curve
roc_results6 <- roc(modelValidate$IsAlert,
                    predict(model2, modelValidate, type="prob")[,1],
                    levels = rev(levels(modelValidate$IsAlert)))

roc_results6
plot(roc_results6, print.thres = c(.5), type = "S",
     print.thres.pattern = "%.3f (Spec = %.3f, Sens = %.3f)",
     print.thres.cex = .8,
     legacy.axes = TRUE)
legend("topright", c("XGBoost Full"))


# Model 3
# Get predictions
test_results7 <- predict(model3,
                         modelValidate,
                         na.action = na.pass,
                         type = "prob")

test_results7$obs <- modelValidate$IsAlert
test_results7$pred <- predict(model3,modelValidate, na.action = na.pass)

# Confusion Matrix
confusionMatrix(test_results7$pred,test_results7$obs, positive="Y")

# AUROC - This is what we compare with other models and select the best one
twoClassSummary(test_results7, lev = c("Y","N"))

# Plot ROC Curve
roc_results7 <- roc(modelValidate$IsAlert,
                    predict(model3, modelValidate, type="prob")[,1],
                    levels = rev(levels(modelValidate$IsAlert)))

roc_results7
plot(roc_results7, print.thres = c(.5), type = "S",
     print.thres.pattern = "%.3f (Spec = %.3f, Sens = %.3f)",
     print.thres.cex = .8,
     legacy.axes = TRUE)
legend("topright", c("GBM Reduced"))

# Model 4
# Get predictions
test_results8 <- predict(model4,
                         modelValidate,
                         na.action = na.pass,
                         type = "prob")

test_results8$obs <- modelValidate$IsAlert
test_results8$pred <- predict(model4,modelValidate, na.action = na.pass)

# Confusion Matrix
confusionMatrix(test_results8$pred,test_results8$obs, positive="Y")

# AUROC - This is what we compare with other models and select the best one
twoClassSummary(test_results8, lev = c("Y","N"))

# Plot ROC Curve
roc_results8 <- roc(modelValidate$IsAlert,
                    predict(model4, modelValidate, type="prob")[,1],
                    levels = rev(levels(modelValidate$IsAlert)))

roc_results8
plot(roc_results8, print.thres = c(.5), type = "S",
     print.thres.pattern = "%.3f (Spec = %.3f, Sens = %.3f)",
     print.thres.cex = .8,
     legacy.axes = TRUE)
legend("topright", c("XGBoost Reduced"))
}
