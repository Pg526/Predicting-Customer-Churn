#STAT-642 FINAL PROJECT#

setwd("~/Desktop/Drexel/MS BA/Winter 2021/STAT 642-674/Final Project")

# CLUSTERING#


#Loading the libraries
library(ggplot2) 
library(caret) 
library(cluster) # clustering
library(factoextra) # cluster validation, plots
library(fpc) # cluster validation
library(Rtsne) # dimension reduction
library(pROC) #ROC or Area under the curve
library(ROCR)

install.packages(ROCR)

?caTools
 
??prediction
## Classification Analysis
## k-Nearest Neighbors (kNN)
#------------------------------------------
#------------------------------------------
######### Preliminary Code #########
#------------------------------------------
## Clear your workspace
# If your workspace is NOT empty, run:
rm(list=ls(cen_mm))
# to clear it
## Load Data
cc2 <- read.csv(file = "CustomerChurn.csv",
               stringsAsFactors = TRUE)

#Remove CustomerID variable, it is not needed

cc3 <- cc2[,-1]

## Data Exploration & Preparation

# First, let's obtain high-level information
# about the cc dataframe to look at the
# variable types and to check for missing (NA)
# values.
str(cc3)

## Prepare Target (Y) Variable
# We can convert our class variable,
# Churn to a nominal factor variable
# Note: No = 1, Yes = 2
cc3$Churn <- factor(x = cc3$Churn,
                        levels = c("No", "Yes"))

# We can plot the distribution of the target
# variable
plot(cc3$Churn, 
     main = "Churn3")

## Prepare Predictor (X) Variables
# All of our potential predictors are numeric,
# but we can look at the output from the table()
# function for integers to assess if they
# are discrete numerical or should be considered
# categorical
names(cc3)[sapply(X = cc3, 
                 FUN = is.integer)]

lapply(X = cc3[,names(cc3)[sapply(X = cc3, 
                                FUN = is.integer)]],
       FUN = table)

# We have two categorical variables, SeniorCitizen and
# tenure. We can keep SeniorCitizen as-is, since it is
# already binary. 
#Also we will keep tenure as-is, since it is an integer but it clearly takes
# so many values and will not be considered as factor variables

# cc3$tenure <- factor(x = cc3$tenure)

# Since we will need to transform numeric
# variables, we can set up a vector of the
# variable names
nums <- names(cc3)[names(cc3) %in% c("MonthlyCharges", "TotalCharges","tenure")]

#------------------------------------------

## Data Preprocessing & Transformation

# For kNN, we know that we are okay keeping 
# irrelevant X variables, but we need to impute 
# or remove any missing values, remove 
# redundant variables, transform/rescale
# our numeric variables and binarize categorical
# variables.

## 1. Missing values
# First, we check for missing values. If
# missing values are present, we can
# either remove them row-wise or perform
# imputation.
any(is.na(cc3))
summary(cc3)
#there are any na values, remove them
cc4 <- na.omit(cc3)
summary(cc4)

## 2. Redundant Variables
# We need to identify highly correlated 
# numeric input (X) variables and exclude 
# them from our predictive model. 

# First, we obtain the correlation matrix
# for our numeric predictor variables
cor_vars <- cor(x = cc4[ ,nums])

#corelation is checked for only numeric values

# We can start by looking at the symbolic 
# correlation matrix to manually identify 
# correlated variables
symnum(x = cor_vars,
       corr = TRUE)

# We can use the findCorrelation() function 
# in the caret package to identify redundant 
# variables for us. Setting names = TRUE will 
# output a list of the variable name that are 
# determined to be redundant. We can then remove 
# them from our vars vector and exclude them 
# from our analysis.
high_corrs <- findCorrelation(x = cor_vars, 
                              cutoff = .75, 
                              names = TRUE)

#input above is corelation matrix created which is empty

# By running a code line with the name of
# the output object (high_corrs), we can
# view the names of the redundant variables
high_corrs

#There is no redundant varibale

# Now, we can remove them from our vars
# vector so that we exclude them from our
# list of input (X) variable names

nums <- nums[!nums %in% high_corrs]
#Since there are no redundant variables, nums vector will remain as is

## 3. Rescale Numeric Variables
# kNN has been shown to perform well with 
# min-max (range) normalization, converting
# the numeric variables to range between
# 0 and 1. We can use the preProcess()
# and predict() functions and save the 
# rescaled data as a new dataframe, 
# cc_mm.
# Note: we can apply range normalization
# to binary variables without issue. Other
# transformations can be problematic

cen4_mm <- preProcess(x = cc4[ ,nums],
                     method = "range")
cc4_mm <- predict(object = cen4_mm,
                 newdata = cc4)


#for min max even the binary variables can be taken into account, it will
#give the same result because min max will try and put the values in 0-1 range

#but for other methods "mean" or centroid the values will be different


## 4. Binarization
# If categorical input (X) variables are 
# used in analysis, they must be converted
# to binary variables using the class2ind()
# function from the caret package for 
# categorical variables with 2 class levels and
# the dummyVars() function from the caret 
# package and the predict() function for
# categorical variables with more than 2
# class levels. 

# We will binarize the factor variable, which
# has

#Below: Check the levels for individual class variables
#nlevels(cc4$gender) # class levels
#nlevels(cc4$Partner)
#nlevels(cc4$Dependents)
#nlevels(cc4$PhoneService)
#nlevels(cc4$MultipleLines)
#nlevels(cc4$InternetService)
#nlevels(cc4$OnlineSecurity)
#nlevels(cc4$OnlineBackup)
#nlevels(cc4$DeviceProtection)
#nlevels(cc4$TechSupport)
#nlevels(cc4$StreamingTV)
#nlevels(cc4$StreamingMovies)
#nlevels(cc4$Contract)
#nlevels(cc4$PaperlessBilling)
#nlevels(cc4$PaymentMethod)

#Creating dummy variables for individual class variables
#cats1 <- dummyVars(formula =  ~ gender,
 #                 data = cc4)
#cats1_dums <- predict(object = cats1, 
 #                    newdata = cc4)

#cats2 <- dummyVars(formula =  ~ Partner,
 #                  data = cc4)
#cats2_dums <- predict(object = cats2, 
 #                     newdata = cc4)
#cats3 <- dummyVars(formula =  ~ Dependents,
 #                  data = cc4)
#cats3_dums <- predict(object = cats3, 
 #                     newdata = cc4)
#cats4 <- dummyVars(formula =  ~ PhoneService,
 #                  data = cc4)
#cats4_dums <- predict(object = cats4, 
 #                     newdata = cc4)
#cats5 <- dummyVars(formula =  ~ MultipleLines,
 #                  data = cc4)
#cats5_dums <- predict(object = cats5, 
 #                     newdata = cc4)

#Convert all the class variables into dummy variables
cats101 <- dummyVars(formula =  "~ .",
                   data = cc4)
cats101_dums <- predict(object = cats101, 
                      newdata = cc4)

#Check for warnings if there are any
#warnings()

#classification models are not able to capture one less dummy variable as binarized 
#as it does in the case of linear regression

# Combine binarized variables with data
# (transformed numeric variables, factor 
# target variable)
cc4_mm_dum <- data.frame(cc4_mm,cats101_dums)
names(cc4_mm_dum)

# Create vars vector of the names of the variables
# to use as input to the kNN model
cc4_mm_dum_final <- cc4_mm_dum[,-c(1:19,65:67)]

vars101 <- names(cc4_mm_dum_final)[!names(cc4_mm_dum_final) %in% "Churn"]
vars101

#------------------------------------------

## Training & Testing

# Finally, we use the createDataPartition()
# function from the caret package to identify
# the row numbers that we will include in our
# training set. Then, all other rows will be
# put in our testing set. We split the data
# using an 85/15 split (85% in training and
# 15% in testing). By using createDataPartition()
# we preserve the distribution of our outcome (Y).

set.seed(29619122) # initialize the random seed

# Generate the list of observations for the
# train dataframe
sub101 <- createDataPartition(y = cc4_mm_dum_final$Churn, 
                           p = 0.85, 
                           list = FALSE)



# Subset the rows of the cc4_mm_dum dataframe
# to include the row numbers in the sub object
# to create the train dataframe
train101 <- cc4_mm_dum_final[sub101, ] 
train102 <- train101[!names(train101) %in% "Churn"]
train102


# Use all observations not in the sub object
# to create the test dataframe
test101 <- cc4_mm_dum_final[-sub101, ]
test102 <- test101[!names(test101) %in% "Churn"]
test102



#------------------------------------------

## Analysis

### Basic Model (knn() in the class package)

## First, we can try using a 'best guess' 
# value of k (square root of the number 
# of training observations)
ceiling(sqrt(nrow(train101)))
# 78

## Naive Model Building
# To build a basic (Naive) kNN model
# we can use the knn() function from the
# class package
knn.pred101 <- knn(train102,test102, 
                cl = train101$Churn, 
                k = 77)



# With kNN, no model is built, so we cannot 
# assess performance on the training set
# (or goodness of fit). 
# For this reason, we move on to looking at
# performance on the testing set.

# We can use the confusionMatrix() function
# from the caret package to obtain a 
# confusion matrix and obtain performance
# measures for our model. We can set mode =
# "everything" to obtain all available
# performance measures

conf_basic <- confusionMatrix(data = knn.pred101, # vector of Y predictions
                              reference = test101$Churn, # actual Y
                              positive = "Yes", # positive class for class-level performance
                              mode = "everything") # all available measures

# We can view the output by running a code
# line with the name of the object
conf_basic
accuracy <- mean(observed.classes == predicted.clases)
#------------------------------------------

### Hyperparameter Tuning 

# (using the train() function in the caret 
# package)

# We can perform k-fold cross validation
# for model tuning to help us choose the
# best possible hyperparameter value (k)

# Note: specifying tuneLength = 15 and no 
# particular hyperparameter search method will 
# perform a default grid search

# By default, the train() function will 
# determine the 'best' model based on Accuracy 
# for classification and RMSE for regression. For
# classification models, the Accuracy and Kappa 
# are automatically computed and provided. 

# First, we set up a trainControl object
# (named ctrl) using the trainControl() 
# function in the caret package. We specify 
# that we want to perform 10-fold cross 
# validation, repeated 3 times. We use this
# object as input to the trControl argument
# in the train() function below.
ctrl <- trainControl(method = "repeatedcv",
                     number = 10,
                     repeats = 3)

# Next, we initialize a random seed for 
# our cross validation
set.seed(29619112)

# Then, we use the train() function to
# train the kNN model using 10-Fold Cross 
# Validation (repeated 3 times). We set
# tuneLength = 15 to try the first 15
# default values of k (odd values from
# k = 5:33)
knnFit101 <- train(x = train102,
                 y = train101$Churn, 
                 method = "knn", 
                 trControl = ctrl, 
                 tuneLength = 50)
??train
# We can view the results of our
# cross validation across k values
# for Accuracy and Kappa. The output
# will also identify the optimal k.
knnFit101

# We can plot the train() object
# (knnFit1) using the plot() function
# to view a plot of the hyperparameter,
# k, on the x-axis and the Accuracy 
# on the y-axis.
plot(knnFit101)

# We can view the confusion matrix
# showing the average performance of the model
# across resamples
confusionMatrix(knnFit101)

#------------------------------------------

### Model Performance

# Finally, we can use our best tuned model to 
# predict the testing data.
# First, we use the predict() function to 
# predict the value of the median_val variable 
# using the model we created using the train()
# function, knnFit1 model and the true 
# classes of the median_val in the test 
# dataframe.
outpreds <- predict(object = knnFit101, 
                    newdata = test102)

# Again, we can use the confusionMatrix() 
# function from the caret package to obtain a 
# confusion matrix and obtain performance
# measures for our model. We can set mode =
# "everything" to obtain all available
# performance measures
conf_tuned <- confusionMatrix(data = outpreds, 
                              reference = test101$Churn, 
                              positive = "Yes",
                              mode = "everything")
conf_tuned


# We can describe the overall performance 
# based on our accuracy and kappa values.

conf_tuned$overall[c("Accuracy", "Kappa")]

# We can describe class-level performance
# for the different class levels. Note,
# above, we set positive = "Above", since we
# are more interested in predicting above median
# properties than below median
conf_tuned$byClass

# Note: We could run it again setting 
# positive = "Below" to get class-level
# performance for the Below class

## Comparing Base & Tuned Models

# Since we do not have a training set,
# we cannot assess goodness of fit. Instead,
# we will compare performance across the
# 'Naive' (Best Guess) and tuned models on 
# the testing data set

# Overall Model Performance
cbind(Base = conf_basic$overall,
      Tuned = conf_tuned$overall)

# Class-Level Model Performance
cbind(Base = conf_basic$byClass,
      Tuned = conf_tuned$byClass)


mean(outpreds == test101$Churn)
plot(knnFit101, print.thres = 0.5, type="S")
outpreds <- predict(object = knnFit101, 
                    newdata = test102, type = "prob")
knnROC <- roc(test101$Churn,outpreds[,"No"], levels = rev(test101$Churn))
knnROC
plot(knnROC, type="S", print.thres= 0.5)
??knnPredict
conf_basic
fp = 55
fn = 151
tp = 129
tn = 719
fpr = fp / (fp + tn)
tpr = tp / (tp + fn)
AUC <- 1/2 - fpr/2 + tpr/2
AUC

conf_basic$table
conf_tuned$table

summary(cc4$churn)

cc4_mm_dum_final_churn <- table(cc4_mm_dum_final$Churn)
cc4_mm_dum_final_churn

onlinesec <- table(cc4$OnlineSecurity)
onlinesec



??knn
knn.pred201 <- knn(train102,test102, 
                   cl = train101$Churn, 
                   k = 21)

conf_basic201 <- confusionMatrix(data = knn.pred201, # vector of Y predictions
                              reference = test101$Churn, # actual Y
                              positive = "Yes", # positive class for class-level performance
                              mode = "everything")
conf_basic201

knnFit201 <- train(x = train102,
                   y = train101$Churn, 
                   method = "knn", 
                   trControl = ctrl, 
                   tuneLength = 50)

plot(knnFit201)
confusionMatrix(knnFit201)
################################################################################

# Load data

churn <- read.csv("CustomerChurn.csv", stringsAsFactors = FALSE)

# Load libraries
library(caret)
library(rpart)
library(rpart.plot)
library(e1071)
library(naivebayes)
install.packages("naivebayes")
# Prepare the target variable

churn$Churn <- factor(churn$Churn)

# To get a general overview of our data, let's plot our target variable

plot(churn$Churn, main = "Churn")

# Let's also look at how our individual variables influence Churn
# We'll use stacked bar plots for our categorical variables and box-plots for our 
# numeric variables
#Descriptive Analytics
ggplot(data = churn, mapping = aes(x = gender,fill = Churn)) + geom_bar()
ggplot(data = churn, mapping = aes(x = SeniorCitizen,fill = Churn)) + geom_bar()
ggplot(data = churn, mapping = aes(x = Partner,fill = Churn)) + geom_bar()
ggplot(data = churn, mapping = aes(x = Dependents,fill = Churn)) + geom_bar()
ggplot(data = churn, mapping = aes(x = Churn, y = tenure)) + geom_boxplot()
ggplot(data = churn, mapping = aes(x = PhoneService,fill = Churn)) + geom_bar()
ggplot(data = churn, mapping = aes(x = MultipleLines,fill = Churn)) + geom_bar()
ggplot(data = churn, mapping = aes(x = InternetService,fill = Churn)) + geom_bar()
ggplot(data = churn, mapping = aes(x = OnlineSecurity,fill = Churn)) + geom_bar()
ggplot(data = churn, mapping = aes(x = OnlineBackup,fill = Churn)) + geom_bar()
ggplot(data = churn, mapping = aes(x = DeviceProtection,fill = Churn)) + geom_bar()
ggplot(data = churn, mapping = aes(x = TechSupport,fill = Churn)) + geom_bar()
ggplot(data = churn, mapping = aes(x = StreamingTV,fill = Churn)) + geom_bar()
ggplot(data = churn, mapping = aes(x = StreamingMovies,fill = Churn)) + geom_bar()
ggplot(data = churn, mapping = aes(x = Contract,fill = Churn)) + geom_bar()
ggplot(data = churn, mapping = aes(x = PaperlessBilling,fill = Churn)) + geom_bar()
ggplot(data = churn, mapping = aes(x = PaymentMethod,fill = Churn)) + geom_bar()
ggplot(data = churn, mapping = aes(x = Churn, y = MonthlyCharges)) + geom_boxplot()
ggplot(data = churn, mapping = aes(x = Churn, y = TotalCharges)) + geom_boxplot()

# We will set up convenience vectors for our numeric and categorical variables]

cats_NB <- c("gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", 
          "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
          "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", 
          "Contract", "PaperlessBilling", "PaymentMethod")
churn[,cats_NB] <- lapply(X = churn[,cats_NB], FUN = factor)

nums_NB <- c("tenure", "MonthlyCharges", "TotalCharges")
churn[,nums_NB] <- lapply(X = churn[,nums], FUN = as.numeric)

# We will combine cats and nums to create vars, which we will use to predict churn

vars_NB <- c(cats_NB, nums_NB)

# We will now check to see if we have any missing values

any(is.na(churn))

# There are indeed missing values, so we must remove them

churn <- na.omit(churn)

# Initialize a seed to be used in calculations

set.seed(29619122)

# With the seed initialized, we can now create our training indices
# We will use a p of 0.85 to get a high amount of training data for our model

sub_NB <- createDataPartition(y = churn$Churn, p = 0.85, list = FALSE)

# We will create 2 dataframes - one training and one testing

train_NB <- churn[sub_NB, ]
test_NB <- churn[-sub_NB, ]

# With the initial setup complete, we can now move into Naive Bayes Classification

# To begin this process, we must first check correlation

cor(x = churn[ ,nums_NB])

# Total charges correlates strongly with both tenure and monthly charges, so we'll
# want to remove it.

vars_NB <- vars_NB[!vars %in% "TotalCharges"]
vars_NB

# We will now want to check our numeric variables to see if they are normally distributed

hist(x = churn$MonthlyCharges)
hist(x = churn$tenure)

# Neither is normally distributed, so we'll need to transform them
# Tenure has 0-values, so we'll want to use a Yeo-Johnson transformation.

Norm_NB <- preProcess(x = churn[ ,vars_NB], method = c("YeoJohnson", "center", "scale"))

# Training and Testing datasets have already been made, so we'll apply 
# transformations directly to them

train_NB_pred <- predict(object = Norm_NB, newdata = train_NB)

test_NB_pred <- predict(object = Norm_NB, newdata = test_NB)

# We must now determine if Laplace smoothing needs to be applied

aggregate(train_NB_pred[ ,cats_NB], by = list(train_NB_pred$Churn), FUN = table)

# There are no 0-categories, so Laplace smoothing does not need to be applied!

# Next, we will ceate our Naive Bayes model

NB_model <- naiveBayes(x = train_NB_pred[ ,vars_NB], y = train_NB_pred$Churn, laplace = 0)
NB_model

# Using our model, we will now generate class predictions

NB_train <- predict(object = NB_model, newdata = train_NB_pred[ ,vars_NB], type = "class")
head(NB_train)

# We will now use a confusion matrix to generate performance measures

NB_train_conf <- confusionMatrix(data = NB_train, reference = train_NB_pred$Churn, 
                                 positive = "Yes", mode = "everything")
NB_train_conf

# We will now go through the same process for our testing model

NB_test <- predict(object = NB_model, newdata = test_NB_pred[ ,vars_NB], type = "class")

NB_test_conf <- confusionMatrix(data = NB_test, reference = test_NB_pred$Churn,
                                positive = "Yes", mode = "everything")

NB_test_conf

# We'll now test our overall performance based on accuracy and kappa values
# This will show the general accuracy of our model and our model's accuracy 
# accounting for chance

NB_test_conf$overall[c("Accuracy", "Kappa")]

# We'll also observe performance by class

NB_test_conf$byClass

# Let's now test goodness of fit for both overall and class-level performance

# Overall
cbind(Training = NB_train_conf$overall, Testing = NB_test_conf$overall)

# By class
cbind(Training = NB_train_conf$byClass, Testing = NB_test_conf$byClass)

# Our results show that the model is balanced

# Next, we'll look at variable importance

# First, we'll set up the grid to meet Naive Bayes' needs
# NOTE: Some code for the variable importance was obtained from the StackOverlow
# page: "Determine Variables of Importance in Naive Bayes Model".
# Code from this website will be preceded with a comment 'SOF'

# SOF
grids_NB <- data.frame(usekernel=TRUE,laplace = 0,adjust=1)

grids_NB

# Next, we'll set up our ctrl object

ctrl_NB <- trainControl(method = "repeatedcv", number = 10, repeats = 3, search = "grid")

set.seed(29619122)

# With this set up, we can make our NB_Fit object using the grids and ctrl objects
# Our method will be naive_bayes
# SOF

NB_Fit <- train(form = Churn ~ .,
                data = train_NB[ ,c(vars_NB, "Churn")],
                method = "naive_bayes",
                trControl = ctrl_NB,
                tuneGrid = grids_NB)

# We can now check and plot our variable importance

NB_Imp <- varImp(NB_Fit)
plot(NB_Imp)

#################################################################################

# Unsupervised learning using Cluster Analysis

# setting the working directory

setwd('D:/Sadhana_Applications/Drexel/Winter Term/STAT 642/Final Project')

#Loading the required libraries

library(ggplot2) 
library(caret) 
library(cluster) 
library(factoextra) 
library(fpc) 
library(Rtsne) 

# Loading the RData file that contains the necessary functions and objects
# which will be used during validation of HCA and KMeans

load("Clustering.RData")

#Reading the given data into a dataset named 'cc'

cc_CA<- read.csv(file = "CustomerChurn.csv")

# viewing the structure of the data
str(cc_CA)

# viewing the summary of the data
summary(cc_CA)

# creating a vector named 'facs' containing categorical variables 
facs_CA <- c("customerID", "gender", "SeniorCitizen","Partner","Dependents","PhoneService",
          "MultipleLines","InternetService","OnlineSecurity","OnlineBackup","DeviceProtection",
          "TechSupport","StreamingTV","StreamingMovies","Contract","PaperlessBilling","PaymentMethod",
          "Churn")

# creating a vector named 'nums' containing numerical variables
nums_CA <- names(cc_CA)[!names(cc_CA) %in% facs_CA]

# viewing what is in nums and facs 
facs_CA
nums_CA

# converting the categorical variables to factors
cc_CA[ ,facs_CA] <- lapply(X = cc_CA[ ,facs_CA], FUN = factor)

# Handling the missing values

any(is.na(cc_CA)) #see if there are any missing values in the data.

cen_CA <- preProcess(x = cc_CA, method = "medianImpute") # replacing missing values by imputing median values in them.

cc_CA_nn <- predict(object = cen,newdata = cc_CA)

any(is.na(cc_CA_nn)) # confirming if the missingness is removed.

#Standardizing the data 

cen_CA_cc <- preProcess(x = cc_CA_nn,
                     method = c("center", "scale"))
cc_CA_nn <- predict(object = cen_CA_cc,
                 newdata = cc_CA_nn)


#Descriptive statistics

#Mode for all variables
??mode
lapply(X = cc_CA, 
       FUN = mode)

#variance only for numeric variables
sapply(X = cc_CA[ ,nums_CA], 
       FUN = var)

#standard deviation for numeric variables
sapply(X = cc_CA[ , nums_CA], 
       FUN = sd)

#Categorical variables- frequency
CFreq<-lapply(X = cc_CA[ ,facs_CA], FUN = table)



#Visualizing the target variable 'Churn'

plot(cc_CA_nn$TechSupport,cc_CA_nn$Churn)

#Dimensionality reduction

pca_CA <- prcomp(x = cc_CA_nn[, nums],
              scale. = TRUE)

pca_CA

screeplot(pca_CA, type = "lines")

#to visualize the numeric variables

ggplot(data = cc_CA_nn, mapping = aes(x=tenure,y=TotalCharges)) +
  geom_point()

ggplot(data = cc_CA_nn, mapping = aes(x=MonthlyCharges,y=TotalCharges)) +
  geom_point()

hist(x=cc_CA_nn$tenure,xlab="",col="Steelblue")
hist(x=cc_CA_nn$MonthlyCharges,xlab="",col="Steelblue")
hist(x=cc_CA_nn$TotalCharges,xlab="",col="Steelblue")

#frequency of categorical variables
contractFreq <- table(cc_CA_nn$Contract)
contractFreq

#Identifying outliers

cc_CA_box <- boxplot(x = cc_CA_nn[ ,nums_CA], 
                  main = "Numerical")

cc_CA_box$out # outlier values

#There are no outliers in the data.

#Finding the correlation between the numerical variables

cor(x=cc_CA_nn[ , nums_CA])

#strong correlation between MonthlyCharges and TotalCharges


#hierarchical clustering--------------------------------------

#we transform the data to normal distribution

cen_CA_yeojo <- preProcess(x= cc_CA_nn, method ="YeoJohnson")
cc_CA_yeojo <- predict(object = cen_CA_yeojo,
                    newdata = cc_CA_nn)

#verifying if the variables are normalized (considering TotalCharges)
hist(cc_CA_nn$TotalCharges) #Before normalization
hist(cc_CA_yeojo$TotalCharges) #After normalization

#removing variables that are not of interest
mydata_CA<-cc_CA_yeojo[ ,c(-1,-21)] #removing variables customerID and Churn 

# Using gower distance method to calculate distances and similarity since there are 
#mixed variables

hdist_CA <- daisy(x = mydata_CA, metric= "gower")

summary(hdist_CA)

#Agglomerative clustering

#Measuring dissimilarity between clusters using single linkage
sing_CA<- hclust(hdist_CA,method = "single")

#dendogram for single linkage

plot(sing_CA, sub = NA, xlab = NA, 
     main = "Single Linkage")

rect.hclust(tree = sing_CA, # hclust object
            k = 3, # # of clusters
            border = hcl.colors(3)) # k colors for boxes

single_clusters <- cutree(tree = sing_CA, k = 3)

#complete linkage

comp_CA<-hclust(hdist_CA, method ="complete")

plot(comp_CA, sub = NA, xlab = NA, 
     main = "complete Linkage")

rect.hclust(tree = comp_CA, k = 3, 
            border = hcl.colors(3))

complete_clusters <- cutree(tree = comp_CA, k = 3)

#average linkage

avg_CA<- hclust(d = hdist_CA, 
             method = "average")
plot(avg_CA, 
     sub = NA, xlab = NA, 
     main = "Average Linkage")

rect.hclust(tree = avg_CA, k = 3, 
            border = hcl.colors(3))

avg_clusters <- cutree(tree = avg_CA, k = 3)

#centroid


cent_CA<- hclust(d = hdist_CA ^ 2, 
              method = "centroid")

plot(cent_CA, 
     sub = NA, xlab = NA, 
     main = "Centroid Linkage")


rect.hclust(tree = cent_CA, k = 3, 
            border = hcl.colors(3))

cent_clusters <- cutree(tree = avg_CA, k = 3)

#wards method

wards_CA <- hclust(d = hdist_CA, 
                method = "ward.D2")

plot(wards_CA, 
     xlab = NA, sub = NA, 
     main = "Ward's Method")

rect.hclust(tree = wards_CA, k = 3, 
            border = hcl.colors(3))

wards_clusters <- cutree(tree = wards_CA, k = 3)

#Visualizing the solutions of the above methods

# Reducing the dimensionality using Rtsne function from Rtsne package

ld_dist <- Rtsne(X = hdist_CA, 
                 is_distance = TRUE)

lddf_dist <- data.frame(ld_dist$Y)

#Single linkage

ggplot(data = lddf_dist, 
       mapping = aes(x = X1, y = X2)) +
  geom_point(aes(color = factor(single_clusters))) +
  labs(color = "Cluster")

# Complete Linkage
ggplot(data = lddf_dist, 
       mapping = aes(x = X1, y = X2)) +
  geom_point(aes(color = factor(complete_clusters))) +
  labs(color = "Cluster")

#centroid linkage

ggplot(data = lddf_dist, 
       mapping = aes(x = X1, y = X2)) +
  geom_point(aes(color = factor(cent_clusters))) +
  labs(color = "Cluster")

#avg_Cluster
ggplot(data = lddf_dist, 
       mapping = aes(x = X1, y = X2)) +
  geom_point(aes(color = factor(avg_clusters))) +
  labs(color = "Cluster")

# Ward's Method
ggplot(data = lddf_dist, 
       mapping = aes(x = X1, y = X2)) +
  geom_point(aes(color = factor(wards_clusters))) +
  labs(color = "Cluster")


#Describing cluster solutions

aggregate(x = cc_CA_nn[ ,nums_CA], 
          by = list(wards_clusters),
          FUN = mean)


aggregate(x = cc_CA_nn[ ,facs_CA], 
          by = list(wards_clusters), 
          FUN = table)


ac_CA <- function(x) {
  agnes(mydata_CA, method = x)$ac
}

w_CA<-agnes(mydata_CA,method = "ward")
c_CA<-agnes(mydata_CA,method = "complete")
a_CA<-agnes(mydata_CA,method = "average")
s_CA<-agnes(mydata_CA,method = "single")

w_CA$ac
c_CA$ac
a_CA$ac
s_CA$ac

# Cluster validation

#k= 3

table(CustomerChurn_CA = cc_CA_nn$Churn, 
      Clusters = complete_clusters)

table(CustomerChurn_CA = cc_CA_nn$Churn, 
      Clusters = wards_clusters)

#Adjusted Rand Index

cluster.stats(d = hdist_CA, # distance matrix
              clustering = wards_clusters, # cluster assignments
              alt.clustering = as.numeric(cc_CA_nn$Churn))$corrected.rand # known groupings

#Internal Validation
#Cophenetic correlation

# Single Linkage
cor(x = hdist_CA, y = cophenetic(x = sing_CA))

# Complete Linkage
cor(x = hdist_CA, y = cophenetic(x = comp_CA))

# Average Linkage
cor(x = hdist_CA, y = cophenetic(x = avg_CA))

# Centroid Linkage
cor(x = hdist_CA ^ 2, y = cophenetic(x = cent_CA))

# Ward's Method
cor(x = hdist_CA, y = cophenetic(x = wards_CA))


#wss or SSE
load("Clustering.RData")
wss_plot(dist_mat = hdist_CA, # distance matrix
         method = "hc", # HCA
         hc.type = "average", # linkage method
         max.k = 15) # maximum k value
## Elbows at k = 3, 5


## Silhouette Method

# Hierarchical Cluster Analysis (method = "hc")
sil_plot(dist_mat = hdist_CA, # distance matrix
         method = "hc", # HCA
         hc.type = "complete", # average linkage
         max.k = 15) # maximum k value
#-----------------------------------------------

#K-means
# seed value 29619122

cen_CA_cc <- preProcess(x = cc_CA_yeojo,
                     method = c("center", "scale"))
cc_CA_yjcc <- predict(object = cen_CA_cc,
                   newdata = cc_CA_yeojo)

set.seed(29619122)

#kmeans1 <- kmeans(x = mydata[ ,nums], # data
#                 centers = 3,# # of clusters
#                trace = FALSE,nstart = 30)

kmeans1 <- kmeans(x = cc_CA_yjcc[,nums_CA], # data
                  centers = 3, # # of clusters
                  trace = FALSE, 
                  nstart = 30)
kmeans1



kmeans1$size

fviz_cluster(object = kmeans1, 
             data = cc_CA_yjcc[ ,nums_CA])

#autoplot(kmeans1,mydata,frame=TRUE)

kmeans1$centers
## Describe the Cluster Solution

clus_means_kMC <- aggregate(x = cc_CA_yjcc[ ,nums_CA], 
                            by = list(kmeans1$cluster), 
                            FUN = mean)
clus_means_kMC

#----------------------------------------------


# validation


table(Customerchurn_CA = cc_CA$Churn, 
      Clusters = kmeans1$cluster)

#External validation using Adjusted Rand

# k-Means Clustering (kMC, k = 3)
cluster.stats(d = dist(cc_CA_yjcc[ ,nums_CA]), # distance matrix for data used
              clustering = kmeans1$cluster, # cluster assignments
              alt.clustering = as.numeric(cc_CA$Churn))$corrected.rand

#internal validation 



# wss plot
set.seed(29619122)

## Elbows at k = 3, 5
wss_plot(scaled_data = cc_CA_yjcc[ ,nums_CA], # dataframe
         method = "kmeans", # kMC
         max.k = 15, # maximum k value
         seed_no = 29619122) 

#Silplot
# kMeans Clustering (kMC) (method = "kmeans")
sil_plot(scaled_data = cc_CA_yjcc[ ,nums_CA], # scaled data
         method = "kmeans", # kMC
         max.k = 15, # maximum k value
         seed_no = 29619122) # seed value for set.seed()
# k = 14 is maximum, k = 2, 5 are local maxima




