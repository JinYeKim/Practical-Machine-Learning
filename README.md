-------------
title= "Practical Machine Learning Project"
Author= "Jin Ye Kim"
0utput: html_document
-------------
##Background
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement ??? a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

This project is Data Scientist Specialization's Practical Machine Learning course assignment in Coursera.

##Data Overview



This human activity recognition research has traditionally focused on discriminating between different activities, i.e. to predict "which" activity was performed at a specific point in time (like with the Daily Living Activities dataset above). The approach we propose for the Weight Lifting Exercises dataset is to investigate "how (well)" an activity was performed by the wearer. The "how (well)" investigation has only received little attention so far, even though it potentially provides useful information for a large variety of applications,such as sports training.

In this work (see the paper) we first define quality of execution and investigate three aspects that pertain to qualitative activity recognition: the problem of specifying correct execution, the automatic and robust detection of execution mistakes, and how to provide feedback on the quality of execution to the user. We tried out an on-body sensing approach (dataset here), but also an "ambient sensing approach" (by using Microsoft Kinect - dataset still unavailable) 

Six young health participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions: exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E).

Class A corresponds to the specified execution of the exercise, while the other 4 classes correspond to common mistakes. Participants were supervised by an experienced weight lifter to make sure the execution complied to the manner they were supposed to simulate. The exercises were performed by six male participants aged between 20-28 years, with little weight lifting experience. We made sure that all participants could easily simulate the mistakes in a safe and controlled manner by using a relatively light dumbbell (1.25kg).


Read more: http://groupware.les.inf.puc-rio.br/har#ixzz4H4O4j8yf


##Set up
###Library set up
```{r}
library(RCurl)
library(caret)
library(randomForest)
library(rpart)
library(rpart.plot)
library(rattle)
```

Let's  bring the data!
```{r}
URL1 <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
A <- getURL(URL1)
train <- read.csv(textConnection(A))
dim(train)
names(train)
head(train[1:6])
URL2 <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
B <- getURL(URL2)
test <- read.csv(textConnection(B))
dim(test)
names(test)
head(test[1:6])
```

###Tidy up the data
We now split the data into training data and testing data. 
```{r}
inTrain <- createDataPartition(train$classe, p=0.7, list=FALSE)
training <- train[inTrain, ]
testing <- train[-inTrain, ]
dim(training); dim(testing)
```

Since we cannot touch testing data, we only get near zero variance from training data and subtract it from both training and testing data.
```{r}
NZV <- nearZeroVar(training)
training <- training[, -NZV]
testing <- testing[, -NZV]
dim(training); dim(testing)
```

Now let's remove NAs with similar idea!
```{r}
sparseness <- function(a){
  n <- length(a)
  na.count <- sum(is.na(a))
  return((n-na.count)/n)
}
variable.sparseness <- apply(training, 2, sparseness)
Training <- training[, variable.sparseness > 0.9]
dim(training); dim(testing)
```

When we looked at the data, we saw that first 5 columns are the participants' information. We don't need them for our project. Let's also remove them!
```{r}
Training <- Training[, -(1:5)]
testing <- testing[, -(1:5)]
dim(Training); dim(testing)
```


###Prediction
We will use three methods learne from Coursera, random forest, decision trees, and gerneralized boosted model. Let's see which method has the most accuracy.

###1. Random Forest
```{r}
modFit <- train(classe ~., data=Training, method="rf")
modFit$finalModel
```

```{r}
predict <- predict(modFit, newdata=testing)
confmat <- confusionMatrix(predict, testing$classe)
confmat
```

```{r}
plot(confmat$table, col=confmat$byClass, main=paste("Random Forest - Accuracy=", round(confmat$overall['Accuracy'],4)))
```

###2. Decision Trees
```{r}
set.seed(36363)
modFitDT <- rpart(classe ~ ., data=Training, method="class")
fancyRpartPlot(modFitDT)
```

```{r}
predictDT <- predict(modFitDT, newdata=testing, type="class")
confmatDT <- confusionMatrix(predictDT, testing$classe)
confmatDT
```

```{r}
plot(confmatDT$table, col=confmatDT$byClass, main=paste("Decision Tree - Accuracy=", round(confmatDT$overall['Accuracy'],4)))
```


###3. Generalized Boosted Model
```{r}
set.seed(36363)
controlGBM <- trainControl(method="repeatedcv", number=5, repeats=1)
modFitGBM <- train(classe ~., data=Training, method="gbm", trControl=controlGBM, verbose=FALSE)
modFitGBM$finalModel
```

```{r}
predictGBM <- predict(modFitGBM, newdata=testing)
confmatGBM <- confusionMatrix(predictGBM, testing$classe)
confmatGBM
```

```{r}
plot(confmatGBM$table, col=confmatGBM$byClass, main=paste("GBM - Accuracy=", round(confmatGBM$overall['Accuracy'],4)))
```

###Result
The accuracy
- Random Forest:
- Decision Tree:
- GBM:

```{r}
Result <- predict(modFit, newdata=testing)
Result
```
