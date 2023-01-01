# Titanic_kaggle

## What is it?

The Titanic problem on Kaggle is a machine learning problem that involves predicting whether or not a passenger on the Titanic survived the shipwreck. The data for this problem is taken from the passenger list of the Titanic, which includes information such as the passenger's age, gender, class, and fare paid. The goal of the problem is to use this information to build a model that can accurately predict whether or not a passenger survived the disaster. This problem is often used as a benchmark for machine learning algorithms and is a popular choice for those learning how to build and evaluate machine learning models.

## CODE - MLP Titanic

The goal of this code is to find the optimal parameters for a multi-layer perceptron (MLP) classifier to predict survival outcomes for passengers on the Titanic. The dataset used for training and testing consists of information on passenger class, gender, age, number of siblings/spouses on board, number of parents/children on board, fare paid, and port of embarkation.

The data is first read in and any rows with missing values are removed. The input and output variables are then defined for both the training and testing sets. The input variables are those that are believed to have the most relevance to the survival outcome, while the output variable is the survival outcome itself.

The code then iterates through a range of hidden layer sizes and active functions to train the MLP and evaluate its performance using k fold cross-validation, k=5. The model with the highest accuracy score is saved and used to make predictions on the test set. The code also scales the input and output variables using the MinMaxScaler to ensure that all variables are on the same scale, between -1 and 1.

## CODE - User_Titanic

This code uses the best parameters (highest accuracy) found in the previous code called Mlp Titanic to train a multilayer perceptron model using the same data from the train.csv file. The input data for the model includes variables such as passenger class, gender, age, number of siblings and spouse, number of parents and children, ticket fare, and port of embarkation. The model is then used to predict whether a hypothetical passenger, whose information is entered by the user, would survive or not. The code also includes a validation step that only allows the user to enter valid values for the input variables.

## Knowledges used in both codes:
1. Pandas library for data manipulation and analysis
2. Numpy library for numerical computation
3. MinMaxScaler function from sklearn.preprocessing for data scaling
4. MLPClassifier class from sklearn.neural_network for creating a multilayer perceptron model
5. Cross_validate function from sklearn.model_selection for model evaluation using cross validation
6. Accuracy_score function from sklearn.metrics for calculating model accuracy
7. Warnings library for suppressing warnings during model training
8. Reading and concatenating data from csv files using pandas
9. Data preprocessing and cleaning, including dropping rows with missing values and selecting specific columns
10. Separating data into inputs and outputs for model training and testing
11. Creating and training a multilayer perceptron model using the MLPClassifier class
12. Evaluating the trained model using k fold cross validation and calculating its accuracy
13. Generating combinations of hyperparameters for model training and selecting the best performing model
14. Predicting survival outcomes for test data using the trained model
15. Validating user input to ensure it is within the acceptable range for model input
16. Predicting survival outcomes for test data using the trained model

