# CustomerRevenuePrediction

In many businesses, identifying which customers will make a purchase (and when) and how much will they spend, is a critical exercise. This is true for both brick-and-mortar outlets and online stores. This project's data is website traffic data acquired from an online retailer.

> Data URL: [Kaggle Link](https://www.kaggle.com/competitions/2022-5103-hw6)

## The challenge: Predict total sales 

The data provides information on customer's website site visit behavior. Customers may visit the store multiple times, on multiple days, with or without making a purchase. The variable $revenue$ lists the amount of money that a customer spends on a given visit in the dataset. My main goal for this project is to predict how much money a customer will spend, in total, across all visits to the site, during the allotted one-year time frame (August 2016 to August 2017).

## Prediction target 

More specifically, I am predicting the transformation of the aggregate customer-level sales value based on the natural log. That is, if a customer has multiple revenue transactions, then the sum of all the revenue generated across all of the transactions, i.e.,:

$$
custRevenue_i =  \sum_{j=1}^{k_i} revenue_{ij} \ \ \  \forall i \in customers
$$

$$
\text{where } k_i \text{ denotes the number of revenue transactions for customer } i 
$$

And then transform this variable as follows:

$$
targetRevenue_i=  \ln(custRevenue_i + 1) \ \ \  \forall i \in customers
$$

## Modelling

For this project, I have used $Linear Discriminant Analysis$ and $Multivariate Adaptive Regression Splines$ to predict the following:

1.  $LDA$ for classifying the customer if they will buy something or not.
2.  $MARS$ for predicting how much they might spend, in terms of $logarithmic$ value.
