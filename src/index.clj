(ns index)

;# Assignment 4: Decision Trees, Boosting, Bagging
;
;Instructions
;
;Load the Concrete Compressive Strength Data Set from [UCI](https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength)
;
;* Use the built-in functions within sklearn to apply the following algorithms and perform analysis using 5-Fold Cross Validation with RMSE as the error function:
  ;* Decision Tree Regression
    ;* What is the best method for choosing the number of max features: Auto, Square Root, or Log2?
    ;* What is the best value for ccp_alpha (the pruning parameter) from the range of 0-1? How deep is the tree with each alpha value?
    ;* Graphically display the best tree found and list the feature importance associated with it. What variables are most important? What is the Root Node?
  ;* Random Forest Regression
    ;* Perform all steps completed for Decision Tree Regression, but don't graph anything.
    ;* What is the optimal number of estimators, up to 100?
    ;* List the feature importance associated with the final model. What variables are most important?
;
;* Modify the "README.md" file to  include the following sections:
  ;* Questions (Answer these):
    ;* Which model performed the best in terms of RMSE?
  ;* Make sure that your README file is formatted properly and is visually appealing. It should be free of grammatical errors, punctuation errors, capitalization issues, etc. Sentences should be complete.
