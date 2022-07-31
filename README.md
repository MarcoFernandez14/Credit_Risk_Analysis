# Credit_Risk_Analysis
#### Environment
Code editor: Jupyter Notebook    
Language: Python    
Libraries: numpy, pandas, scikit-learn, imblearn  

## Overview
A lending service company wants to use machine learning to predict credit risk in order to provide a quicker and more reliable lending experience. This company is also interested in the accurate identification of good candidates for loans.  
The analysis performed utilizes several machine learning models and techniques such as re-sampling and boosting. The models performance is evaluated to see how accurate the predictions are.


## Results  
#### Naive Random Oversampling    
![RandomOverSampler BAS ](https://github.com/MarcoFernandez14/Credit_Risk_Analysis/blob/main/Resources/RandomOverSampler%20BAS%20.png)  
![RandomOverSampler PRE REC ](https://github.com/MarcoFernandez14/Credit_Risk_Analysis/blob/main/Resources/BalancedRandomForestClassifier%20PRE%20REC%20.png)  
* Balanced accuracy score: 65%
* Precision: The precision is low for high_risk loans and is high for low_risk loans.
* Recall: high_risk = 68% , low_risk = 63%

#### SMOTE Oversampling    
![SMOTE BAS ](https://github.com/MarcoFernandez14/Credit_Risk_Analysis/blob/main/Resources/SMOTE%20BAS%20.png)  
![SMOTE PRE REC](https://github.com/MarcoFernandez14/Credit_Risk_Analysis/blob/main/Resources/SMOTE%20PRE%20REC.png)  
* Balanced accuracy score: 64%
* Precision: The precision is low for high_risk loans and is high for low_risk loans.
* Recall: high_risk = 60% , low_risk = 67%

#### Undersampling   
![ClusterCentroids BAS ](https://github.com/MarcoFernandez14/Credit_Risk_Analysis/blob/main/Resources/ClusterCentroids%20BAS%20.png)  
![ClusterCentroids PRE REC](https://github.com/MarcoFernandez14/Credit_Risk_Analysis/blob/main/Resources/ClusterCentroids%20PRE%20REC.png)  
* Balanced accuracy score: 55%
* Precision: The precision is low for high_risk loans and is high for low_risk loans.
* Recall: high_risk = 42% , low_risk = 67%

#### Combination (Over and Under) Sampling    
![SMOTEENN BAS](https://github.com/MarcoFernandez14/Credit_Risk_Analysis/blob/main/Resources/SMOTEENN%20BAS.png)  
![SMOTEENN PRE REC](https://github.com/MarcoFernandez14/Credit_Risk_Analysis/blob/main/Resources/SMOTEENN%20PRE%20REC.png)  
* Balanced accuracy score: 65%
* Precision: The precision is low for high_risk loans and is high for low_risk loans.
* Recall: high_risk = 74% , low_risk = 57%

#### Easy Ensemble AdaBoost Classifier   
![EasyEnsembleClassifier BAS ](https://github.com/MarcoFernandez14/Credit_Risk_Analysis/blob/main/Resources/EasyEnsembleClassifier%20BAS%20.png)  
![EasyEnsembleClassifier PRE REC ](https://github.com/MarcoFernandez14/Credit_Risk_Analysis/blob/main/Resources/EasyEnsembleClassifier%20PRE%20REC%20.png) 
* Balanced accuracy score: 93%
* Precision: The precision is low for high_risk loans and is high for low_risk loans.
* Recall: high_risk = 91% , low_risk = 94%

#### Balanced Random Forest Classifier    
![BalancedRandomForestClassifier BAS ](https://github.com/MarcoFernandez14/Credit_Risk_Analysis/blob/main/Resources/BalancedRandomForestClassifier%20BAS%20.png)  
![BalancedRandomForestClassifier PRE REC ](https://github.com/MarcoFernandez14/Credit_Risk_Analysis/blob/main/Resources/BalancedRandomForestClassifier%20PRE%20REC%20.png)  
* Balanced accuracy score: 79%
* Precision: The precision is low for high_risk loans and is high for low_risk loans.
* Recall: high_risk = 67% , low_risk = 91%

## Summary
In terms of balanced accuracy, Easy Ensemble AdaBoost Classifier seems to the best model (none of the other models is above 80%).  
The precision is similar in all models: low for high_risk loans and high for low_risk loans.  
The recall indicator varies from model to model and it shows high percentages in Easy Ensemble AdaBoost Classifier.
Considering the balance accuracy and recall scores, Easy Ensemble AdaBoost Classifier is the recommended machine learning model for this case.

