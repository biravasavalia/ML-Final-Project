## Online Shopper Behavior Analysis and Prediction Using Machine Learning
### By Birava Savalia


#### Introduction

The dataset contains Online Shoppers Intention, and the part of project aims to analyze online shopper's behavioral patterns, derive valuable insights from important metrics like page values, exit rates, and bounce rates, and develop predictive models to determine whether a shopping session results in a purchase. The businesses can use this to boost conversion rates, enhance customer satisfaction, and optimize their websites.

#### Introduction to Data

The dataset consists of feature vectors belonging to 12,330 sessions. The dataset was formed so that each session would belong to a different user in a 1-year period to avoid any tendency to a specific campaign, special day, user profile, or period.

18 Features:
Administrative: It is the number of administrative pages that the user visited.
Administrative_Duration: Its is the amount of time spent for an administrative page by a user.
Informational: It is the number of informational pages that the user visited.
Informational_Duration: It is the amount of time spent for informational pages by a user.
ProductRelat: It is the number of product related pages that the user visited.
ProductRelated_Duration: It is the amount of time spent for product related pages by a user.
BounceRates: The percentage of visitors who enter the page and exit without trigger.
ExitRates: The percentage of pageviews by the user on the website and that has ended.
PageValues: The average value of the page by the value of the target page in order to complete the eCommerce transaction.
SpecialDay: This value represents the closeness of the browsing date to special days or holidays.
Month: it is the month when pageview occurred, in string form.
OperatingSystems: It is an integer value representing the operating system of the user.
Browser: An integer value representing the browser that the user used.
Region: An integer value representing the region where the user is located.
TrafficType: An integer value representing the user categorization.
VisitorType: A string represents either New Visitor or Returning Visitor or Other.
Weekend: A boolean representing if the session was on a weekend.
Revenue: A boolean representing if the user completed the purchase.

#### Visual Analysis

This chart compares the monthly counts of those who made purchases (orange) versus those who didn’t (blue). May had the highest activity overall, with 2,999 non-purchases and 365 purchases. November and December showed a significant increase in purchases (760 and 216, respectively), likely due to seasonal factors like holidays. Across all months, non-purchases consistently outnumber purchases, indicating room to improve conversion rates. Lower activity is evident in January, July, and October, suggesting these months might benefit from targeted campaigns.



#### Preprocessing

The Online Shoppers Intention dataset consists of 12,330 records and 18 features, both numerical and categorical. Before modeling, several preprocessing steps were undertaken to ensure the data was clean, consistent, and suitable for predictive analysis:

1. Missing Values: Each feature was inspected for missing values. Although the dataset appeared complete, exploratory data analysis confirmed no significant gaps that would impact modeling.
   
2. Normalization and Scaling: Since the dataset includes numerical features like Page Values, Exit Rates, and Bounce Rates, these were normalized to bring them onto a comparable scale. Standardization was applied using z-scores to improve the performance of algorithms sensitive to feature scaling.

3. Categorical Variable Encoding: Categorical features such as VisitorType and Month were converted into numerical representations using one-hot encoding. This step enabled the machine learning algorithms to process these variables effectively.

4. Splitting Data: The dataset was divided into training (80%) and testing (20%) subsets to evaluate the model's performance on unseen data. This division ensured the results reflected the generalizability of the models.

Visualizations, such as box plots and histograms, were generated to identify any outliers or imbalances in the data. By addressing these preprocessing tasks, the dataset was transformed into a robust format for machine learning.

#### Methods

Two machine learning models, Logistic Regression and Random Forest, were selected for this project to predict whether a shopping session ends in a purchase:

1. Logistic Regression: 
This linear model was chosen for its simplicity, interpretability, and speed. Logistic Regression is well-suited for binary classification tasks like this one, as it estimates the probability of an outcome based on input features. The coefficients of the model provide insight into the importance of each feature, making it an excellent baseline.

2. Random Forest: 
Random Forest, an ensemble learning method, was selected for its ability to handle nonlinear relationships and its robustness against overfitting. This model builds multiple decision trees and averages their predictions, resulting in improved accuracy. Hyperparameter tuning was performed to optimize parameters such as the number of trees and maximum depth, aiming to enhance performance further.

Performance Metrics:
Both models were evaluated using metrics including accuracy, precision, recall, and the F1 score. These metrics ensured a comprehensive evaluation of the models, balancing the trade-off between precision and recall, especially given the potential imbalance in purchase outcomes.

Cross-Validation:
To ensure reliability, k-fold cross-validation was employed during training. This technique provided a robust estimate of model performance and minimized the risk of overfitting by validating on multiple subsets of the data.


#### Results

1. Logistic Regression: 
Logistic Regression achieved a baseline accuracy of approximately 78%. While its simplicity made it computationally efficient, it struggled to capture complex patterns in the data, leading to lower performance on recall metrics.



2. Random Forest:
The Random Forest model outperformed Logistic Regression with an accuracy of 85%. Its ability to model nonlinear relationships and emphasize feature importance contributed to superior results. Hyperparameter tuning further improved its performance, with the tuned model achieving an F1 score of 83%, indicating a strong balance between precision and recall.

The confusion matrix shows how well the Random Forest model predicts purchased and did not purchase.

- True Positive (81.18%): The model correctly predicted 3,003 people as did not purchase. 
- True Negative (7.76%): It correctly identified 287 people as purchased.  
- False Positive (7.89%): The model mistakenly predicted 292 people as purchased when they did not.  
- False Negative (3.16%): It incorrectly classified 117 people as did not purchase when they actually purchased.  

The high true positive rate indicates the model performs well for predicting did not purchase.



#### Observations:
Random Forest demonstrated a clear advantage over Logistic Regression due to its flexibility and robustness. Overfitting was mitigated by cross-validation, and feature importance analysis revealed Page Values and Exit Rates as the most predictive features. Loss and accuracy curves showed steady improvements during training, highlighting the efficacy of the Random Forest model.

Graphs of model accuracy, feature importance, and confusion matrices were generated to visualize these findings.


#### Conclusions

This project successfully applied Logistic Regression and Random Forest to predict online shopping behaviors.
Random Forest outperformed Logistic Regression, demonstrating the value of ensemble methods for complex datasets.
Feature importance analysis highlighted actionable insights, such as focusing on optimizing Page Values and Exit Rates to enhance user engagement and conversion rates.
The study reinforced the importance of preprocessing, especially feature scaling and encoding, in achieving reliable model performance.

Learnings: 
The project provided hands-on experience in data preparation, algorithm selection, and model evaluation. One limitation was the computational cost of Random Forest, which could be addressed by exploring more efficient algorithms or dimensionality reduction techniques. Future work may involve incorporating additional features, such as customer demographics, to improve predictive power further.


#### References

Géron, Aurélien. Hands-on Machine Learning with Scikit-Learn and TensorFlow Concepts, Tools, and Techniques to Build Intelligent Systems. 2nd ed., O’Reilly Media, Inc., Sept. 2019.

GeeksforGeeks. “Logistic Regression Using Python.” GeeksforGeeks, 29 Apr. 2019, www.geeksforgeeks.org/ml-logistic-regression-using-python/#

‌“Random Forest in Python and Coding It with Scikit-Learn (Tutorial).” Data36, 30 May 2022, data36.com/random-forest-in-python/

DataSet: Sakar, C. and Yomi Kastro. "Online Shoppers Purchasing Intention Dataset." UCI Machine Learning Repository, 2018, https://doi.org/10.24432/C5F88Q


#### Acknowledgments

habiburrahamanfahim. “Analyzing the Purchase Intent of Online Consumers.”
Kaggle.com, Kaggle, 22 Jan. 2024, www.kaggle.com/code/habiburrahamanfahim/analyzing-the-purchase-intent-of-online-consumers/notebook#Logistic-Regressio
TomaIjatomi. “Online-Shoppers-Intention-Prediction/Online Shoppers Intention.ipynb at Master · TomaIjatomi/Online-Shoppers-Intention-Prediction.” GitHub, 2024, github.com/TomaIjatomi/Online-shoppers-intention-prediction/blob/master/Online%20Shoppers%20Intention.ipynb
hongtaepn. “Online Shopping Intention Analytics & Prediction.” Kaggle.com, Kaggle, 6 Dec. 2023, www.kaggle.com/code/hongtaepn/online-shopping-intention-analytics-prediction
“Introduction — Online Shoppers Purchasing Intention.” Ubc-Mds.github.io, ubc-mds.github.io/online-shoppers-purchasing-intention/intro.html
tmcdonald92. “GitHub - Tmcdonald92/Online-Shoppers-Purchasing-Intention: Using Classification Models with Cross-Validation and Hyperparameter Tuning to Predict Shoppers Decision to Make Online Purchase.” GitHub, 2020, github.com/tmcdonald92/Online-Shoppers-Purchasing-Intention
