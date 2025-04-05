# Health Insurance Premium Price Prediction
1. Introduction
Objective
The objective of this project is to develop a predictive model to estimate health insurance premiums based on various demographic and health-related factors such as age, BMI, smoking habits, number of children, and region. By analyzing these variables, we aim to understand their impact on insurance charges and build an accurate regression model for premium prediction.
Dataset
The dataset used for this project contains the following features:
•	Age: Age of the individual in years.
•	Sex: Gender of the insured person (Male/Female).
•	BMI (Body Mass Index): A measure of body fat based on height and weight.
•	Children: Number of dependents covered by the insurance.
•	Smoker: Whether the individual smokes (Yes/No).
•	Region: The region where the insured person lives (Northeast, Northwest, Southeast, Southwest).
•	Charges: The total insurance premium paid by the individual (Target variable).
2. Data Collection and Preprocessing
Data Exploration
•	Loaded the dataset and checked for missing values. No missing values were found.
•	Analyzed the distribution of categorical variables using count plots.
•	Examined numerical variables using histograms and box plots to detect outliers.
•	Found that BMI and charges had some skewness, indicating the need for potential transformations.
Handling Categorical Data
•	Applied One Hot Encoding to convert categorical variables (sex, smoker, region) into numerical format.
•	Created separate binary columns for each category to ensure proper model interpretation.
Feature Scaling
•	Checked the scale of different numerical variables.
•	Since Linear Regression models are not significantly affected by feature scaling, no normalization was applied.
Multicollinearity Check
•	Used Variance Inflation Factor (VIF) to detect multicollinearity among features.
•	All predictors had VIF <= 5, confirming that there was no significant multicollinearity in the dataset.
3. Data Visualization and Analysis
Correlation Analysis
•	Plotted a correlation heatmap to observe relationships between independent variables and the target variable.
•	Found that age, BMI, and smoking status had a strong correlation with charges.
Scatter Plot Matrix
•	Visualized the relationships between numerical predictors using scatter plot matrices.
•	Observed a positive trend between age and charges, as well as BMI and charges.
Individual Feature Analysis
•	Age vs Charges: Older individuals tend to have higher premiums.
•	BMI vs Charges: Higher BMI values often lead to increased charges, indicating a possible link to health risks.
•	Smoker vs Charges: Smokers pay significantly higher premiums compared to non-smokers.
•	Region vs Charges: No significant difference in charges across regions.
4. Model Training and Evaluation
Linear Regression
•	Implemented a Linear Regression model to predict insurance charges.
•	Split the dataset into training (80%) and testing (20%) sets.
•	Trained the model using the training dataset and evaluated performance on the test dataset.
Model Interpretation using Statsmodels
•	Used Statsmodels API to analyze the regression coefficients and p-values.
•	Found that sex and region were statistically insignificant predictors based on high p-values.
•	The regression equation derived from the model helps in estimating premiums for new individuals.
Model Performance Metrics
•	Mean Absolute Error (MAE): Measures the average absolute difference between predicted and actual values.
•	Mean Squared Error (MSE): Measures the average squared difference between predictions and actual values.
•	R-squared Score: Indicates how well the model explains the variability in insurance charges.
•	Root Mean Squared Error (RMSE): Provides an estimate of prediction error in the same unit as the target variable.
•	Adjusted R-squared Score: Adjusts the R-squared score based on the number of predictors, providing a better estimate of model performance when multiple features are used.
Lasso Regression
•	Implemented Lasso Regression to handle potential overfitting and feature selection.
•	Lasso penalizes less significant coefficients, making the model more interpretable.
•	Evaluated performance using the same test set and compared results with Linear Regression.
•	Lasso Regression also helped in feature selection by reducing coefficients of less important predictors to zero.
5. Results and Key Findings
•	The Linear Regression model provided reasonable accuracy in predicting insurance charges.
•	The Lasso Regression model slightly improved interpretability by reducing the impact of less significant features.
•	Features such as age, BMI, and smoking status were the most influential factors in determining insurance charges.
•	Sex and region were found to have minimal impact on premium prediction.
•	The model achieved:
o	R-squared Score: ~0.75, indicating that 75% of the variation in charges is explained by the model.
o	RMSE: A measure of prediction accuracy, showing how far predictions deviate from actual values on average.
o	MAE: Lower values indicate better prediction accuracy.
6. Future Improvements
•	Experiment with more advanced models such as Ridge Regression, Decision Trees, or Random Forests.
•	Apply hyperparameter tuning to improve model accuracy.
•	Collect additional data to improve model generalization, especially for underrepresented categories.
•	Investigate interaction effects between features to improve prediction accuracy.
•	Explore deep learning models, such as neural networks, for improved prediction capabilities.
•	Perform feature engineering to create new informative features that could enhance model performance.
7. Conclusion
This project successfully developed a predictive model for health insurance premiums using linear regression techniques. The analysis showed that age, BMI, and smoking status play a crucial role in determining insurance costs. Future enhancements could involve using non-linear models and optimizing feature selection for better accuracy. Additionally, implementing ensemble learning techniques may further improve predictive performance.

