# Salary_Prediction
### Salary Prediction based on job description
## Project Summary
This project will predict the salaries based on job descriptions. There are different factors on which salary depends like education of person, number of years of experience in industry, Designation in the company, type of industry, how many miles a person lives away from city etc.
This project will help the Candidates to estimate their current salary and to predict new salary if they want to switch. On the other part, It will also help the companies in deciding the salaries of new hires.

## 1. Introduction
The training data is split into two CSV files, one with the features variables and other with the target variable. The test feature file contains the Jobid's and other features for which we need to predict the salaries. Following are the features and their descriptions:

* JobId: The Id of job. It is unique for every employee.
* companyid: The Id of company.
* JobType: The designation in the company.
* degree: Highest degree of a Employee.
* major: The field or subject on which, employee had completed his/her degree from university.
* industry: The field of company like Health, Finance, Oil etc.
* YearsofExperience: Years of Experiene of an Employee in the job.
* milesFromMetropolis: The distance in miles, the employee lives away from his/her office.
* salary: This is the target variable. It is the amount each employee receives.

## 2. Descriptive statistics

The data set consists of 10k rows and 9 columns with the following information:

![alt text](https://github.com/samitsingh/Salary_Prediction/blob/master/reports/figures/Training%20Dataset.PNG "Train Data")

### Feature Distribution

* Numerical features:- **yearsExperience** and **milesFromMetropolis** are evenly distributed within their respective ranges, whereas **salary** has close to normal distribution.

![alt text](https://github.com/samitsingh/Salary_Prediction/blob/master/reports/figures/Histogram.PNG "Histogram")

* Categorical features:-

![alt text](https://github.com/samitsingh/Salary_Prediction/blob/master/reports/figures/boxen_plot_jobtype_degree.PNG "Boxen plot :- Jobtype,degree vs Salary")

**Jobtype**:- 
The CEO position has highest salary, whereas Janitor has lowest. The salary increase rapidly according to level of jobType.

**Degree**:- 
The Doctoral degree has highest salary, whereas None has lowest.


![alt text](https://github.com/samitsingh/Salary_Prediction/blob/master/reports/figures/boxen_plot_major_industry.PNG "Boxen plot :- Major,Indutry vs Salary")

**Major**:- 
Average salaries of different majors are comparatively similar. The Engineering has highest salary whereas None has lowest

**Industry**:- 
The Finance & Oil are highest paying industries.
Based on above statistics of Box plot, we observed that there are some outliers are present in salary column i.e minimum value is zero.  



* Correlation Martix

![alt text](https://github.com/samitsingh/Salary_Prediction/blob/master/reports/figures/Correlation%20Matrix.PNG "Correlation Matrix")

According to above correlation matrix, mean salary of **JobType** column is highly correlated with **salary**, followed by **degree**, **major**, **industry** and **Years of Experience**. The **Mile from Metropolis** is negatively correlated to **salary** i.e salary decrease with increase in miles.


## 3. Data pre-processing

### Feature engineering

Label encoding is applied to transform categorical features into numeric features. To enhance the performance of the models, following new features have been created by grouping Jobtype, degree, major, industry and companyId columns:

* group_mean   :- Average salary of the group.
* group_median :- Median salary of the group.
* group_max    :- Maximum salary of the group.
* group_min    :- Minimum salary of the group.
* group_mad    :- Mean absolute deviation of group.

## 4. Model development

### Evaluation Metric

To measure the effectiveness of the models that we are going to develop, we have used *mean squared error (MSE)* as evaluation metric.

### Baseline model

For any baseline model, We can calculate the Mean of target, which is a good estimate. In this project, we had alreay calculated 'group_mean' of target. So, we have used the same for our baseline model.

The mean MSE error obtained is :- *644.256325*

### Best model selection

For this regression task, 3 different machine learning algorithms were selected, one linear and two ensembles, to see which performs better for the problem:

* Linear Regressor.
* Extra Tree Regressor.
* LightGBM Regressor.

We have tuned both ensemble models and following results have been achieved:

| Model                 | MSE           |
| ----------------------|:-------------:|
| Linear Regressor      | 358.148420    |
| Extra Tree Regressor  | 316.868358    |
| LightGBM Regressor    | 307.094494    |

The lowest MSE error was returned by *LightGBM Regressor*.
Once the best model was fitted with the train data, the feature importance generated by the model are following:

![alt text](https://github.com/samitsingh/Salary_Prediction/blob/master/reports/figures/Feature_Importances.PNG "Feature Importance")

From above figure, we can conclude that *Miles from metropolis* and *Years of Experience* strongly influence salary as people get compensated more if they live closer to the city and have more years of experience, vice-versa.

## 5. Conclusion

We can conclude that we have developed a model with MSE of *307.094494*, to predict the salary based on the features given and newly generated features.

Salary varies according to the following

* Salary decreases linearly with miles away from city
* Salary increases linearly with years of experience
* Job position: CEO > CTO, CFO > VP > Manager > Senior > Junior > Janitor
* Oil and finance industries are the highest paying sectors, while service and education are the lowest paying.

We can further try to improve the model by generating new features from 'yearsofExperience' and 'Milesfrom Metropolis' columns.

### Final thoughts and recommendations

Based on the analyses we can recommend that years of experience and location highly influence the salary, and *Oil* and *Finance* industries have highest salary even for entry level positions.

Based on all of this information, companies can estimate the salary of new hire considering all factors, also candidates can decide the type of industry, location etc. to achieve the desired salary. 
