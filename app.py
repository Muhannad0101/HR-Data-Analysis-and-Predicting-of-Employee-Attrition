import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score


df_train = pd.read_csv('data/train.csv')
df_train = df_train.drop('id', axis=1)



def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.25)
    quartile3 = dataframe[variable].quantile(0.75)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


cols_outliers = ['MonthlyRate', 'MonthlyIncome', 'NumCompaniesWorked', 'StockOptionLevel', 'TotalWorkingYears',
                 'YearsAtCompany', 'YearsSinceLastPromotion', 'YearsWithCurrManager']


def replace_with_thresholds(dataframe,columns):
    for col in columns:
        low_limit, up_limit = outlier_thresholds(dataframe, col)
        dataframe.loc[(dataframe[col] < low_limit), col] = low_limit
        dataframe.loc[(dataframe[col] > up_limit), col] = up_limit

replace_with_thresholds(df_train, df_train[cols_outliers])


cat_features = np.array([i for i in df_train.columns.tolist() if df_train[i].dtype == 'object'])
num_features = np.array([i for i in df_train.columns.tolist() if df_train[i].dtype != 'object'])

le = LabelEncoder()
for feature in cat_features:
    df_train[feature] = le.fit_transform(df_train[feature])


features = ['Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EmployeeCount', 'EnvironmentSatisfaction',
            'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome', 'MonthlyRate',
            'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction',
            'StandardHours', 'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',
            'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']

scaler = MinMaxScaler()
df_train[features] = scaler.fit_transform(df_train[features])


X = df_train.drop(['Attrition'], axis=1)
y = df_train['Attrition']
rf_clf = RandomForestClassifier(n_estimators=100, max_features='sqrt', max_depth=10, random_state=123)
rf_clf.fit(X, y)


st.title('Employee Attrition Prediction App')


age = st.slider('Age', 18, 70, 30)
business_travel = st.selectbox('BusinessTravel', ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'])
daily_rate = st.slider('Daily Rate', 100, 1500, 500)
department = st.selectbox('Department', ['Human Resources', 'Research & Development', 'Sales'])
distance_from_home = st.slider('Distance from Home', 1, 30, 10)
education = st.selectbox('Education', ['Below College', 'College', 'Bachelor', 'Master', 'Doctor'])
education_field = st.selectbox('EducationField', ['Human Resources', 
                                                  'Life Sciences', 'Marketing', 
                                                  'Medical', 'Other', 'Technical Degree'])

environment_satisfaction = st.slider('Environment Satisfaction', 1, 4, 2)
gender = st.selectbox('Gender', ['Female', 'Male'])
hourly_rate = st.slider('HourlyRate', 20, 100, 50)
job_involvement = st.slider('Job Involvement', 1, 4, 2)
job_level = st.slider('JobLevel', 1, 5, 3)
job_role = st.selectbox('JobRole', ['Healthcare Representative', 'Human Resources', 
                                    'Laboratory Technician', 'Manager', 'Manufacturing Director', 
                                    'Research Director', 'Research Scientist', 'Sales Executive', 
                                    'Sales Representative'])

job_satisfaction = st.slider('JobSatisfaction', 1, 4, 2)
marital_status = st.selectbox('MaritalStatus', ['Divorced', 'Married', 'Single'])
monthly_income = st.slider('MonthlyIncome', 1000, 20000, 5000)
monthly_rate = st.slider('MonthlyRate', 1000, 25000, 5000)
num_companies_worked = st.slider('NumCompaniesWorked', 0, 10, 5)
over_18 = st.selectbox('Over18', ['Y'])
over_time = st.selectbox('OverTime',['No', 'Yes'])
percent_salary_hike = st.slider('PercentSalaryHike', 0, 25, 12)
performance_rating = st.selectbox('PerformanceRating', [3, 4])
relationship_satisfaction = st.slider('RelationshipSatisfaction', 1, 4, 2)
stock_option_level = st.slider('StockOptionLevel', 0, 3, 1)
total_working_years = st.slider('TotalWorkingYears', 0, 40, 10)
training_times_last_year = st.slider('TrainingTimesLastYear', 0, 6, 2)
work_life_balance = st.slider('WorkLifeBalance', 1, 4, 2)
years_at_company = st.slider('YearsAtCompany', 0, 20, 5)
years_in_current_role = st.slider('YearsInCurrentRole', 0, 15, 3)
years_since_last_promotion = st.slider('YearsSinceLastPromotion', 0, 15, 3)
years_with_curr_manager = st.slider('YearsWithCurrManager', 0, 15, 3)


input_data = {'Age': age,
              'BusinessTravel': business_travel,
              'DailyRate': daily_rate,
              'Department': department,
              'DistanceFromHome': distance_from_home,
              'Education': education,
              'EducationField': education_field,
              'EnvironmentSatisfaction': environment_satisfaction,
              'Gender': gender,
              'HourlyRate': hourly_rate,
              'JobInvolvement': job_involvement,
              'JobLevel': job_level,
              'JobRole': job_role,
              'JobSatisfaction': job_satisfaction,
              'MaritalStatus': marital_status,
              'MonthlyIncome': monthly_income,
              'MonthlyRate': monthly_rate,
              'NumCompaniesWorked': num_companies_worked,
              'Over18': over_18,
              'OverTime': over_time,
              'PercentSalaryHike': percent_salary_hike,
              'PerformanceRating': performance_rating,
              'RelationshipSatisfaction': relationship_satisfaction,
              'StockOptionLevel': stock_option_level,
              'TotalWorkingYears': total_working_years,
              'TrainingTimesLastYear': training_times_last_year,
              'WorkLifeBalance': work_life_balance,
              'YearsAtCompany': years_at_company,
              'YearsInCurrentRole': years_in_current_role,
              'YearsSinceLastPromotion': years_since_last_promotion,
              'YearsWithCurrManager': years_with_curr_manager}
input_df = pd.DataFrame([input_data])


for feature in cat_features:
    input_df[feature] = le.transform(input_df[feature])


input_df[features] = scaler.transform(input_df[features])


prediction = rf_clf.predict(input_df)
if prediction[0] == 0:
    st.write('The employee is likely to stay.')
else:
    st.write('The employee is likely to leave.')

