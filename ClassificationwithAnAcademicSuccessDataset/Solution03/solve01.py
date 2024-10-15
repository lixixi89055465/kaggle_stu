# -*- coding: utf-8 -*-
# @Time : 2024/10/13 14:04
# @Author : nanji
# @Site : https://www.kaggle.com/code/satyaprakashshukl/h2o-automl-academic-performance
# @File : solve01.py
# @Software: PyCharm 
# @Comment : H2O👩🏿‍🎓 Automl🎓 Academic📈Performance
import h2o
import pandas as pd
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, StandardScaler
# import h2o
# from h2o.automl import H2OAutoML
from itertools import combinations
from scipy.stats import gmean, hmean
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np

h2o.init()

# Load test Data
df_train = pd.read_csv('../input/playground-series-s4e6/train.csv')
# Train Data
df_test = pd.read_csv('../input/playground-series-s4e6/test.csv')
df_sub = pd.read_csv('../input/playground-series-s4e6/sample_submission.csv')
# print(df_train.shape)
# print('1' * 100)
# print(df_sub.shape)
# print('2' * 100)
# print(df_train.columns)
# print('3' * 100)
# print(df_train.head())
df_train.drop(columns=['id'], inplace=True)
df_test.drop(columns=['id'], inplace=True)
# print('4' * 100)
# print(df_train.describe())

df_train_excluded = df_train.drop(columns=['Course'])
desc = df_train_excluded.describe()
desc_t = desc.transpose()
# desc_t = desc_t.drop(index='Course')
# fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(15, 20))
# for ax, metric in zip(axes.flatten(), desc_t.columns):
#     ax.barh(desc_t.index, desc_t[metric], color='skyblue')
#     ax.set_title(metric)
#     ax.set_xlabel(metric)
#     ax.set_ylabel('Columns')
# plt.tight_layout()
# plt.show()

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
df_train['Target'] = label_encoder.fit_transform(df_train['Target'])

import seaborn as sns

correlation_matrix = df_train.corr()
# plt.figure(figsize=(20, 15))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".1f", linewidths=0.5)
# plt.xticks(rotation=90)
# plt.yticks(rotation=0)
# plt.title('Correlation Matrix')
# plt.show()

highly_correlated = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) >= 0.7:
            highly_correlated.append((correlation_matrix.index[i], correlation_matrix.columns[j]))

less_correlated = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) <= 0.3:
            less_correlated.append((correlation_matrix.index[i], correlation_matrix.columns[j]))

target_column = 'Target'

correlation_with_target = df_train.corr()[target_column].sort_values(ascending=False)
# print("Correlation with Target:>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
# print(correlation_with_target)

# plt.figure(figsize=(10, 8))
# correlation_with_target.plot(kind='bar', color='skyblue')
# plt.title('Correlation with Target Variable')
# plt.xlabel('Features')
# plt.ylabel('Correlation Coefficient')
# plt.xticks(rotation=90)
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.show()

import autoviz
from autoviz.AutoViz_Class import AutoViz_Class

AV = AutoViz_Class()
# print('AV:')
# print(AV)
dftc = AV.AutoViz(df_train)
# print('dftc:')
# print(dftc)

df_train = pd.read_csv('../input/interstellar-data/training.csv')
df_test = pd.read_csv('../input/interstellar-data/testing.csv')
# print('df_train.shape:')
# print(df_train.shape)
# print('df_test.shape:')
# print(df_test.shape)
# print('df_train.head():')
# print(df_train.head())
# print('df_test.head():')
# print(df_test.head())

from scipy.stats import skew


def handle_skewed_columns(df):
    numerical_features = df.select_dtypes(include=[np.number])
    skewness = numerical_features.apply(lambda x: skew(x.dropna()))
    skewed_features = skewness[abs(skewness) > 1]
    for col in skewed_features.index:
        if df[col].min() > -1:
            df[f'{col}_log'] = np.log1p(df[col])
    return df


def add_interaction_features(df, columns):
    interaction_df = pd.DataFrame()
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            col1, col2 = columns[i], columns[j]
            interaction_df[f'{col1}_plus_{col2}'] = df[col1] + df[col2]
            interaction_df[f'{col1}_minus_{col2}'] = df[col1] - df[col2]
            interaction_df[f'{col1}_times_{col2}'] = df[col1] * df[col2]
            interaction_df[f'{col1}_div_{col2}'] = df[col1] / (df[col1] + df[col2])
    return interaction_df


def add_curriculum_interaction_features(df):
    df['Curriculum_interaction_1st_sem'] = df['Curricular units 1st sem (credited)'] * df[
        'Curricular units 1st sem (enrolled)'] * df['Curricular units 1st sem (evaluations)'] * df[
                                               'Curricular units 1st sem (approved)']
    df['Curriculum_interaction_2nd_sem'] = df['Curricular units 2nd sem (credited)'] * df[
        'Curricular units 2nd sem (enrolled)'] * df['Curricular units 2nd sem (evaluations)'] * df[
                                               'Curricular units 2nd sem (approved)']


def add_grade_interaction_features(df):
    df['Grade_interaction_1st_sem'] = df['Admission grade'] * df['Curricular units 1st sem (grade)']
    df['Grade_interaction_2nd_sem'] = df['Admission grade'] * df['Curricular units 2nd sem (grade)']


def add_age_interaction_features(df):
    df['Age_interaction'] = df['Age at enrollment'] * df['Previous qualification (grade)']


def add_economic_interaction_features(df):
    df['Economic_interaction'] = df['Unemployment rate'] * df['Inflation rate'] * df['GDP']


def add_qualification_interaction_features(df):
    df['Qualification_interaction'] = df['Previous qualification'] * df['Admission grade']
    df['Course_interaction'] = df['Previous qualification'] * df['Course']


def add_occupation_interaction_features(df):
    df["Occupation_interaction"] = df["Mother's occupation"] * df["Father's occupation"]


def add_enrollment_evaluation_interaction_features(df):
    df['Enrollment_evaluation_interaction_1st_sem'] = (
            df['Curricular units 1st sem (enrolled)'] *
            df['Curricular units 1st sem (evaluations)'])
    df['Enrollment_evaluation_interaction_2nd_sem'] = df['Curricular units 2nd sem (enrolled)'] * df[
        'Curricular units 2nd sem (evaluations)']


def add_gender_marital_interaction_features(df):
    df['Gender_marital_interaction'] = df['Gender'] * df['Marital status']


def add_tuition_scholarship_interaction_features(df):
    df['Tuition_scholarship_interaction'] = df['Tuition fees up to date'] * df['Scholarship holder']


def feature_engineer_train(df_train):
    df_train = handle_skewed_columns(df_train)
    economic_cols = ['Unemployment rate', 'Inflation rate', 'GDP']
    relationship_cols = ["Mother's qualification"]
    economic_interactions = add_interaction_features(df_train, economic_cols)
    relationship_interactions = add_interaction_features(df_train, relationship_cols)
    df_train = pd.concat([df_train, economic_interactions, relationship_interactions], axis=1)
    add_curriculum_interaction_features(df_train)
    add_grade_interaction_features(df_train)
    add_age_interaction_features(df_train)
    add_economic_interaction_features(df_train)
    add_qualification_interaction_features(df_train)
    add_occupation_interaction_features(df_train)
    add_enrollment_evaluation_interaction_features(df_train)
    add_gender_marital_interaction_features(df_train)
    add_tuition_scholarship_interaction_features(df_train)
    return df_train

    # numerical_features = df.select_dtypes(include=[np.number])
    # skewness = numerical_features.apply(lambda x: skew(x.dropna()))
    # skewed_features = skewness[abs(skewness) > 1]
    # for col in skewed_features.index:
    #     if df[col].min() > -1:
    #         df[f'{col}_log'] = np.log1p(df[col])
    # return df


def feature_engineer_test(df_test):
    df_test = handle_skewed_columns(df_test)
    economic_cols = ['Unemployment rate', 'Inflation rate', 'GDP']
    relationship_cols = ["Mother's qualification"]
    economic_interactions = add_interaction_features(df_test, economic_cols)
    relationship_interactions = add_interaction_features(df_test, relationship_cols)
    df_test = pd.concat([df_test, economic_interactions, relationship_interactions], axis=1)
    add_curriculum_interaction_features(df_test)
    add_grade_interaction_features(df_test)
    add_age_interaction_features(df_test)
    add_economic_interaction_features(df_test)
    add_qualification_interaction_features(df_test)
    add_occupation_interaction_features(df_test)
    add_enrollment_evaluation_interaction_features(df_test)
    add_gender_marital_interaction_features(df_test)
    add_tuition_scholarship_interaction_features(df_test)
    return df_test


df_train = feature_engineer_train(df_train)
df_test = feature_engineer_test(df_test)


# print(df_train.head())

def add_interaction_features(df):
    df['Application_mode_x_Application_order'] = df['Application mode'] * df['Application order']
    df['Course_x_Curricular_units_1st_sem_enrolled'] = df['Course'] * df['Curricular units 1st sem (enrolled)']
    df['Daytime_evening_attendance_x_Age_at_enrollment'] = df['Daytime/evening attendance'] * df['Age at enrollment']
    df['Previous_qualification_grade_x_Admission_grade'] = df['Previous qualification (grade)'] * df['Admission grade']
    df['Displaced_x_Curricular_units_1st_sem_approved'] = df['Displaced'] * df['Curricular units 1st sem (approved)']
    df['Scholarship_holder_x_Tuition_fees_up_to_date'] = df['Scholarship holder'] * df['Tuition fees up to date']
    df['Curricular_units_1st_sem_approved_x_Curricular_units_2nd_sem_approved'] = df[
                                                                                      'Curricular units 1st sem (approved)'] * \
                                                                                  df[
                                                                                      'Curricular units 2nd sem (approved)']
    df['Unemployment_rate_x_Inflation_rate'] = df['Unemployment rate'] * df['Inflation rate']
    return df


df_train = add_interaction_features(df_train)
df_test = add_interaction_features(df_test)
print('1' * 100)
print(df_train.shape)
print(df_test.shape)
print('2' * 100)

print(df_train.sample(4))

df_train['Target']

print(df_test.sample(4))


def find_duplicate_columns(df):
    """
       Find duplicate columns in a dataframe.

       Args:
       df (pd.DataFrame): The dataframe to check for duplicate columns.

       Returns:
       list: A list of lists, where each sublist contains the names of duplicate columns.
       """
    duplicate_columns = []
    columns = df.columns
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            if df[columns[i]].equals(df[columns[j]]):
                duplicate_columns.append((columns[i], columns[j]))
    return duplicate_columns


duplicate_columns_train = find_duplicate_columns(df_train)
columns_to_drop_train = list(set([col for pair in duplicate_columns_train for col in pair[1:]]))
df_train = df_train.drop(columns=columns_to_drop_train)

duplicate_columns_test = find_duplicate_columns(df_test)
columns_to_drop_test = list(set([col for pair in duplicate_columns_test for col in pair[1:]]))
df_test = df_test.drop(columns=columns_to_drop_test)
print('3' * 100)
print(df_train.shape)
print(df_test.shape)

from h2o.automl import H2OAutoML

train_data = h2o.H2OFrame(df_train)
aml = H2OAutoML(max_runtime_secs=8000, seed=42)
aml.train(y='Target', training_frame=train_data)
# What leadboard has to say
leaderboard = aml.leaderboard
print(leaderboard)
best_model = aml.leader
print(best_model)
