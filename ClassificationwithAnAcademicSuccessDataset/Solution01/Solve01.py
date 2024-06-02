# -*- coding: utf-8 -*-
# @Time    : 2024/6/2 下午3:51
# @Author  : nanji
# @Site    : https://www.kaggle.com/code/gauravduttakiit/pss4e6-flaml-roc-auc-ovo
# @File    : Solve01.py
# @Software: PyCharm 
# @Comment :

import pandas as pd

train = pd.read_csv('../input/playground-series-s4e6/train.csv')
train.head()
test = pd.read_csv('../input/playground-series-s4e6/test.csv')
test.head()
print('0' * 100)
print(train.info())
print('1' * 100)
print(train.nunique())

train = train.drop(['id'], axis=1)
print('2' * 100)
print(train.head())

test = test.drop(['id'], axis=1)
print(test.head())

print('3' * 100)
print('4' * 100)
train['Marital status'] = train['Marital status'].replace({1: 'single',
														   2: 'married',
														   3: 'widower',
														   4: 'divorced',
														   5: 'facto union',
														   6: 'legally separated'})
print(train['Marital status'].value_counts())

test['Marital status'] = test['Marital status'].replace({1: 'single',
														 2: 'married',
														 3: 'widower',
														 4: 'divorced',
														 5: 'facto union',
														 6: 'legally separated'})

print(test['Marital status'].value_counts())
train['Application mode'] = train['Application mode'].replace({
	1: '1st phase - general contingent',
	2: 'Ordinance No. 612/93',
	5: '1st phase - special contingent (Azores Island)',
	7: 'Holders of other higher courses',
	10: 'Ordinance No. 854-B/99',
	15: 'International student (bachelor)',
	16: '1st phase - special contingent (Madeira Island)',
	17: '2nd phase - general contingent',
	18: '3rd phase - general contingent',
	26: 'Ordinance No. 533-A/99, item b2) (Different Plan)',
	27: 'Ordinance No. 533-A/99, item b3 (Other Institution)',
	39: 'Over 23 years old',
	42: 'Transfer',
	43: 'Change of course',
	44: 'Technological specialization diploma holders',
	51: 'Change of institution/course',
	53: 'Short cycle diploma holders',
	57: 'Change of institution/course (International)'})
train['Application mode'].value_counts()
test['Application mode'] = test['Application mode'].replace({
	1: '1st phase - general contingent',
	2: 'Ordinance No. 612/93',
	5: '1st phase - special contingent (Azores Island)',
	7: 'Holders of other higher courses',
	10: 'Ordinance No. 854-B/99',
	15: 'International student (bachelor)',
	16: '1st phase - special contingent (Madeira Island)',
	17: '2nd phase - general contingent',
	18: '3rd phase - general contingent',
	26: 'Ordinance No. 533-A/99, item b2) (Different Plan)',
	27: 'Ordinance No. 533-A/99, item b3 (Other Institution)',
	39: 'Over 23 years old',
	42: 'Transfer',
	43: 'Change of course',
	44: 'Technological specialization diploma holders',
	51: 'Change of institution/course',
	53: 'Short cycle diploma holders',
	57: 'Change of institution/course (International)'})

r1 = test['Application mode'].value_counts()
print(r1)

r2 = train[''].value_counts()
print(r2)
train['Course'] = train['Course'].replace({
	33: 'Biofuel Production Technologies',
	171: 'Animation and Multimedia Design',
	8014: 'Social Service (evening attendance)',
	9003: 'Agronomy',
	9070: 'Communication Design',
	9085: 'Veterinary Nursing',
	9119: 'Informatics Engineering',
	9130: 'Equinculture',
	9147: 'Management',
	9238: 'Social Service',
	9254: 'Tourism',
	9500: 'Nursing',
	9556: 'Oral Hygiene',
	9670: 'Advertising and Marketing Management',
	9773: 'Journalism and Communication',
	9853: 'Basic Education',
	9991: 'Management (evening attendance)'})

print('2' * 100)
print(train['Course'].value_counts())

test['Course'] = test['Course'].replace({
	33: 'Biofuel Production Technologies',
	171: 'Animation and Multimedia Design',
	8014: 'Social Service (evening attendance)',
	9003: 'Agronomy',
	9070: 'Communication Design',
	9085: 'Veterinary Nursing',
	9119: 'Informatics Engineering',
	9130: 'Equinculture',
	9147: 'Management',
	9238: 'Social Service',
	9254: 'Tourism',
	9500: 'Nursing',
	9556: 'Oral Hygiene',
	9670: 'Advertising and Marketing Management',
	9773: 'Journalism and Communication',
	9853: 'Basic Education',
	9991: 'Management (evening attendance)'
})
print('3' * 100)
print(test['Course'].value_counts())

train['Daytime/evening attendance'] = \
	train['Daytime/evening attendance'].replace({
		1: 'daytime',
		0: 'evening'
	})
print('4' * 100)
print(train['Daytime/evening attendance'].value_counts())

test['Daytime/evening attendance'] = \
	test['Daytime/evening attendance'].replace({1: 'daytime',
												0: 'evening'})
print(test['Daytime/evening attendance'].value_counts())
train['Previous qualification'] = train['Previous qualification'].replace({
	1: 'Secondary education',
	2: "Higher education - bachelor's degree",
	3: 'Higher education - degree',
	4: "Higher education - master's",
	5: "Higher education - doctorate",
	6: "Frequency of higher education",
	9: "12th year of schooling - not completed",
	10: "11th year of schooling - not completed",
	12: "Other - 11th year of schooling",
	14: "10th year of schooling",
	15: "10th year of schooling - not completed",
	19: "Basic education 3rd cycle (9th/10th/11th year) or equiv.",
	38: "Basic education 2nd cycle (6th/7th/8th year) or equiv.",
	39: "Technological specialization course",
	40: "Higher education - degree (1st cycle)",
	42: "Professional higher technical course",
	43: "Higher education - master (2nd cycle)"
})
r5 = train['Previous qualification'].value_counts()
print(r5)

test['Previous qualification'] = \
	test['Previous qualification'].replace({
		1: 'Secondary education',
		2: "Higher education - bachelor's degree",
		3: 'Higher education - degree',
		4: "Higher education - master's",
		5: "Higher education - doctorate",
		6: "Frequency of higher education",
		9: "12th year of schooling - not completed",
		10: "11th year of schooling - not completed",
		12: "Other - 11th year of schooling",
		14: "10th year of schooling",
		15: "10th year of schooling - not completed",
		19: "Basic education 3rd cycle (9th/10th/11th year) or equiv.",
		38: "Basic education 2nd cycle (6th/7th/8th year) or equiv.",
		39: "Technological specialization course",
		40: "Higher education - degree (1st cycle)",
		42: "Professional higher technical course",
		43: "Higher education - master (2nd cycle)"})
r6 = test['Previous qualification'].value_counts()
print(r6)

r7 = train['Previous qualification (grade)'].describe()

r8 = test['Previous qualification (grade)'].describe()

train['Nacionality'] = train['Nacionality'].replace({
	1: 'Portuguese',
	2: 'German',
	6: 'Spanish',
	11: 'Italian',
	13: 'Dutch',
	14: 'English',
	17: 'Lithuanian',
	21: 'Angolan',
	22: 'Cape Verdean',
	24: 'Guinean',
	25: 'Mozambican',
	26: 'Santomean',
	32: 'Turkish',
	41: 'Brazilian',
	62: 'Romanian',
	100: 'Moldova (Republic of)',
	101: 'Mexican',
	103: 'Ukrainian',
	105: 'Russian',
	108: 'Cuban',
	109: 'Colombian'})

print('8' * 100)
r8 = train['Nacionality'].value_counts()

test['Nacionality'] = test['Nacionality'].replace({
	1: 'Portuguese',
	2: 'German',
	6: 'Spanish',
	11: 'Italian',
	13: 'Dutch',
	14: 'English',
	17: 'Lithuanian',
	21: 'Angolan',
	22: 'Cape Verdean',
	24: 'Guinean',
	25: 'Mozambican',
	26: 'Santomean',
	32: 'Turkish',
	41: 'Brazilian',
	62: 'Romanian',
	100: 'Moldova (Republic of)',
	101: 'Mexican',
	103: 'Ukrainian',
	105: 'Russian',
	108: 'Cuban',
	109: 'Colombian'})
r9 = test['Nacionality'].value_counts()
print('9' * 100)
print(r9)

train = train.rename({'Nacionality': 'Nationality'}, axis='columns')
print('0' * 100)
print(train.info())
test = test.rename({'Nacionality': 'Nationality'}, axis='columns')
print(test.info())
print('1' * 100)

train["Mother's qualification"] = train["Mother's qualification"].replace({
	1: "Secondary Education - 12th Year of Schooling or Eq.",
	2: "Higher Education - Bachelor's Degree",
	3: "Higher Education - Degree",
	4: "Higher Education - Master's",
	5: "Higher Education - Doctorate",
	6: "Frequency of Higher Education",
	9: "12th Year of Schooling - Not Completed",
	10: "11th Year of Schooling - Not Completed",
	11: "7th Year (Old)",
	12: "Other - 11th Year of Schooling",
	14: "10th Year of Schooling",
	18: "General commerce course",
	19: "Basic Education 3rd Cycle (9th/10th/11th Year) or Equiv.",
	22: "Technical-professional course",
	26: "7th year of schooling",
	27: "2nd cycle of the general high school course",
	29: "9th Year of Schooling - Not Completed",
	30: "8th year of schooling",
	34: "Unknown",
	35: "Can't read or write",
	36: "Can read without having a 4th year of schooling",
	37: "Basic education 1st cycle (4th/5th year) or equiv.",
	38: "Basic Education 2nd Cycle (6th/7th/8th Year) or Equiv.",
	39: "Technological specialization course",
	40: "Higher education - degree (1st cycle)",
	41: "Specialized higher studies course",
	42: "Professional higher technical course",
	43: "Higher Education - Master (2nd cycle)",
	44: "Higher Education - Doctorate (3rd cycle)"})
train["Mother's qualification"].value_counts()
