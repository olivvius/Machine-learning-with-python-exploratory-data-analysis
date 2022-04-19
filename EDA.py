import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_row', 111)
pd.set_option('display.max_column', 111)
url = 'https://raw.githubusercontent.com/MachineLearnia/Python-Machine-Learning/master/Dataset/dataset.csv'
data = pd.read_csv(url, encoding = "ISO-8859-1")
df = data.copy()
df.dtypes.value_counts()
#plt.figure(figsize=(20,10))
#sns.heatmap(df.isna(), cbar=False)
#Napercent=(df.isna().sum()/df.shape[0]).sort_values(ascending=True)
df.dropna(thresh=0.1*len(df),axis=1, inplace=True)
#df = df[df.columns[df.isna().sum()/df.shape[0] <0.9]]
shape=df.shape
df = df.drop('Patient ID', axis=1)
#identification target: 10 % of positive case
comptepos=df['SARS-Cov-2 exam result'].value_counts(normalize=True)
# for col in df.select_dtypes('float'):
#     plt.figure()
#     sns.distplot(df[col])
#sns.distplot(df['Patient age quantile'], bins=20)
#qualitatives variables 
# for col in df.select_dtypes('object'):
#     print(f'{col :-<50} {df[col].unique()}')

#2 interesting : viral & rhinovirus

#variables - target relation
positive_df = df[df['SARS-Cov-2 exam result'] == 'positive']
negative_df = df[df['SARS-Cov-2 exam result'] == 'negative']
missing_rate = df.isna().sum()/df.shape[0]
blood_columns = df.columns[(missing_rate < 0.9) & (missing_rate >0.88)]
viral_columns = df.columns[(missing_rate < 0.88) & (missing_rate > 0.75)]

#relations blood blood
# for col in blood_columns:
#     plt.figure()
#     sns.distplot(positive_df[col], label='positive')
#     sns.distplot(negative_df[col], label='negative')
#     plt.legend()
#on note des differences sur les platelets, monocytes; leukocytes

# bloodcolumns relation
sns.countplot(x='Patient age quantile', hue='SARS-Cov-2 exam result', data=df)

#quantitative viral data analysis

crosstableau=pd.crosstab(df['SARS-Cov-2 exam result'], df['Influenza A'])
#afined analysis
# for col in viral_columns:
#     plt.figure()
#     sns.heatmap(pd.crosstab(df['SARS-Cov-2 exam result'], df[col]), annot=True, fmt='d')


# variables - variables relation with pairplot
#sns.pairplot(df[blood_columns])
#sns.heatmap(df[blood_columns].corr())
#sns.clustermap(df[blood_columns].corr())
#strong correlation between some variables

# for col in blood_columns:
#     plt.figure()
#     sns.lmplot(x='Patient age quantile', y=col, hue='SARS-Cov-2 exam result', data=df)

agecorrel=df.corr()['Patient age quantile'].sort_values()
#age not correlated

pd.crosstab(df['Influenza A'], df['Influenza A, rapid test'])
#influenza not trustworthy

# dissease-blood relation
# 'detected'column or covid
df['est malade'] = np.sum(df[viral_columns[:-2]] == 'detected', axis=1) >=1
malade_df = df[df['est malade'] == True]
non_malade_df = df[df['est malade'] == False]
# for col in blood_columns:
#     plt.figure()
#     sns.distplot(malade_df[col], label='malade')
#     sns.distplot(non_malade_df[col], label='non malade')
#     plt.legend()
    
def hospitalisation(df):
    if df['Patient addmited to regular ward (1=yes, 0=no)'] == 1:
        return 'surveillance'
    elif df['Patient addmited to semi-intensive unit (1=yes, 0=no)'] == 1:
        return 'soins semi-intensives'
    elif df['Patient addmited to intensive care unit (1=yes, 0=no)'] == 1:
        return 'soins intensifs'
    else:
        return 'inconnu'
df['statut'] = df.apply(hospitalisation, axis=1)  

# for col in blood_columns:
#     plt.figure()
#     for cat in df['statut'].unique():
#         sns.distplot(df[df['statut']==cat][col], label=cat)
#     plt.legend()
    






















