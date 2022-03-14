"""
This file is invoked from preprocessing_pipeline.py.

* It is used for pre processing the raw input dataset 'mortgage.csv'
* It stores the resultant file is 'mortgage_preprocessed.csv' 

File Name : preprocessor_mortgage.py
Written By: Syed Munazzir Ahmed
Version: 1.0
Revisions: None
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import datetime
from datetime import date
from datetime import timedelta
import category_encoders as ce
import configurations as config
from pandas_profiling import ProfileReport
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder, StandardScaler

post_processed_data_path = config.post_processed_data_path
intermediate_data_path = config.intermediate_data_path
reports_path = config.reports_path

class mortgage_preprocessor_class:

    def __init__(self,path,log_writer):

        self.raw_data = path
        self.file_object = open("training_logs/pre_processing_log.txt", 'a+')
        self.log_writer = log_writer
 
    def preprocessing_mortgage(self):

        try:

            #Reading the raw data file
            mortgage_load_df = pd.read_csv(self.raw_data)
            mortgage_df = mortgage_load_df.copy()
            message = 'preprocessing_mortgage: Reading the raw data file'
            self.log_writer.log(self.file_object,message)


            #Generate the profile
            profile = ProfileReport(mortgage_df)
            profile.to_file(output_file=reports_path + 'mortgage_report.html')
            message = 'preprocessing_mortgage: Profile generated!'
            self.log_writer.log(self.file_object,message)


            #Drop 'full_name'            
            #mortgage_df.drop('full_name', axis=1, inplace = True)
            #message = 'preprocessing_mortgage: Drop "full_name"'
            #self.log_writer.log(self.file_object,message)


            #Convert 'dob' to datetime
            message = 'preprocessing_mortgage: Calculate "age" from "dob" and apply sqrt transformation'
            self.log_writer.log(self.file_object,message)

            today = date.today()
            year = today.year
            year = 2018

            mortgage_df['dob'] = pd.to_datetime(mortgage_df['dob'])
            mortgage_df['age'] = year - pd.DatetimeIndex(mortgage_df['dob']).year 
            mortgage_df.drop('dob', axis=1, inplace = True)

            upper_cap = round(mortgage_df['age'].quantile(.99),2)
            mortgage_df['age'] = np.where(mortgage_df['age']>upper_cap,upper_cap,mortgage_df.age)
            mortgage_df['age_deskewed'] = np.sqrt(mortgage_df['age'])
          
            #Frequency encode 'town'
            message = 'preprocessing_mortgage: Frequency encode town'
            self.log_writer.log(self.file_object,message)

            freq = mortgage_df.groupby('town').size()/len(mortgage_df)  
            mortgage_df.loc[:, "{}_freq_encode".format('town')] = mortgage_df['town'].map(freq)  

            #'salary_band' preprocessing. Use data from currency conversion API and convert into GBP 
            message = 'preprocessing_mortgage: convert salary to base currency GBP'
            self.log_writer.log(self.file_object,message)

            mortgage_df['salary_band_new'] = mortgage_df['salary_band'].astype(str).str.lower()
            mortgage_df['salary_band_new'].replace({' yearly': '', '£':''}, regex=True, inplace=True)
            mortgage_df['salary_band_new'].replace({' pw': ' * 52', ' per month':' * 12'}, regex=True, inplace=True)
            mortgage_df['salary_band_new'].replace({' range': '', ' - ': ' + '}, regex=True, inplace=True)
            mortgage_df['salary_band_new'].replace({'^[a-z]+$': 0}, regex=True, inplace=True)
            mortgage_df.loc[mortgage_df.astype(str).salary_band_new.str.contains('\+'),'salary_band_new'] = '(' + mortgage_df['salary_band_new'].astype(str) + ') / 2'
            mortgage_df['currency'] = mortgage_df['salary_band_new'].str[-3:]            
            mortgage_df['currency'] = mortgage_df['currency'].fillna('gbp')
            mortgage_df['currency'].replace({'[^a-z]+$': 'gbp'}, regex=True, inplace=True)
            mortgage_df['salary_band_new'].replace({'[a-z]+$': ''}, regex=True, inplace=True)
            mortgage_df['salary_band_new'] = mortgage_df['salary_band_new'].astype(str)
            mortgage_df['salary_band_new'] = mortgage_df['salary_band_new'].apply(lambda x: eval(x) if (pd.notnull(x)) else x)

            currency_conversion = pd.read_csv(intermediate_data_path + 'currency_conversion.csv')
            currency_conversion.drop('Unnamed: 0',axis = 1,inplace=True)    

            mortgage_df = pd.merge(mortgage_df,currency_conversion,on='currency',how='inner')
            mortgage_df = mortgage_df.assign(salary_in_gbp= lambda x:(x['salary_band_new'] / x['rate']))
            mortgage_df.drop(['salary_band','salary_band_new','currency','rate'],axis=1,inplace=True)

            upper_cap = round(mortgage_df['salary_in_gbp'].quantile(0.92),2)

            mortgage_df['salary_in_gbp'] = np.where(mortgage_df['salary_in_gbp']>upper_cap,upper_cap,mortgage_df.salary_in_gbp)
            mortgage_df['salary_in_gbp'].to_csv('salary.csv')
            mortgage_df['salary_in_gbp_deskewed'] = np.sqrt(mortgage_df['salary_in_gbp'])

            message = 'preprocessing_mortgage: Drop paye'
            self.log_writer.log(self.file_object,message)
            mortgage_df.drop('paye',axis=1,inplace=True)

            #Convert capital_gain and capital_loss into net_profit
            message = 'preprocessing_mortgage: calculate net_profit from capital_gain and capital_loss and OneHotEncode'
            self.log_writer.log(self.file_object,message)  

            mortgage_df['net_profit'] = mortgage_df['capital_gain'] - mortgage_df['capital_loss']
            mortgage_df.drop(['capital_gain','capital_loss'],axis=1,inplace=True)

            upper_cap = round(mortgage_df['net_profit'].quantile(0.97),2)
            mortgage_df['net_profit'] = np.where(mortgage_df['net_profit']>upper_cap,upper_cap,mortgage_df.net_profit)
            mortgage_df['net_profit_deskewed'] = np.sqrt(mortgage_df['net_profit'])

            #Cap outliers and apply sqrt employement_duration_years
            message = 'preprocessing_mortgage: Cap outliers and apply sqrt on "employement_duration_years"'
            self.log_writer.log(self.file_object,message)

            mortgage_df['new_mortgage'] = 1
            mortgage_df['employement_duration_years'] =  round(((mortgage_df['years_with_employer'] * 12) + (mortgage_df['months_with_employer']))/12,2)
            mortgage_df.drop(['years_with_employer','months_with_employer'],axis=1,inplace=True)

            upper_cap = round(mortgage_df['employement_duration_years'].quantile(0.97),2)
            mortgage_df['employement_duration_years'] = np.where(mortgage_df['employement_duration_years']>upper_cap,upper_cap,mortgage_df.employement_duration_years)
            mortgage_df['employement_duration_years_deskewed'] = np.sqrt(mortgage_df['employement_duration_years'])


            #Cap outliers and apply sqrt on hours_per_week
            message = 'preprocessing_mortgage: Cap outliers and apply sqrt on "hours_per_week"'
            self.log_writer.log(self.file_object,message)

            upper_cap = round(mortgage_df['hours_per_week'].quantile(0.97),2)
            mortgage_df['hours_per_week'] = np.where(mortgage_df['hours_per_week']>upper_cap,upper_cap,mortgage_df.hours_per_week)


            #OneHotEncode gender
            message = 'preprocessing_mortgage: OneHotEncode "gender"'
            self.log_writer.log(self.file_object,message)

            substitute = {'Male' : 1, 'Female' : 0}
            mortgage_df['sex'] = mortgage_df['sex'].map(substitute)

            #Cap outliers and apply sqrt on 'demographic_characteristic'
            message = 'preprocessing_mortgage: Cap outliers and apply sqrt on  "demographic_characteristic"'
            self.log_writer.log(self.file_object,message)  

            upper_cap = round(mortgage_df['demographic_characteristic'].quantile(.99),2)
            mortgage_df['demographic_characteristic'] = np.where(mortgage_df['demographic_characteristic']>upper_cap,upper_cap,mortgage_df.demographic_characteristic)
            mortgage_df['demographic_characteristic_deskewed'] = np.sqrt(mortgage_df['demographic_characteristic'])


            #Frequency encode 'religion'
            message = 'preprocessing_mortgage: Frequency encode "religion"'
            self.log_writer.log(self.file_object,message)  

            freq = mortgage_df.groupby('religion').size()/len(mortgage_df)  
            # mapping values to dataframe
            mortgage_df.loc[:, "{}_freq_encode".format('religion')] = mortgage_df['religion'].map(freq)  

            #Frequency encode 'relationship'
            message = 'preprocessing_mortgage: Frequency encode "relationship"'
            self.log_writer.log(self.file_object,message)  

            freq = mortgage_df.groupby('relationship').size()/len(mortgage_df)  
            # mapping values to dataframe
            mortgage_df.loc[:, "{}_freq_encode".format('relationship')] = mortgage_df['relationship'].map(freq)  

            #Frequency encode 'race'
            message = 'preprocessing_mortgage: Frequency encode "race"'
            self.log_writer.log(self.file_object,message)  

            freq = mortgage_df.groupby('race').size()/len(mortgage_df)  
            mortgage_df.loc[:, "{}_freq_encode".format('race')] = mortgage_df['race'].map(freq)  


            #Frequency encode 'native_country'
            message = 'preprocessing_mortgage: Frequency encode "native_country"'
            self.log_writer.log(self.file_object,message) 

            freq = mortgage_df.groupby('native_country').size()/len(mortgage_df)   
            mortgage_df.loc[:, "{}_freq_encode".format('native_country')] = mortgage_df['native_country'].map(freq)  

            #Frequency encode 'native_country'
            message = 'preprocessing_mortgage: Frequency encode "workclass"'
            self.log_writer.log(self.file_object,message) 

            freq = mortgage_df.groupby('workclass').size()/len(mortgage_df)  
            mortgage_df.loc[:, "{}_freq_encode".format('workclass')] = mortgage_df['workclass'].map(freq)  

            columns = ['hours_per_week','demographic_characteristic_deskewed','age_deskewed','salary_in_gbp_deskewed','employement_duration_years_deskewed']
            scaler = StandardScaler()
            scaled_df = scaler.fit_transform(mortgage_df[columns])
            scaled_df = pd.DataFrame(scaled_df)
            columns_scaled = ['hours_per_week_scaled','demographic_characteristic_scaled','age_scaled','salary_in_gbp_scaled','employement_duration_years_scaled']
            scaled_df.columns = columns_scaled

            mortgage_df = pd.concat([mortgage_df, scaled_df], axis=1)
            mortgage_df.to_csv(intermediate_data_path + 'mortgage_df_cleaned.csv')


            return True

        except Exception as e:
            raise e           