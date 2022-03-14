"""
This file is invoked from preprocessing_pipeline.py.

* It is used for pre processing the raw input dataset 'campaign.csv'
* It stores the resultant file is 'campaign_df_cleaned.csv' 

File Name : preprocessor_campaign.py
Written By: Syed Munazzir Ahmed
Version: 1.0
Revisions: None
"""

import numpy as np
import pandas as pd
import configurations as config
import category_encoders as ce
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
from pandas_profiling import ProfileReport

post_processed_data_path = config.post_processed_data_path
intermediate_data_path = config.intermediate_data_path
reports_path = config.reports_path

class campaign_preprocessor_class:

    def __init__(self,path,log_writer):

        self.raw_data = path
        self.file_object = open("training_logs/pre_processing_log.txt", 'a+')
        self.log_writer = log_writer
    
    def preprocessing_campaign(self):

        try:

            message="preprocessing_campaign: Preprocess - preprocessing_campaign() begin!"
            self.log_writer.log(self.file_object,message)
            
            #Reading the raw data file
            campaign_load_df = pd.read_csv(self.raw_data)
            campaign_df = campaign_load_df.copy()
            message = 'preprocessing_campaign: Reading the raw data file'
            self.log_writer.log(self.file_object,message)

            #Generate the profile
            profile = ProfileReport(campaign_df)
            profile.to_file(output_file=reports_path + 'campaign_report.html')
            message = 'preprocessing_campaign: Profile generated!'
            self.log_writer.log(self.file_object,message)

            #Creating bins on 'age' and applying OneHotEncoding
            message = 'preprocessing_campaign: Creating bins on "age" and apply OneHotEncoding'
            self.log_writer.log(self.file_object,message)

            campaign_df['age_by_decade'] = pd.cut(x=campaign_df['age'], bins=[10, 20, 29, 39, 49, 59, 69, 79, 89], labels=['teens','20s', '30s', '40s', '50s', '60s', '70s', '80s'])
            ohe_age = OneHotEncoder(sparse=False)
            X = campaign_df['age_by_decade']
            age_by_decade_df = pd.DataFrame(ohe_age.fit_transform(np.array(X).reshape(-1, 1)))
            age_by_decade_df.columns = ['age_by_decade_20s','age_by_decade_30s','age_by_decade_40s','age_by_decade_50s','age_by_decade_60s','age_by_decade_70s','age_by_decade_80s','age_by_decade_90s','teens']
            age_by_decade_df.drop('age_by_decade_90s',axis =1, inplace=True)
            campaign_df = pd.concat([campaign_df,age_by_decade_df],axis=1)

            #Apply LabelEncoder on 'postcode'
            message = 'preprocessing_campaign: LabelEncoder on "postcode"'
            self.log_writer.log(self.file_object,message)

            postcode_list = campaign_df['postcode'].tolist()
            le_postcode = LabelEncoder()
            le_postcode.fit(postcode_list)
            campaign_df['postcode_encoded'] = pd.DataFrame(le_postcode.transform(postcode_list))

            #Apply OneHotEncoding on 'marital_status'
            message = 'preprocessing_campaign: Apply OneHotEncoding on "marital_status"'
            self.log_writer.log(self.file_object,message)

            ohe_marital_status = OneHotEncoder(sparse=False)
            X = campaign_df['marital_status']
            marital_status_df = pd.DataFrame(ohe_marital_status.fit_transform(np.array(X).reshape(-1, 1)))
            marital_status_df.columns = ['marital_status_divorced','marital_status_married-AF-spouse','marital_status_married-civ-spouse','marital_status_married-spouse-absent','marital_status_never-married','marital_status_separated','marital_status_widowed']
            marital_status_df.drop('marital_status_married-AF-spouse',axis =1, inplace=True)
            campaign_df = pd.concat([campaign_df,marital_status_df],axis=1)

            #Apply OneHotEncoding on 'education'
            message = 'preprocessing_campaign: Apply OrdinalEncoder on "education"'
            self.log_writer.log(self.file_object,message)
            ohe_education = OrdinalEncoder(categories = [['Preschool','1st-4th','5th-6th','7th-8th','9th','10th','11th','12th','HS-grad','Some-college','Bachelors','Masters','Assoc-voc','Assoc-acdm','Prof-school','Doctorate']])
            X = campaign_df['education']
            education_df = pd.DataFrame(ohe_education.fit_transform(np.array(X).reshape(-1,1)))
            education_df.columns = ['education_encoded']
            campaign_df = pd.concat([campaign_df,education_df],axis=1)

            #Apply LabelEncoder on 'job_title'
            message = 'preprocessing_campaign: Apply LabelEncoder on "job_title"'
            self.log_writer.log(self.file_object,message)
            job_title = campaign_df['job_title'].tolist()
            le_job_title = LabelEncoder()
            le_job_title.fit(job_title)
            campaign_df['job_title_encoded'] = pd.DataFrame(le_job_title.transform(job_title))

            #Converting 'created_account' to numeric
            substitute = {'Yes' : 1, 'No' : 0}
            campaign_df['created_account'] = campaign_df['created_account'].map(substitute)

            #Fill missing values of name_title with blank and create full_name
            #campaign_df['name_title']= campaign_df['name_title'].fillna(' ')
            #campaign_df['full_name'] = campaign_df['name_title'] + ' ' + campaign_df['first_name'] + ' ' + campaign_df['last_name']
            #message = 'Fill missing values of "name_title" with blank and create "full_name"'
            #self.log_writer.log(self.file_object,message)

            #Dropping other columns
            message = "preprocessing_campaign: Dropping other columns ['participant_id', 'age', 'name_title', 'first_name', 'last_name','postcode','marital_status', 'education','age_by_decade','job_title','company_email']"
            self.log_writer.log(self.file_object,message)
            campaign_df.drop(['participant_id', 'age', 'name_title', 'first_name', 'last_name','postcode','marital_status', 'education','age_by_decade','job_title','company_email'], axis = 1, inplace = True)

            message = 'preprocessing_campaign: campaign_df_cleaned.csv created'
            self.log_writer.log(self.file_object,message)
            campaign_df.to_csv(intermediate_data_path + '/campaign_df_cleaned.csv')

            message = 'preprocessing_campaign - preprocessing_campaign() end!'
            self.log_writer.log(self.file_object,message)
            
            message = '-------------------------------------------------------------'
            self.log_writer.log(self.file_object,message)
            
            return True

        except Exception as e:
            raise e