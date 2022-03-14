"""
This file is invoked from preprocessing_pipeline.py.
* It is used to predict the target variable of the cleaned campaign dataset 'campaign_target_predict_df.csv'

File Name : predict_target_from_model.py
Written By: Syed Munazzir Ahmed
Version: 1.0
Revisions: None
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn import metrics
from sklearn import model_selection
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import configurations as config
from sklearn.metrics import confusion_matrix, roc_auc_score
from matplotlib import pyplot as plt
import seaborn as sns
import pickle
from sklearn.pipeline import make_pipeline, Pipeline

post_processed_data_path = config.post_processed_data_path
figures_path = config.figures_path
models_path = config.models_path


class predict_target_class:

    def __init__(self,raw_data,path,log_writer):

        self.raw_data = raw_data
        self.intermediate_data = path
        self.file_object = open("prediction_logs/model_predict_log.txt", 'a+')
        self.log_writer = log_writer
 
    def predict_target(self):

        try:

            message="Predict target:  predict_target() begin!"
            self.log_writer.log(self.file_object,message)

            #Reading the cleaned data file
            message='Predict target: Reading the raw data file'
            self.log_writer.log(self.file_object,message)

            #Reading the raw data file
            campaign_load_df = pd.read_csv(self.raw_data)
            available_df = campaign_load_df[campaign_load_df['created_account'].notnull()]
            target_df = campaign_load_df[campaign_load_df['created_account'].isnull()]
            
            message='Predict target: Fetching missing target values'
            self.log_writer.log(self.file_object,message)

            target_df_copy = target_df.copy()

            #Creating bins on 'age' and applying OneHotEncoding
            message = 'Predict target: Creating bins on "age" and apply OneHotEncoding'
            self.log_writer.log(self.file_object,message)

            target_df['age_by_decade'] = pd.cut(x=target_df['age'], bins=[10, 20, 29, 39, 49, 59, 69, 79, 89, 100], labels=['teens','20s', '30s', '40s', '50s', '60s', '70s', '80s', '90s'])
            ohe = OneHotEncoder(sparse=False)
            X = target_df['age_by_decade']
            age_by_decade_df = pd.DataFrame(ohe.fit_transform(np.array(X).reshape(-1, 1)))
            age_by_decade_df.columns = ['age_by_decade_20s','age_by_decade_30s','age_by_decade_40s','age_by_decade_50s','age_by_decade_60s','age_by_decade_70s','age_by_decade_80s','age_by_decade_90s','teens']
            age_by_decade_df.drop(['age_by_decade_90s'],axis =1, inplace=True)            
            target_df = pd.concat([target_df.reset_index(drop=True),age_by_decade_df.reset_index(drop=True)], axis=1)
            target_df.drop('age',axis=1,inplace=True)

            #Apply LabelEncoder on 'postcode'
            message = 'Predict target: LabelEncoder on "postcode"'
            self.log_writer.log(self.file_object,message)

            postcode_list = target_df['postcode'].tolist()
            le_postcode = LabelEncoder()
            le_postcode.fit(postcode_list)
            target_df['postcode_encoded'] = pd.DataFrame(le_postcode.transform(postcode_list))
            target_df.drop('postcode',axis=1,inplace= True)


            #Apply OneHotEncoding on 'marital_status'
            message = 'Predict target: Apply OneHotEncoding on "marital_status"'
            self.log_writer.log(self.file_object,message)

            ohe_marital_status = OneHotEncoder(sparse=False)
            X = target_df['marital_status']
            marital_status_df = pd.DataFrame(ohe_marital_status.fit_transform(np.array(X).reshape(-1, 1)))
            marital_status_df.columns = ['marital_status_divorced','marital_status_married-AF-spouse','marital_status_married-civ-spouse','marital_status_married-spouse-absent','marital_status_never-married','marital_status_separated','marital_status_widowed']
            marital_status_df.drop('marital_status_married-AF-spouse',axis =1, inplace=True)
            target_df = pd.concat([target_df,marital_status_df],axis=1)
            target_df.drop('marital_status',axis=1,inplace=True)


            #Apply OneHotEncoding on 'education'
            message = 'Predict target: Apply OrdinalEncoder on "education"'
            self.log_writer.log(self.file_object,message)

            ohe_education = OrdinalEncoder(categories = [['Preschool','1st-4th','5th-6th','7th-8th','9th','10th','11th','12th','HS-grad','Some-college','Bachelors','Masters','Assoc-voc','Assoc-acdm','Prof-school','Doctorate']])
            X = target_df['education']
            education_df = pd.DataFrame(ohe_education.fit_transform(np.array(X).reshape(-1,1)))
            education_df.columns = ['education_encoded']

            target_df = pd.concat([target_df,education_df],axis=1)
            target_df.drop('education',axis=1,inplace = True)


            #Apply LabelEncoder on 'job_title'
            message = 'Predict target: Apply LabelEncoder on "job_title"'
            self.log_writer.log(self.file_object,message)
            job_title = target_df['job_title'].tolist()

            le_job_title = LabelEncoder()
            le_job_title.fit(job_title)
            target_df['job_title_encoded'] = pd.DataFrame(le_job_title.transform(job_title))

            #Filter the required columns for prediction
            message = 'Predict target: Filter columns for prediction'
            self.log_writer.log(self.file_object,message)

            columns = ['occupation_level', 'education_num', 'familiarity_FB', 'view_FB',
                       'interested_insurance', 'age_by_decade_20s', 'age_by_decade_30s',
                       'age_by_decade_40s', 'age_by_decade_50s', 'age_by_decade_60s',
                       'age_by_decade_70s', 'age_by_decade_80s', 'teens', 'postcode_encoded',
                       'marital_status_divorced', 'marital_status_married-civ-spouse',
                       'marital_status_married-spouse-absent', 'marital_status_never-married',
                       'marital_status_separated', 'marital_status_widowed',
                       'education_encoded', 'job_title_encoded']

            target_df = target_df[columns]
            

            #Load model
            message = 'Predict target: load model'
            self.log_writer.log(self.file_object,message)

            model = pickle.load(open(models_path + 'random_forests_model.pkl', 'rb'))

            #Make predictions
            message = 'Predict target: make predictions'
            self.log_writer.log(self.file_object,message)

            predictions=pd.DataFrame(model.predict(target_df))
            predictions.columns = ['created_account']

            substitute = {0:'No',1:'Yes'}
            predictions['created_account'] = predictions['created_account'].map(substitute)

            #Concatenate predictions and target isnull dataframes           
            target_df_copy.drop('created_account',inplace=True,axis=1)
            final_prediction = pd.concat([target_df_copy.reset_index(drop=True),predictions.reset_index(drop=True)], axis=1)
 
            message = 'Predict target: campaign_target_predict_df.csv created in post_processed_data_path'
            self.log_writer.log(self.file_object,message)           
            campaign_target_predict_df = pd.concat([available_df.reset_index(drop=True),final_prediction.reset_index(drop=True)], axis=0)
            campaign_target_predict_df.to_csv(post_processed_data_path + 'campaign_target_predict_df.csv')

            message="Predict target:  predict_target() ends!"
            self.log_writer.log(self.file_object,message)
            
            message = '-------------------------------------------------------------'
            self.log_writer.log(self.file_object,message)


        except Exception as e:
            raise e