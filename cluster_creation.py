"""
This files is invoked from main.py.

* It reads mortgage_df_cleaned.csv and applies KMeans to create clusters 
* Output is stored in post_processed_data folder

File Name : cluster_creation.py
Written By: Syed Munazzir Ahmed
Version: 1.0
Revisions: None
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import configurations as config

post_processed_data_path = config.post_processed_data_path

class cluster_class:

    def __init__(self,raw_data,path,log_writer):

        self.raw_data = raw_data
        self.intermediate_data = path
        self.file_object = open("training_logs/pre_processing_log.txt", 'a+')
        self.log_writer = log_writer
 
    def create_clusters(self):

        try:

            message="Clustering:  create_clusters() begin!"
            self.log_writer.log(self.file_object,message)

            #Reading the cleaned data file
            message='Clustering: Reading the cleaned file'
            self.log_writer.log(self.file_object,message)

            mortgage_load_cleaned_df = pd.read_csv(self.intermediate_data + 'mortgage_df_cleaned.csv')
            mortgage_df_cleaned = mortgage_load_cleaned_df.copy()

            #Reading the raw data file
            message='Clustering: Reading the raw file'
            self.log_writer.log(self.file_object,message)

            mortgage_load_df = pd.read_csv(self.raw_data)
            #mortgage_df = mortgage_load_df.copy()

            columns_for_kmeans = ['new_mortgage','sex', 'town_freq_encode','religion_freq_encode',
                              'race_freq_encode','native_country_freq_encode',
                              'workclass_freq_encode','demographic_characteristic_scaled','age_scaled',
                             'salary_in_gbp_scaled','employement_duration_years_scaled']

            kmeans_df = mortgage_df_cleaned[columns_for_kmeans]


            #Apply KMeans
            message='Clustering: Apply KMeans'
            self.log_writer.log(self.file_object,message)

            kmeans = KMeans(n_clusters = 3, max_iter = 50, random_state = 1)
            kmeans.fit(kmeans_df)

            kmeans_df.loc[:, 'Cluster Id - KMeans'] = kmeans.labels_
            
            cluster_analysis = pd.concat([mortgage_df_cleaned[['full_name','age','religion','workclass','demographic_characteristic','relationship','race','workclass','town']],
                              kmeans_df[['Cluster Id - KMeans']]],axis=1)

            #Assign cluster numbers
            message='Clustering: Assign cluster numbers after KMeans'
            self.log_writer.log(self.file_object,message)

            cluster_0 = cluster_analysis[cluster_analysis['Cluster Id - KMeans'] == 0]
            cluster_1 = cluster_analysis[cluster_analysis['Cluster Id - KMeans'] == 1]
            cluster_2 = cluster_analysis[cluster_analysis['Cluster Id - KMeans'] == 2]

            
            #Storing the clusters as seperate files for further analysis
            message='Clustering: Storing 3 clusters in post_processed_data_path'
            self.log_writer.log(self.file_object,message)

            cluster_0.to_csv(post_processed_data_path + 'cluster_0.csv')
            cluster_1.to_csv(post_processed_data_path + 'cluster_1.csv')
            cluster_2.to_csv(post_processed_data_path + 'cluster_2.csv')

            message = '-------------------------------------------------------------'
            self.log_writer.log(self.file_object,message)
            
            return True
        

        except Exception as e:
            raise e