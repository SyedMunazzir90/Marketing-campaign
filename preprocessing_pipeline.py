"""
This file is invoked from main.py.

* It invokes preprocessor_campaign.py for pre processing the raw input dataset 'campaign.csv' 
* It invokes preprocessor_mortgage.py for pre processing the raw input dataset 'mortgage.csv' 

File Name : preprocessing_pipeline.py
Written By: Syed Munazzir Ahmed
Version: 1.0
Revisions: None
"""

from application_logging import logger
from preprocessor_campaign import campaign_preprocessor_class
from preprocessor_mortgage import mortgage_preprocessor_class
from cluster_creation import cluster_class
from model_build import model_build_class
from predict_target_from_model import predict_target_class
import configurations as config

intermediate_data_path = config.intermediate_data_path


class pre_processor_class:

    def __init__(self,path):
        self.raw_data = path
        self.file_object = open("training_logs/pre_processing_log.txt", 'a+')
        self.log_writer = logger.App_Logger()
    
    def pre_process_campaign_data(self):
        try:
            message = 'Preprocess - pre_process_campaign_data() begin!'
            self.log_writer.log(self.file_object,message)
            
            campaign_obj = campaign_preprocessor_class(self.raw_data,self.log_writer)                  #Object initialisation
            campaign_result = campaign_obj.preprocessing_campaign()                                    #calling preprocessing_campaign function
    
            lr_obj = model_build_class(self.raw_data,intermediate_data_path,self.log_writer)           #Object initialisation
            lr_obj_result = lr_obj.logistic_regression()                                               #calling logistic_regression

            rfc_obj = model_build_class(self.raw_data,intermediate_data_path,self.log_writer)          #Object initialisation
            rfc_obj_result = rfc_obj.random_forest()                                                   #calling random_forest

            predict_obj = predict_target_class(self.raw_data,intermediate_data_path,self.log_writer)   #Object initialisation
            predict_obj_result = predict_obj.predict_target()                                          #calling impute_target

            message = 'Returning to main function from pre_process_campaign_data()'
            return message

        except Exception as e:
            raise e

    def pre_process_mortgage_data(self):
        try:
            message = 'Preprocess - pre_process_mortgage_data() begin!'
            self.log_writer.log(self.file_object,message)
            
            mortgage_obj = mortgage_preprocessor_class(self.raw_data,self.log_writer)              #Object initialisation
            mortgage_result = mortgage_obj.preprocessing_mortgage()                                #calling pre_process_mortgage_data function

            message='Cluster - create_clusters() begin!'
            self.log_writer.log(self.file_object,message)

            cluster_obj = cluster_class(self.raw_data,intermediate_data_path,self.log_writer)      #Object initialisation
            cluster_result = cluster_obj.create_clusters()                                         #calling create_clusters function


            #message = 'Returning to main function from pre_process_mortgage_data()'
            message=''
            return message

        except Exception as e:
            raise e