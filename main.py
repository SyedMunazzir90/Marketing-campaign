"""
This is the initiation point of the application. 

File Name : main.py
Written By: Syed Munazzir Ahmed
Version: 1.0
Revisions: None
"""

import os
import configurations as config
from preprocessing_pipeline import pre_processor_class
from application_logging import logger

raw_data_campaign_path = config.raw_data_campaign_path
raw_data_mortgage_path = config.raw_data_mortgage_path

def Run_All_Pipeline():

    file_object = open("training_logs/pre_processing_log.txt", 'a+')             #Logging
    log_writer = logger.App_Logger()

    mortgage_obj = pre_processor_class(raw_data_mortgage_path)                   #Object initialisation
    mortgage_result = mortgage_obj.pre_process_mortgage_data()                   #calling pre_process_mortgage_data function
    log_writer.log(file_object,mortgage_result)

    campaign_obj = pre_processor_class(raw_data_campaign_path)                   #Object initialisation
    campaign_result = campaign_obj.pre_process_campaign_data()                   #calling pre_process_campaign_data function
    log_writer.log(file_object,campaign_result)

if __name__ == '__main__':
    Run_All_Pipeline()