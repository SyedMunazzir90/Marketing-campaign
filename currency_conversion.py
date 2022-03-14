"""
This files is used to invoke currency rates of all currencies to GBP
* The file 'currency_conversion.csv' is created and later used for currency conversion.

File Name : currency_conversion.py
Written By: Syed Munazzir Ahmed
Version: 1.0
Revisions: None
"""

import requests
import pandas as pd
import configurations as config
from application_logging import logger

intermediate_data_path = config.intermediate_data_path

file_object = open("training_logs/pre_processing_log.txt", 'a+')                                      #Logging
log_writer = logger.App_Logger()

#GBP is the base currency
url = 'https://v6.exchangerate-api.com/v6/c2e13a4c4fd4aab6bd7a70a0/latest/GBP'

# Making our request
response = requests.get(url)

message = 'currency_conversion API invoked!'
log_writer.log(file_object,message)

data = response.json()
currency_dict = data['conversion_rates']

currency_converter_df = pd.DataFrame(list(currency_dict.items()))
currency_converter_df.columns = ['currency','rate']
currency_converter_df['currency'] = currency_converter_df['currency'].astype(str).str.lower()
currency_converter_df.to_csv(intermediate_data_path + 'currency_conversion.csv')

message = 'currency_conversion.csv created!'
log_writer.log(file_object,message)