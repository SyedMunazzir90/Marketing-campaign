"""
Logging is essential for debugging. This file is used for logging every important step of the application

File Name : logger.py
Written By: Syed Munazzir Ahmed
Version: 1.0
Revisions: None

"""

from datetime import datetime

class App_Logger:
    def __init__(self):
        pass

    def log(self, file_object, log_message):
        self.now = datetime.now()
        self.date = self.now.date()
        self.current_time = self.now.strftime("%H:%M:%S")
        file_object.write(
            str(self.date) + "/" + str(self.current_time) + "\t\t" + log_message +"\n")