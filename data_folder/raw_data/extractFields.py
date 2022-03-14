import pandas as pd
import numpy as np
import time
import xlrd
import json


def extractFields(data):
    
    print("hello")
    jsonArray =[]
    list = data['full_name'].tolist()
    for name in list:
        jsonObj = {}
        strList = name.split()
        print(strList)
        if any("." in s or "iss" in s for s in strList):
            print("yes")
            jsonObj['title'] = strList[0]
            jsonObj['fname'] = strList[1]
            jsonObj['lname'] = strList[len(strList)-1]
            # print(title,fname,lname)
        else:
            print("no")    
            title = ""
            jsonObj['fname'] = strList[0]
            jsonObj['lname'] = strList[len(strList)-1]
            # print(title,fname,lname)
        jsonArray.append(jsonObj)
    print(json.dumps(jsonArray))
    df_names = pd.DataFrame(jsonArray)
    final_df = pd.concat([data, df_names],axis=1)
    final_df.to_excel("adnanOut.xlsx")
    

def readExcel():
    file_name = "/Users/akarim/Documents/Automation/mortgage.xlsx"
    loc = (str(file_name))
        
    cols = [1]

    # df = pd.read_excel(loc, usecols=cols)
    df = pd.read_excel(loc)
    extractFields(df)
    
    

if __name__ == '__main__':
    # extractFields()
    readExcel()