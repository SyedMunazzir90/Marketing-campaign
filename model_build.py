"""
This file is invoked from preprocessing_pipeline.py.
* It is used for building various models by using the cleaned dataset 'campaign_df_cleaned.csv'

File Name : model_build.py
Written By: Syed Munazzir Ahmed
Version: 1.0
Revisions: None
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn import metrics
from sklearn import model_selection
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
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

class model_build_class:

    def __init__(self,raw_data,path,log_writer):

        self.raw_data = raw_data
        self.intermediate_data = path
        self.file_object = open("training_logs/model_build_log.txt", 'a+')
        self.log_writer = log_writer
 
    def logistic_regression(self):

        try:

            message="Modelling Logistic Regression:  logistic_regression() begin!"
            self.log_writer.log(self.file_object,message)

            #Reading the cleaned data file
            message='Modelling Logistic Regression: Reading the cleaned file'
            self.log_writer.log(self.file_object,message)

            message='Modelling Logistic Regression: Fetching available target values'
            self.log_writer.log(self.file_object,message)

            lr_load_df = pd.read_csv(self.intermediate_data + 'campaign_df_cleaned.csv')
            available_df = lr_load_df[lr_load_df['created_account'].notnull()]

            available_scaled_df = available_df[['familiarity_FB','view_FB','occupation_level','education_num','postcode_encoded','job_title_encoded']]

            message='Modelling Logistic Regression: Apply StandardScaler'
            self.log_writer.log(self.file_object,message)

            scaler = StandardScaler()
            available_scaled_df = pd.DataFrame(scaler.fit_transform(available_scaled_df))
            available_scaled_df.columns = ['familiarity_FB_scaled','view_FB_scaled','occupation_level_scaled','education_num_scaled','postcode_encoded_scaled','job_title_encoded_scaled']

            available_df = pd.concat([available_df,available_scaled_df],axis=1)
            available_df.drop(['familiarity_FB','view_FB','occupation_level','education_num','postcode_encoded','job_title_encoded'],axis=1,inplace=True)

            message='Modelling Logistic Regression: Save campaign_df_scaled.csv'
            self.log_writer.log(self.file_object,message)
            available_df.drop(['Unnamed: 0','education_encoded'],axis=1,inplace=True)
            available_df.to_csv(self.intermediate_data + 'campaign_df_scaled.csv')

            y = available_df.pop('created_account')
            X = available_df

            message='Modelling Logistic Regression: Train test split'
            self.log_writer.log(self.file_object,message)

            X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.8,test_size=0.2,random_state=111)

            message='Modelling Logistic Regression: Apply SMOTE'
            self.log_writer.log(self.file_object,message)

            sm = SMOTE(random_state=2)
            X_train_smote, y_train_smote = sm.fit_resample(X_train, y_train.ravel())

            message='Modelling Logistic Regression: Finding best_params_ using GridSearchCV'
            self.log_writer.log(self.file_object,message)

            logistic_model = LogisticRegression(max_iter = 1500)

            #Default-Run of default-hyperparameters
            parameters = {"C": [0.01, 0.1, 1, 10, 100, 1000]}

            scorer = metrics.make_scorer(metrics.roc_auc_score,
                              greater_is_better=True,
                              needs_proba=True,
                              needs_threshold=False)

            logistic_model = model_selection.GridSearchCV(estimator=logistic_model,
                                       param_grid=parameters,
                                       n_jobs=-1,
                                       cv=3,
                                       scoring=scorer,
                                       refit=True)

            logistic_model.fit(X_train_smote, y_train_smote)
            best_params = logistic_model.best_params_

            message1='Modelling Logistic Regression: Build LogisticRegression on best_params_ :'
            message2=str(best_params)
            self.log_writer.log(self.file_object,message)

            message=message1+message2
            self.log_writer.log(self.file_object,message)

            logistic_model = LogisticRegression(C=best_params['C'])
            logistic_model.fit(X_train_smote, y_train_smote)
            predictions = logistic_model.predict(X_test)

            message='Modelling Logistic Regression: Saving confusion_matrix'
            self.log_writer.log(self.file_object,message)

            plt.figure(dpi=120)
            mat = confusion_matrix(y_test, predictions)
            sns.heatmap(mat.T, annot=True, fmt='d', cbar=False)
            plt.title('Logistic Regression - Smote')
            plt.xlabel('True label')
            plt.ylabel('Predicted label')
            plt.savefig(figures_path + 'Logistic Regression - Confusion Matrix - SMOTE')

            message='Modelling Logistic Regression: Saving classification_report'
            self.log_writer.log(self.file_object,message)

            message=classification_report(y_test,predictions)
            self.log_writer.log(self.file_object,message)

            # Predicted probability
            message='Modelling Logistic Regression: Predict and save ROC curve'
            self.log_writer.log(self.file_object,message)

            y_test_pred_proba = logistic_model.predict_proba(X_test)[:,1]
            model = 'LogisticRegression'
            
            #Plot the ROC curve
            draw_roc(y_test, y_test_pred_proba, model)

            message='Modelling Logistic Regression: Dump logistic_regression_model pickle file in models path'
            self.log_writer.log(self.file_object,message)

            filepath = models_path + 'logistic_regression_model.pkl'
            with open(filepath, "wb") as f:
                pickle.dump(logistic_model, f)

            message = '-------------------------------------------------------------'
            self.log_writer.log(self.file_object,message)

        except Exception as e:
            raise e


    def random_forest(self):

        try:

            message="Modelling Random Forests:  random_forest() begin!"
            self.log_writer.log(self.file_object,message)

            message='Modelling Random Forests: Reading the cleaned file'
            self.log_writer.log(self.file_object,message)

            rfc_load_df = pd.read_csv(self.intermediate_data + 'campaign_df_cleaned.csv')

            message='Modelling Random Forests: Fetching available target values'
            self.log_writer.log(self.file_object,message)

            available_df = rfc_load_df[rfc_load_df['created_account'].notnull()]
            available_df.drop('Unnamed: 0',axis=1,inplace=True)
            y = available_df.pop('created_account')
            X = available_df

            message='Modelling Random Forests: Train test split'
            self.log_writer.log(self.file_object,message)

            X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.8,test_size=0.2,random_state=111)

            message='Modelling Random Forests: Apply SMOTE'
            self.log_writer.log(self.file_object,message)

            sm = SMOTE(random_state=2)
            X_train_smote, y_train_smote = sm.fit_resample(X_train, y_train.ravel())

            #Number of trees in random forest:
            n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]

            #Number of features to consider at every split:
            max_features = ['auto', 'sqrt']

            #Maximum number of levels in tree:
            max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]

            #Minimum number of samples required to split a node :
            min_samples_split = [2, 5, 10, 15, 100]

            #Minimum number of samples required at each leaf node :
            min_samples_leaf = [1, 2, 5, 10]

            #Setting the random grid :
            param_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

            message1="Modelling Random Forests:  Setting the random grid "
            message2=str(param_grid)
            message=message1+message2
            self.log_writer.log(self.file_object,message)

            message="Modelling Random Forests:  Initiating RandomizedSearchCV"
            self.log_writer.log(self.file_object,message)

            random_forests = RandomForestClassifier()
            random_forests_random = RandomizedSearchCV(estimator = random_forests, param_distributions = param_grid,
                               scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, 
                               random_state=42, n_jobs = -1)

            random_forests_random.fit(X_train_smote,y_train_smote)
            best_params = random_forests_random.best_params_

            message1="Modelling Random Forests:  Getting best_params_ "
            message2=str(best_params)
            message=message1+message2
            self.log_writer.log(self.file_object,message)

            random_forests = RandomForestClassifier( n_estimators = best_params['n_estimators'],
                                         min_samples_split = best_params['min_samples_split'],
                                         min_samples_leaf = best_params['min_samples_leaf'],
                                         max_features = best_params['max_features'],
                                         max_depth = best_params['max_depth'],
                                         random_state = 101)

            random_forests = random_forests.fit(X_train_smote,y_train_smote)
            predictions=random_forests.predict(X_test)

            message='Modelling Random Forests: Saving confusion_matrix'
            self.log_writer.log(self.file_object,message)

            plt.figure(dpi=120)
            mat = confusion_matrix(y_test, predictions)
            sns.heatmap(mat.T, annot=True, fmt='d', cbar=False)

            plt.title('Random Forest - SMOTE')
            plt.xlabel('True label')
            plt.ylabel('Predicted label')
            plt.savefig(figures_path + 'Random Forests - Confusion Matrix - SMOTE')

            message='Modelling Random Forests: Saving classification_report'
            self.log_writer.log(self.file_object,message)

            message=classification_report(y_test,predictions)
            self.log_writer.log(self.file_object,message)

            message='Modelling Random Forests: Predict and save ROC curve'
            self.log_writer.log(self.file_object,message)

            y_test_pred_proba = random_forests.predict_proba(X_test)[:,1]
            model = 'RandomForests'
            
            #Plot the ROC curve
            draw_roc(y_test, y_test_pred_proba, model)

            message='Modelling Random Forests: Dump random_forests_model pickle file in models path'
            self.log_writer.log(self.file_object,message)

            filepath = models_path + 'random_forests_model.pkl'
            with open(filepath, "wb") as f:
                pickle.dump(random_forests, f)

            message = '-------------------------------------------------------------'
            self.log_writer.log(self.file_object,message)

        except Exception as e:
            raise e


# ROC Curve function
def draw_roc( actual, probs, model):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(15, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig(figures_path + 'ROC-Curve-' + model)
    
    return None











