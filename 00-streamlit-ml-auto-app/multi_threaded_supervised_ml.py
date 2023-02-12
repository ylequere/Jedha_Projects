# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 00:15:38 2022

@author: Yann
"""
import numpy as np
import pandas as pd
import time
from abc import ABC, abstractmethod
from  multiprocessing.pool import ThreadPool
import warnings

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder #, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.utils import all_estimators
from sklearn.base import RegressorMixin
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score, r2_score, mean_squared_error, average_precision_score
from sklearn.linear_model import LogisticRegressionCV

import xgboost
import catboost
import lightgbm
import imblearn.ensemble 

warnings.filterwarnings("ignore")

removed_classifiers = [
    "ClassifierChain",
    "GaussianProcessClassifier",
    "MultiOutputClassifier", 
    "OneVsOneClassifier",
    "OneVsRestClassifier",
    "OutputCodeClassifier",
    "RadiusNeighborsClassifier",
    "CalibratedClassifierCV",
]

removed_regressors = [
    "TheilSenRegressor",
    "ARDRegression", 
    "CCA", 
    "IsotonicRegression", 
    "MultiOutputRegressor", 
    "MultiTaskElasticNet", 
    "MultiTaskElasticNetCV", 
    "MultiTaskLasso", 
    "MultiTaskLassoCV", 
    "PLSCanonical", 
    "PLSRegression", 
    "RadiusNeighborsRegressor", 
    "RegressorChain", 
    'SVR',
    'GaussianProcessRegressor',
    'KernelRidge',
    'QuantileRegressor',
]

CLASSIFIERS = [
    est
    for est in all_estimators()
    if (issubclass(est[1], ClassifierMixin) and (est[0] not in removed_classifiers))
]

REGRESSORS = [
    est
    for est in all_estimators()
    if (issubclass(est[1], RegressorMixin) and (est[0] not in removed_regressors))
]

REGRESSORS.append(("XGBRegressor", xgboost.XGBRegressor))
REGRESSORS.append(("LGBMRegressor", lightgbm.LGBMRegressor))
REGRESSORS.append(('CatBoostRegressor',catboost.CatBoostRegressor))
REGRESSORS.sort()

CLASSIFIERS.append(("XGBClassifier", xgboost.XGBClassifier))
CLASSIFIERS.append(("LGBMClassifier", lightgbm.LGBMClassifier))
CLASSIFIERS.append(('CatBoostClassifier',catboost.CatBoostClassifier))
CLASSIFIERS.append(('BalancedBaggingClassifier', imblearn.ensemble.BalancedBaggingClassifier))
CLASSIFIERS.append(('EasyEnsembleClassifier', imblearn.ensemble.EasyEnsembleClassifier))
CLASSIFIERS.append(('RUSBoostClassifier', imblearn.ensemble.RUSBoostClassifier))
CLASSIFIERS.append(('BalancedRandomForestClassifier', imblearn.ensemble.BalancedRandomForestClassifier))
CLASSIFIERS.sort()

numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]
)

categorical_transformer_low = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoding", OneHotEncoder(handle_unknown='ignore', sparse=False)),
    ]
)

categorical_transformer_high = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoding", OrdinalEncoder(handle_unknown='ignore')),
    ]
)

def get_card_split(df, cols, n=11):
    cond = df[cols].nunique() > n
    card_high = cols[cond]
    card_low = cols[~cond]
    return card_low, card_high

def adjusted_rsquared(r2, n, p):
    return 1 - (1 - r2) * ((n - 1) / (n - p - 1))

def get_numeric_categorical_features(df):
    numeric_features = df.select_dtypes(include=[np.number]).columns
    categorical_features = df.select_dtypes(include=["object"]).columns
    return numeric_features, categorical_features

class MultiThreadedModel(ABC):
    def __init__(
        self,
        random_state=42,
        models="all",
        num_workers=8, 
        stacking_estimator_choice="No",
        stacking_estimators=[],
        voting_estimator_choice="No",
        voting_estimators=[],
    ):
        self.random_state = random_state
        self.models = models
        self.num_workers = num_workers
        self.stacking_estimator_choice = stacking_estimator_choice
        self.stacking_estimators = stacking_estimators
        self.voting_estimator_choice = voting_estimator_choice
        self.voting_estimators = voting_estimators 

        self.params = {}
        self.errors = []
        self.scores_train = None
        self.scores_test = None
        self.df_errors = None
        self.X_train = None
        self.preprocessor = None
                
    def print_progress(self):
        thread_count = min(self.active_thread_count, self.num_workers)
        print('\r Models [' + '♥'*self.num_task_achieved*2 + ' '*(self.num_tasks-self.num_task_achieved)*2 + ']'
              + f' {self.num_task_achieved}/{self.num_tasks} - Threads [' 
              + '★'*thread_count + ' '*(self.num_workers-thread_count) 
              + f'] {min(thread_count, self.num_workers)}/{self.num_workers}', end='')

    def _add_error(self, model_name, error_msg):
        self.errors.append([model_name, error_msg])
        
    def _get_estimator(self, estimator_class):
        
        if estimator_class.__name__ in ('OneClassSVM'):
            return estimator_class(kernel="rbf")
        
        if estimator_class.__name__ in ('BalancedBaggingClassifier', 'EasyEnsembleClassifier', 'RUSBoostClassifier', 'BalancedRandomForestClassifier'):
            return estimator_class(random_state=self.random_state)
        
        if estimator_class.__name__ in ('OneVsOneClassifier', 'OneVsRestClassifier'):
            return estimator_class(LogisticRegressionCV(random_state=self.random_state))
        elif estimator_class in (catboost.CatBoostRegressor, catboost.CatBoostClassifier):
            return estimator_class(random_seed=self.random_state, silent=True)
        elif "random_state" in estimator_class().get_params().keys():
            return estimator_class(random_state=self.random_state)
        else:
            return estimator_class()
    
    @abstractmethod
    def _init_fit(self):
        pass

    @abstractmethod
    def _add_metrics(self, name, y_train, Y_train_pred, y_test, Y_test_pred, X_train, X_test, train_time):	
        pass

    @abstractmethod
    def _get_metrics(self):
        pass
    
    def multithread_fit(self, X_train, X_test, y_train, y_test, preprocessor=None, sort_by=None, y_label_encoder=True):
        
        s = time.time()

        self.y_label_encoder = y_label_encoder
        self.sort_by = sort_by

        self._init_fit()

        if self.y_label_encoder:
            le = LabelEncoder()
            y_train = le.fit_transform(y_train)
            y_test = le.transform(y_test)
        
        if isinstance(X_train, np.ndarray):
            X_train = pd.DataFrame(X_train)
            X_test = pd.DataFrame(X_test)

        if preprocessor != 'forced_none':
            if preprocessor is None:
                numeric_features, categorical_features = get_numeric_categorical_features(X_train)
                categorical_low, categorical_high = get_card_split(X_train, categorical_features)
        
                preprocessor = ColumnTransformer(
                    transformers=[
                        ("numeric", numeric_transformer, numeric_features),
                        ("categorical_low", categorical_transformer_low, categorical_low),
                        ("categorical_high", categorical_transformer_high, categorical_high),
                    ]
                )
    
            X_train = preprocessor.fit_transform(X_train)
            X_test = preprocessor.transform(X_test)

        self.active_thread_count = 0
        
        def _thread_fit(classifier):
            self.active_thread_count += 1
            self.print_progress()
            name, model = classifier

            try:
                if name in ("StackingClassifier", "StackingRegressor"):
                    if self.stacking_estimator_choice != "No":
                        # Building Stacking estimators
                        dict_all_models = dict(self.models)
                        estimators = []
                        for stacking_estimator in self.stacking_estimators:
                            if stacking_estimator != "None":
                                sc_class = dict_all_models[stacking_estimator]
                                estimators.append(('sc'+str(len(estimators)+1), self._get_estimator(sc_class)))
                        final_estimator = estimators.pop()[1]
                        
                        pipeline_model = model(estimators=estimators, final_estimator=final_estimator)
                    else:
                        return
                elif name in ("VotingClassifier", "VotingRegressor"):
                    if self.voting_estimator_choice != "No":
                        # Building Voting estimators
                        dict_all_models = dict(self.models)
                        estimators = []
                        for voting_estimator in self.voting_estimators:
                            if voting_estimator != "None":
                                vc_class = dict_all_models[voting_estimator]
                                estimators.append(('vc'+str(len(estimators)+1), self._get_estimator(vc_class)))
                        
                        pipeline_model = model(estimators=estimators)
                    else:
                        return
                else:
                    if self.stacking_estimator_choice == "Only" or self.voting_estimator_choice  == "Only":
                        return
                    pipeline_model = self._get_estimator(model)

                pipe = Pipeline(steps=[("model", pipeline_model)])

                start = time.time()
                # Fit all the transformers one after the other and transform the data. Finally, fit the transformed data using the final estimator.
                pipe.fit(X_train, y_train)
                train_time = time.time() - start
                self.params[name] = pipe
                # Transform the data, and apply predict with the final estimator.
                Y_train_pred = pipe.predict(X_train)
                Y_test_pred = pipe.predict(X_test)
                
                self._add_metrics(name, y_train, Y_train_pred, y_test, Y_test_pred, X_train, X_test, train_time)
                
            except Exception as exception:
                self._add_error(name, str(exception))
                
            self.num_task_achieved +=1
            self.active_thread_count -= 1
            self.print_progress()

        if self.stacking_estimator_choice == "Only" or self.voting_estimator_choice == "Only":
            if self.stacking_estimator_choice == "No" or self.voting_estimator_choice == "No":
                self.num_tasks = 1
            else:
                self.num_tasks = 2
        else:
            self.num_tasks = len(self.models)
            if self.stacking_estimator_choice == "No":
                self.num_tasks -= 1
            if self.voting_estimator_choice == "No":
                self.num_tasks -= 1
            
        self.num_task_achieved = 0
        self.print_progress()
        
        pool = ThreadPool(self.num_workers)
        pool.map(_thread_fit, self.models)
        pool.close()
        pool.join()
        
        self.active_thread_count = 0
        self.print_progress()

        self._get_metrics()

        self.scores_train = self.scores_train.sort_values(by=sort_by, ascending=False).set_index("Model")
        self.scores_test = self.scores_test.sort_values(by=sort_by, ascending=False).set_index("Model")
        self.df_errors = pd.DataFrame(columns=['Model', 'Error'], data=self.errors)
        if preprocessor != 'forced_none':
            self.X_train = pd.DataFrame(columns=preprocessor.get_feature_names_out(), data=X_train)
            self.preprocessor = preprocessor
        else:
            self.X_train = X_train
            self.preprocessor = None
            
        print(f'\nALL DONE in {round(time.time()-s)} seconds !')
    
class MultiThreadedRegressor(MultiThreadedModel):
        
    def _init_fit(self):
        
        self.train_R2 = []
        self.train_ADJR2 = []
        self.train_RMSE = []
        self.train_names = []
        self.train_TIME = []

        self.test_R2 = []
        self.test_ADJR2 = []
        self.test_RMSE = []
        self.test_names = []

        if self.models=='all':
            self.models = REGRESSORS
    
        if self.sort_by is None:
            self.sort_by="Adjusted R-Squared"
    
        self.y_label_encoder = False
        
    def _add_metrics(self, name, y_train, Y_train_pred, y_test, Y_test_pred, X_train, X_test, train_time):
        r_squared = r2_score(y_train, Y_train_pred)
        adj_rsquared = adjusted_rsquared(r_squared, X_train.shape[0], X_train.shape[1])
        rmse = np.sqrt(mean_squared_error(y_train, Y_train_pred))

        self.train_names.append(name)
        self.train_R2.append(r_squared)
        self.train_ADJR2.append(adj_rsquared)
        self.train_RMSE.append(rmse)
        self.train_TIME.append(round(train_time, 2))
        
        r_squared = r2_score(y_test, Y_test_pred)
        adj_rsquared = adjusted_rsquared(r_squared, X_test.shape[0], X_test.shape[1])
        rmse = np.sqrt(mean_squared_error(y_test, Y_test_pred))

        self.test_names.append(name)
        self.test_R2.append(r_squared)
        self.test_ADJR2.append(adj_rsquared)
        self.test_RMSE.append(rmse)
        
        
    def _get_metrics(self):
        self.scores_train = pd.DataFrame(
            {
                "Model": self.train_names,
                "Adjusted R-Squared": self.train_ADJR2,
                "R-Squared": self.train_R2,
                "RMSE": self.train_RMSE,
                "Time Taken": self.train_TIME,
            }
            )

        self.scores_test = pd.DataFrame(
            {
                "Model": self.test_names,
                "Adjusted R-Squared": self.test_ADJR2,
                "R-Squared": self.test_R2,
                "RMSE": self.test_RMSE,
            }
            )
        

class MultiThreadedClassifier(MultiThreadedModel):
        
    def _init_fit(self):

        self.train_Accuracy = []
        self.train_B_Accuracy = []
        self.train_ROC_AUC = []
        self.train_F1 = []
        self.train_names = []
        self.train_TIME = []
        self.train_PR_AUC = []

        self.test_Accuracy = []
        self.test_B_Accuracy = []
        self.test_ROC_AUC = []
        self.test_F1 = []
        self.test_names = []
        self.test_PR_AUC = []

        if self.models=='all':
            self.models = CLASSIFIERS
        
        if self.sort_by is None:
            self.sort_by="F1 Score"

    def _add_metrics(self, name, y_train, Y_train_pred, y_test, Y_test_pred, X_train, X_test, train_time):
                
        accuracy = accuracy_score(y_train, Y_train_pred, normalize=True)
        b_accuracy = balanced_accuracy_score(y_train, Y_train_pred)
        try:
            f1 = f1_score(y_train, Y_train_pred)
        except:
            f1 = f1_score(y_train, Y_train_pred, average=None)
        try:
            roc_auc = roc_auc_score(y_train, Y_train_pred)
        except:
            roc_auc = None
        try:
            pr_auc = average_precision_score(y_train, Y_train_pred)
        except:
            pr_auc = None
            
        self.train_names.append(name)
        self.train_Accuracy.append(accuracy)
        self.train_B_Accuracy.append(b_accuracy)
        self.train_ROC_AUC.append(roc_auc)
        self.train_F1.append(f1)
        self.train_TIME.append(round(train_time, 2))
        self.train_PR_AUC.append(pr_auc)
        
        accuracy = accuracy_score(y_test, Y_test_pred, normalize=True)
        b_accuracy = balanced_accuracy_score(y_test, Y_test_pred)
        try:
            f1 = f1_score(y_test, Y_test_pred)
        except:
            f1 = f1_score(y_test, Y_test_pred, average=None)
        try:
            roc_auc = roc_auc_score(y_test, Y_test_pred)
        except:
            roc_auc = None
        try:
            pr_auc = average_precision_score(y_test, Y_test_pred)
        except:
            pr_auc = None

        self.test_names.append(name)
        self.test_Accuracy.append(accuracy)
        self.test_B_Accuracy.append(b_accuracy)
        self.test_ROC_AUC.append(roc_auc)
        self.test_F1.append(f1)
        self.test_PR_AUC.append(pr_auc)

    def _get_metrics(self):
        self.scores_train = pd.DataFrame(
            {
                "Model": self.train_names,
                "Accuracy": self.train_Accuracy,
                "Balanced Accuracy": self.train_B_Accuracy,
                "ROC AUC": self.train_ROC_AUC,
                "F1 Score": self.train_F1,
                "PR AUC": self.train_PR_AUC,
                "Time Taken": self.train_TIME,
            }
            )

        self.scores_test = pd.DataFrame(
            {
                "Model": self.test_names,
                "Accuracy": self.test_Accuracy,
                "Balanced Accuracy": self.test_B_Accuracy,
                "ROC AUC": self.test_ROC_AUC,
                "F1 Score": self.test_F1,
                "PR AUC": self.test_PR_AUC,
            }
            )    
        
