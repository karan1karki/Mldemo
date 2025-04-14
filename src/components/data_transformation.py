import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
import os

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'processor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_tranformation_config= DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
            this function is responsible for data transformation
        '''
        try:
            numerical_columns = ["writing_score","reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_eductaion",
                "lunch",
                "test_preperation_course",
            ]

            num_pipeline = Pipeline (
                steps= [
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler),
                ]

            )
            cat_pipelines = Pipeline(
                steps = [
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler())
                ]
            )

            logging.info(f"Categorical colums:{categorical_columns}")
            logging.info(f"Numerical columns:{numerical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipelines",cat_pipelines,categorical_columns)
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        
    def initate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("obtaining preprocessor completed")

            preprocessing_obj= self.get_data_transformer_object()

            traget_column_name = "math_score"
            numerical_columns = ["writing_score","reading_score"]

            input_feature_train_df = train_df.drop(columns=[traget_column_name],axis=1)
            traget_feature_train_df = train_df[traget_column_name]

            input_feature_test_df = test_df.drop(columns=[traget_column_name],axis=1)
            traget_feature_test_df = test_df[traget_column_name]

            logging.info(
                f"Applying preprocessing object on training datafrane and testing dataframe"
            )

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.fit_transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(traget_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(traget_feature_test_df)
            ]

            logging.info("Saved preprocessing object.")

            save_object(
                file_path = self.data_tranformation_config.preprocessor_obj_file_path,
                obj= preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_tranformation_config.preprocessor_obj_file_path,
            )
        except:
            pass
