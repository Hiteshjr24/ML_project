import sys
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
@dataclass
class data_formation_config:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')


class data_transformation:
    def __init__(self):
        self.datatransformation_config = data_formation_config()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation

        '''
        try:
            numerical_columns = ["reading_score",	"writing_score"]
            catagorical_columns = ["gender","race_ethnicity","parental_level_of_education","lunch",	"test_preparation_course",]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scalar",StandardScaler(with_mean=False))
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))

                ]

            )
            logging.info("column info encoded")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipeline",cat_pipeline,catagorical_columns)
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("reading train and test data completed")
            logging.info("obtaining preprossesing object")
            preprocessing_obj = self.get_data_transformer_object()
            target_column_name = "math_score"

            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(f"applying preprocessing on train and test dataframe"
                         )

            input_feature_train_matrix =preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_matrix = preprocessing_obj.transform(input_feature_test_df)
            train_arr = np.c_[input_feature_train_matrix,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_matrix,np.array(target_feature_test_df)]

            logging.info(f"saved the preprocessing objects")
            save_object(
                file_path = self.datatransformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj

            )

            return(
                train_arr,
                test_arr,
                self.datatransformation_config.preprocessor_obj_file_path,

            )
        except Exception as e:
            raise CustomException(e,sys)


