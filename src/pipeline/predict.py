import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class prediction_pipeline:
    def __init__(self):
        pass
    def predict(self,features):
        model_path ="/Users/hiteshms/Desktop/untitled folder/ML_project/src/components/artifacts/model.pkl"
        preprocessor_path = "/Users/hiteshms/Desktop/untitled folder/ML_project/src/components/artifacts/preprocessor.pkl"
        model = load_object(file_path=model_path)
class custom_data:
    def __init__(self,
                 gender:str, race_ethnicity:int, parental_level_of_education, lunch:int, test_preparation_course:int,reading_score:int,writing_score:int):
        self.gender = gender
        self.race_eth = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preperation = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score


    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_eth],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preperation],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
