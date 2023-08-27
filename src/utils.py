import os
import sys
import numpy as np
import pandas as pd
import pickle

from sklearn.metrics import r2_score

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(x_train, y_train, x_test, y_test, models):
    try:
        report = {}
        for model_name, model in models.items():
            model.fit(x_train, y_train)
            predicted = model.predict(x_test)
            test_model_score = r2_score(y_test, predicted)
            report[model_name] = test_model_score
        return report
    except Exception as e:
        raise CustomException(e, sys)




