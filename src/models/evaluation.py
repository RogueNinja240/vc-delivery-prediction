import pandas as pd 
import joblib 
import logging
import mlflow
import dagshub 
from pathlib import Path 
from sklearn.model_selection import cross_val_score 
from sklearn.metrics import mean_absolute_error,r2_score
from sklearn.compose import TransformedTargetRegressor

import dagshub
dagshub.init(repo_owner='RogueNinja240', repo_name='vc-delivery-prediction', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/RogueNinja240/vc-delivery-prediction.mlflow")
mlflow.set_experiment("DVC Pipeline")

TARGET ='time_taken'

logger = logging.getLogger("model_training")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

logger.addHandler(handler)

formatter = logging.Formatter(
    fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
handler.setFormatter(formatter)

def load_data(data_path: Path)->pd.DataFrame:
    try:
       df = pd.read_csv(data_path)
    except FileNotFoundError:
       logger.error('The file to load doesnt exist')
    return df


def make_X_and_y(data:pd.DataFrame,target_column:str):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return X,y  

def load_model(model_path: Path):
    model = joblib.load(model_path)
    return model 

if __name__ =="__main__":
    root_path = Path(__file__).parent.parent.parent 
    train_data_path = root_path / "data" / "processed" / "train_trans.csv"
    test_data_path = root_path / "data" / "processed" / "test_trans.csv"
    model_path = root_path / "models" / "model.joblib"

    train_data = load_data(train_data_path)
    logger.info("Train data loaded successfully")
    test_data = load_data(test_data_path)
    logger.info("Test data loaded successfully")

    X_train,y_train = make_X_and_y(train_data,TARGET)
    X_test,y_test  = make_X_and_y(test_data,TARGET)
    logger.info("Data splitted successfully")

    model = load_model(model_path)
    logger.info("Model loaded successfully")

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    logger.info("Prediction successfully completed")

    train_mae = mean_absolute_error(y_train,y_train_pred)
    test_mae = mean_absolute_error(y_test,y_test_pred)
    logger.info("Error calculated successfully")

    train_r2 = r2_score(y_train,y_train_pred)
    test_r2 = r2_score(y_test,y_test_pred)
    logger.info("R2 score calculated successfully")

    cv_scores = cross_val_score(model,X_train,y_train,cv=5,scoring='neg_mean_absolute_error',n_jobs=-1)
    logger.info("cross validation completed")

    mean_csv_score = -(cv_scores.mean())
    with mlflow.start_run():
        mlflow.set_tag('model','food_delivery_time_regressor')
        mlflow.log_params(model.get_params())
        mlflow.log_metric('train_mae',train_mae)
        mlflow.log_metric('test_mae',test_mae)
        mlflow.log_metric('train_r2',train_r2)
        mlflow.log_metric('test_r2',test_r2)
        mlflow.log_metric('mean_cv_score',-(cv_scores.mean()))

    mlflow.log_metrics({f"CV {num}": score for num,score in  enumerate(-cv_scores)})
    
    train_data_input = mlflow.data.from_pandas(train_data,targets=TARGET)
    test_data_input = mlflow.data.from_pandas(test_data,targets=TARGET)

    mlflow.log_input(dataset=train_data_input,context='training')
    mlflow.log_input(dataset=test_data_input,context='validation')

    model_signature = mlflow.models.infer_signature(
    model_input=X_train.sample(20, random_state=42),
    model_output=model.predict(X_train.sample(20, random_state=42)))

    mlflow.sklearn.log_model(model,"model",signature=model_signature)
    logger.info("MLFlow logging complete and model logged")




   






