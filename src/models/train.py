import pandas as pd 
import yaml 
import joblib 
import logging
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import PowerTransformer
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from pathlib import Path
from sklearn.ensemble import StackingRegressor

TARGET = "time_taken"

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

def read_params(file_path):
   with open(file_path,'r') as f:
      params_file = yaml.safe_load(f)
   return params_file

def save_model(model,save_dir:Path,model_name:str):
   save_location = save_dir / model_name
   joblib.dump(value=model,filename=save_location)

def save_transformer(transformer,save_dir:Path,transformer_name:str):
   save_location = save_dir / transformer_name
   joblib.dump(transformer,save_location)

def train_model(model,X_train:pd.DataFrame,y_train):
   model.fit(X_train,y_train)
   return model 

def make_X_and_y(data:pd.DataFrame,target_column:str):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return X,y  

if __name__ =="__main__":
    root_path = Path(__file__).parent.parent.parent
    data_path  = root_path / "data" / "processed" / "train_trans.csv"
    params_file_path = root_path / "params.yaml"
    training_data = load_data(data_path)
    logger.info("trained data completed successfully")

    X_train,y_train = make_X_and_y(training_data,TARGET)
    logger.info("data splitting completed successfully")

    model_params = read_params(params_file_path)['Train']
    rf_params = model_params['Random_Forest']
    logger.info("random forest parameters read successfully")

    rf = RandomForestRegressor(**rf_params)
    logger.info("random forest model build successfully")

    lgbm_params = model_params['LightGBM']
    logger.info("lightGBM parameters read successfully")
    lgbm = LGBMRegressor(**lgbm_params)
    logger.info("LightGBM model build successfully")

    lr = LinearRegression()
    logger.info("linear regression model build successfully")

    power_transform = PowerTransformer()
    logger.info("Target tranformer model build successfully")

    stacking_reg = StackingRegressor(estimators=[('rf_model',rf),('lgmb_model',lgbm)],final_estimator=lr,cv=5,n_jobs=-1)
    logger.info("Stacking regressor model build successfully")

    model = TransformedTargetRegressor(regressor=stacking_reg,transformer=power_transform)
    logger.info("Models wrapped inside wrapper")

    train_model(model,X_train,y_train)
    logger.info("Model training completed")

    model_filename = "model.joblib"
    model_save_dir = root_path / "models"
    model_save_dir.mkdir(exist_ok=True)

    stacking_model = model.regressor_
    transformer = model.transformer_

    save_model(model=model,save_dir=model_save_dir,model_name=model_filename)
    logger.info("Trained model saved to location")

    stacking_filename = "stacking_regressor.joblib"
    save_model(model=stacking_model,save_dir=model_save_dir,model_name=stacking_filename)
    logger.info("Trained model saved to location")

    transformer_filename = "power_transformer.joblib"
    transformer_save_dir = model_save_dir
    save_transformer(transformer,transformer_save_dir,transformer_filename)
    logger.info("Transfomer saved to location")









