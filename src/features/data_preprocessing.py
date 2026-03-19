import pandas as pd 
import logging 
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (OneHotEncoder,MinMaxScaler,OrdinalEncoder)
import joblib
from sklearn import set_config 

set_config(transform_output="pandas")
num_cols = ['age','ratings','pickup_time_minutes','distance']
nominal_cat_cols = ['weather','type_of_order','type_of_vehicle','festival','city_type','is_weekend','order_time_of_day']
ordinal_cat_cols = ['traffic','distance_type']
target_col = 'time_taken'
traffic_order = ['low','medium','high','jam']
distance_type_order = ['short','medium','long','very_long']

logger = logging.getLogger("data_preprocessing")
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

def drop_missing_values(data:pd.DataFrame)->pd.DataFrame:
    logger.info(f"The original dataset has {data.shape[0]} rows and cols {data.shape[1]}")
    df_dropped = data.dropna()
    logger.info(f"After dropping the dataset has {data.shape[0]} rows and cols {data.shape[1]}")
    missing_vals = df_dropped.isna().sum().sum()

    if missing_vals>0:
       raise ValueError("Dataframe has missing values")
    return df_dropped

def save_transformer(transformer,save_dir:Path, tranformer_name:str):
    save_location = save_dir / tranformer_name
    # FIXED: Changed `value=tranformer_name` to `value=transformer`
    joblib.dump(value=transformer, filename=save_location)

def train_preprocessor(preprocessor,data:pd.DataFrame):
    preprocessor.fit(data)
    return preprocessor

def perform_tranformations(preprocessor,data:pd.DataFrame):
    transformed_data = preprocessor.transform(data)
    return transformed_data

def save_data (data:pd.DataFrame,save_path:Path)->None:
    data.to_csv(save_path,index=False)

def make_X_and_y(data:pd.DataFrame,target_column:str):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return X,y 

def join_X_and_y(X:pd.DataFrame,y:pd.Series):
    joined_df = X.join(y,how='inner')
    return joined_df

if __name__ =="__main__":
    root_path = Path(__file__).parent.parent.parent
    train_data_path = root_path / "data" / "interim" / "train.csv"
    test_data_path = root_path / "data" / "interim" / "test.csv"
    save_data_dir = root_path / "data" / "processed"
    save_data_dir.mkdir(exist_ok=True,parents=True)
    train_trans_filename = "train_trans.csv"
    test_trans_filename = "test_trans.csv"

    save_train_trans_path = save_data_dir / train_trans_filename
    save_test_trans_path = save_data_dir / test_trans_filename

    preprocessor = ColumnTransformer(transformers=[
        ('scale',MinMaxScaler(),num_cols),
        ('nominal_encode',OneHotEncoder(drop='first',handle_unknown='ignore',sparse_output=False),nominal_cat_cols),
        ('ordinal_encode',OrdinalEncoder(categories=[traffic_order,distance_type_order],encoded_missing_value=-999,handle_unknown="use_encoded_value",unknown_value=-1),ordinal_cat_cols)
    ], remainder='passthrough',n_jobs=-1,force_int_remainder_cols=False,verbose_feature_names_out=False)
  
    train_df = drop_missing_values(load_data(data_path=train_data_path))
    logger.info("Train data loaded successfully")
    test_df = drop_missing_values(load_data(data_path=test_data_path))
    logger.info("Test data loaded successfully")

    X_train,y_train = make_X_and_y(data=train_df,target_column=target_col)
    X_test,y_test = make_X_and_y(data=test_df,target_column=target_col)
    logger.info("Data Splitting completed")

    # FIXED: Added the fitting step here before transforming
    preprocessor = train_preprocessor(preprocessor=preprocessor, data=X_train)
    logger.info("Preprocessor is fitted")

    X_train_trans = perform_tranformations(preprocessor=preprocessor,data=X_train)
    logger.info("Train data is transformed")
    X_test_trans = perform_tranformations(preprocessor=preprocessor,data=X_test)
    logger.info("Test data is transformed")


    train_trans_df = join_X_and_y(X_train_trans,y_train)
    test_trans_df = join_X_and_y(X_test_trans,y_test)
    logger.info("Dataset is joined")

    data_subsets = [train_trans_df,test_trans_df]
    data_paths = [save_train_trans_path,save_test_trans_path]
    filename_list = [train_trans_filename,test_trans_filename]

    for filename,path,data in zip(filename_list,data_paths,data_subsets):
        save_data(data=data,save_path=path)
        logger.info(f"{filename.replace('.csv','')} data saved to location")

    transformer_filename ="preprocessor.joblib"
    transformer_save_dir = root_path / "models"
    transformer_save_dir.mkdir(exist_ok=True)
    save_transformer(transformer=preprocessor,save_dir=transformer_save_dir,tranformer_name=transformer_filename)
    logger.info("Transformer saved to location")