import numpy as np
import pandas as pd 
from pathlib import Path 
import logging 

logger = logging.getLogger("data_cleaning")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

logger.addHandler(handler)

formatter = logging.Formatter(
    fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

handler.setFormatter(formatter)

columns_to_drop = ['rider_id','restaurant_latitude','restaurant_longitude','delivery_latitude','delivery_longitude',
'order_date','order_time_hour','order_day','city_name','order_day_of_week','order_month']


def load_data(data_path: Path)->pd.DataFrame:
    try:
       df = pd.read_csv(data_path)
    except FileNotFoundError:
       logger.error('The file to load doesnt exist')
    return df


def change_column_names(data:pd.DataFrame)->pd.DataFrame:
  return (
      data.rename(str.lower,axis=1).rename({
          "delivery_person_id":"rider_id",
          "delivery_person_age":"age",
          "delivery_person_ratings" : "ratings",
          "delivery_location_latitude": "delivery_latitude",
          "delivery_location_longitude": "delivery_longitude",
          "time_orderd":"order_time",
          "time_order_picked":"order_picked_time",
          "weatherconditions":"weather",
          "road_traffic_density":"traffic",
          "city":"city_type",
          "time_taken(min)" : "time_taken"},axis=1)
  )


def data_cleaning(data:pd.DataFrame)->pd.DataFrame:
  minors_data = data.loc[data['age'].astype('float')<18]
  minor_index = minors_data.index.tolist()
  six_star_data = data.loc[data['ratings'] == "6"]
  six_star_index = six_star_data.index.tolist()
  return(
      data.drop(columns="id").drop(index=minor_index).drop(index=six_star_index).replace("NaN",np.nan)
      .assign(city_name= lambda x: x['rider_id'].str.split("RES").str.get(0),
              age = lambda x: x['age'].astype(float),
              ratings = lambda x: x['ratings'].astype(float),

              restaurant_latitude = lambda x:x['restaurant_latitude'].abs(),
              restaurant_longitude = lambda x:x['restaurant_longitude'].abs(),
              delivery_latitude = lambda x:x['delivery_latitude'].abs(),
              delivery_longitude = lambda x:x['delivery_longitude'].abs(),

              order_date = lambda x:pd.to_datetime(x['order_date'],dayfirst=True),
              order_day = lambda x:x['order_date'].dt.day,
              order_month = lambda x:x['order_date'].dt.month,
              order_day_of_week = lambda x:x['order_date'].dt.day_name().str.lower(),
              is_weekend = lambda x:x['order_date'].dt.day_name().isin(["Saturday","Sunday"]).astype(int),

              order_time = lambda x: pd.to_datetime(x['order_time'], format='mixed', errors='coerce'),
              order_picked_time = lambda x: pd.to_datetime(x['order_picked_time'], format='mixed', errors='coerce'),
              pickup_time_minutes = lambda x: (x['order_picked_time'] - x['order_time']).dt.total_seconds() / 60,
              order_time_hour = lambda x: x['order_time'].dt.hour,
              order_time_of_day = lambda x: x['order_time_hour'].pipe(time_of_day),

              weather = lambda x: (x['weather'].str.replace('conditions ',"").str.lower().replace("nan",np.nan)),

              traffic = lambda x: x["traffic"].str.rstrip().str.lower(),
              type_of_order = lambda x: x['type_of_order'].str.rstrip().str.lower(),
              type_of_vehicle = lambda x: x['type_of_vehicle'].str.rstrip().str.lower(),

              festival = lambda x: x['festival'].str.rstrip().str.lower(),
              city_type = lambda x: x['city_type'].str.rstrip().str.lower(),

              multiple_deliveries = lambda x: x['multiple_deliveries'].astype(float),

              time_taken = lambda x: (x['time_taken'].str.replace("(min) ","").astype(int)))
              .drop(columns=["order_time","order_picked_time"])
  )


def clean_lat_long(data:pd.DataFrame,threshold: float=1.0)->pd.DataFrame:
  location_columns = ['restaurant_latitude','restaurant_longitude','delivery_latitude','delivery_longitude']
  return(
      data.assign(**{
          col:(
              np.where(data[col]<threshold,np.nan,data[col].values)
          ) for col in location_columns
      })
  )


def extract_datetime_features(ser:pd.Series)->pd.DataFrame:
  date_col = pd.to_datetime(ser,dayfirst=True)
  return(
      pd.DataFrame(
          {
              "day":date_col.dt.day,
              "month":date_col.dt.month,
              "year":date_col.dt.year,
              "day_of_week":date_col.dt.day_name(),
              "is_weekend":date_col.dt.day_name().isin(["Saturday","Sunday"]).astype(int)
          }
      )
  )



def time_of_day(ser:pd.Series):
  time_col = pd.to_datetime(ser,format='mixed').dt.hour
  return(
     pd.cut(ser,bins=[0,6,12,17,20,24],right=True,labels=['after_midnight','morning','afternoon','evening','night'])
  )



def calculate_haversine_distance(df:pd.DataFrame)->pd.DataFrame:
  location_columns = ['restaurant_latitude','restaurant_longitude','delivery_latitude','delivery_longitude']
  lat1 = df[location_columns[0]]
  lon1 = df[location_columns[1]]
  lat2 = df[location_columns[2]]
  lon2 = df[location_columns[3]]

  lon1,lat1,lon2,lat2 = map(np.radians,[lon1,lat1,lon2,lat2])
  dlon = lon2-lon1
  dlat = lat2-lat1

  a = np.sin(dlat/2.0)**2+np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2

  c = 2*np.arcsin(np.sqrt(a))
  distance = 6371*c
  return (
      df.assign(distance=distance)
  )

def create_distance_type(data:pd.DataFrame)->pd.DataFrame:
  return(
    data.assign(distance_type = pd.cut(data['distance'],bins=[0,5,10,15,25],right=False,labels=['short','medium','long','very long']
                                       ))
  )

def drop_columns(data:pd.DataFrame,columns:list)->pd.DataFrame:
  df = data.drop(columns=columns)
  return df 

def  perform_data_cleaning(data:pd.DataFrame,saved_data_path:Path)->None:
  cleaned_data = (data
                  .pipe(change_column_names)
                  .pipe(data_cleaning)
                  .pipe(clean_lat_long)
                  .pipe(calculate_haversine_distance)
                  .pipe(create_distance_type)
                  .pipe(drop_columns,columns=columns_to_drop)
                  )
  

  cleaned_data.to_csv(saved_data_path,index=False)


if __name__ =="__main__":
  root_path = Path(__file__).parent.parent.parent
  cleaned_data_save_dir = root_path/"data"/"cleaned"
  cleaned_data_save_dir.mkdir(exist_ok=True,parents=True)
  cleaned_data_filename = "swiggy_cleaned.csv"
  cleaned_data_save_path = cleaned_data_save_dir / cleaned_data_filename
  data_load_path = root_path / "data" /"raw"/"swiggy.csv"

  df=load_data(data_load_path)
  logger.info("Data read successfully")

  perform_data_cleaning(data=df,saved_data_path=cleaned_data_save_path)
  logger.info("Data cleaned and saved successfully")
