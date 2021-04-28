from cdo_api_py import Client
import pandas as pd
from datetime import datetime
from pprint import pprint


token = "qMupuTscCGIIZzRMRpaSbRJcrVjpPrtG"
my_client = Client(token, default_units='metric', default_limit=1000)

			
extent = {
"north": 40.003162,
"south": 36.993016,
"east": -94.588413,
"west": -102.051744,
}

startdate = datetime(2009, 1, 1)
enddate = datetime(2016, 12, 31)

datasetid='GHCND'
datatypeid=['TMIN']
stations = my_client.find_stations(
    datasetid=datasetid,
    extent=extent,
    startdate=startdate,
    enddate=enddate,
    datatypeid=datatypeid,
    return_dataframe=True
)

big_df = pd.DataFrame()
for rowid, station in stations.iterrows(): # remember this is a pandas dataframe!
    station_data = my_client.get_data_by_station(
        datasetid=datasetid,
        stationid=station['id'],
        startdate=startdate,
        enddate=enddate,
        return_dataframe=True,
        include_station_meta=True # flatten station metadata with ghcnd readings
    )
    big_df = pd.concat([big_df, station_data])

big_df = big_df.sort_values(by='date').reset_index()
big_df.to_csv('test.csv')
# https://anthonylouisdagostino.com/bounding-boxes-for-all-us-states/
