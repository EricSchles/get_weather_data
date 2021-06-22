import pandas as pd
import networkx as nx
from scipy.stats import pearsonr, spearmanr
import numpy as np
from sklearn.manifold import MDS
from pickle import load, dump
from pyproj import Proj

# the metrics
# rotate and translate as much as possible or scale it
# now it's an optimization
# put them next to each other and see if it looks resonable
# Take the data,
# notion of similarity of
# embedding into R2 - using MDS
# after that - get an embedding - you'll get X,y coordinates for every data point
# We have a scale factor to deal with, and a rotation and translation to deal with
# two metrics on the set of stations - two embeddings of those stations based on their metrics
# lat,long -> projected into a distance preserving metric (lambert conformal) -> kilometers

# 1. translate lat, long to lambert conformal, or some projection
# 2. take an embedding of the distance of temperature distance
# 3. tune the embeddings to recover a direct map from one embedding to another embedding by tuning scale, translation and rotation
# 3a. for iterating try a heuristic (get the means to match) 
# 3b. for iteration try a heuristic (get the standard deviation to match - translation
# 3c. for iteration try a heuristic (get the angular statistics to match) - rotation - mean subtract the vectors and z-score everything, and try to find the mean on a circle and get everything to match.  Or a matrix thing.
# 5. or a least squares optimization
# 6. maybe divide by bottom right coefficient so it becomes 1
# can you use arc_sin and arc_cos?
# use top equation again - because we have P now, we take P * D[1] and get predicted D[2]

def filter_df(df):
    columns = df.columns.tolist()
    columns_to_drop = [
       'WT01', 'WT03', 'PRCP', 'SNOW', 'SNWD', 'TOBS', 'datacoverage',
       'mindate', 'maxdate', 'name', 'Unnamed: 0',
       'WT05', 'WT11', 'DAPR', 'MDPR', 'WESF', 'WT06', 'WT04', 'EVAP', 'MNPN',
       'MXPN', 'WDMV', 'SN32', 'SX32', 'WESD', 'SN52', 'SX52', 'SN31', 'SX31',
       'AWND', 'WDFG', 'WSFG', 'TAVG', 'PSUN', 'TSUN', 'WDF2', 'WDF5', 'WSF2',
       'WSF5', 'WT02', 'WT08', 'WT10', 'WT09', 'PGTM', 'MDSF', 'THIC', 'SN02', 'SN12', 'SX02',
       'SX12', 'DASF', 'DAEV', 'MDEV', 'WT07', 'FMTM', 'WT13', 'WT14', 'WT15',
       'WT16', 'WT17', 'WT18', 'WT19', 'WT21', 'WT22'
    ]
    final_columns_to_drop = []
    for column in columns_to_drop:
        if column in columns:
            final_columns_to_drop.append(column)
    df = df.drop(labels=final_columns_to_drop, axis="columns")
    return df[df["TMAX"].notnull()]

def days_to_weeks(station, df, num_days):
    rows, cols = df.shape
    latitude = df.iloc[0]["latitude"]
    longitude = df.iloc[0]["longitude"]
    tmax = df["TMAX"]
    tmin = df["TMIN"]
    dates = df["date"]
    if rows % num_days != 0:
        num_elem_add = num_days - (rows % num_days)
        tmax = tmax.append(pd.Series([np.nan for _ in range(num_elem_add)]))
        tmin = tmin.append(pd.Series([np.nan for _ in range(num_elem_add)]))
        dates = dates.append(pd.Series([np.nan for _ in range(num_elem_add)]))
        rows = len(tmax)
    tmax = tmax.values
    tmin = tmin.values
    dates = dates.values
    num_rows = rows // num_days
    tmax = tmax.reshape(num_rows, num_days)
    tmin = tmin.reshape(num_rows, num_days)
    dates = dates.reshape(num_rows, num_days)
    num_days += 1
    tmax_df = pd.DataFrame(tmax, columns=[f"tmax day {i}" for i in range(1, num_days)])
    tmin_df = pd.DataFrame(tmin, columns=[f"tmin day {i}" for i in range(1, num_days)])
    dates_df = pd.DataFrame(dates, columns=[f"date {i}" for i in range(1, num_days)])
    final_df = pd.concat([tmax_df, tmin_df, dates_df], axis=1, join="inner")
    final_df["station"] = station
    final_df["latitude"] = latitude
    final_df["longitude"] = longitude
    return final_df

def power(x):
    return pow(x, 2)

def sqrt(x):
    return pow(x, 0.5)

def distance_function(vect_one, vect_two):
    diffs = vect_one.values - vect_two.values
    sq_diffs = power(diffs)
    return sqrt(sq_diffs.sum())
    
def align_stations(station_one, station_two):
    dates_one = station_one["date"]
    dates_two = station_two["date"]
    intersection = set(dates_one.values).intersection(dates_two.values)
    station_one = station_one[station_one["date"].isin(intersection)]
    station_two = station_two[station_two["date"].isin(intersection)]
    return station_one, station_two

def featurize_df(df, num_days):
    stations = []
    final_df = pd.DataFrame()
    for station, tmp in df.groupby("station"):
        final_df = final_df.append(
            days_to_weeks(station, tmp, num_days),
            ignore_index=True
        )
    return final_df

def clip_stations(final_df, num_days=False):
    stations = {}
    station_pairs = []
    counter = 0
    for station_one, station_one_df in final_df.groupby("station"):
        for station_two, station_two_df in final_df.groupby("station"):
            if station_one == station_two:
                continue
            if (station_one, station_two) in station_pairs:
                continue
            station_pairs.append((
                station_one, 
                station_two
            ))
            station_pairs.append((
                station_two,
                station_one
            ))
            
            (station_one_res, station_two_res) = align_stations(
                station_one_df,
                station_two_df
            )
            if num_days:
                station_one_res = featurize_df(station_one_res, num_days)
                station_two_res = featurize_df(station_two_res, num_days)
            if station_one_res.shape[0] == 0 or station_two_res.shape[0] == 0:
                continue
            stations[(station_one, station_two)] = (station_one_res, station_two_res)
    return stations
        
def gen_graph_dist_embedding(stations, final_df):
    G = nx.Graph()
    [G.add_node(station) for station in final_df["station"].unique()]
    inf_distance = 10000000
    for station_pair in stations:
        station_one, station_two = station_pair
        station_one_df, station_two_df = stations[station_pair]
        cols = station_one_df.columns.tolist()
        one_df = station_one_df["TMAX"].fillna(0)
        two_df = station_two_df["TMAX"].fillna(0)
        if one_df.empty or two_df.empty:
            G.add_edge(station_one, station_two, dist=inf_distance)
        else:
            try:
                dist = distance_function(
                    one_df, two_df
                )
                G.add_edge(station_one, station_two, dist=dist)
            except:
                G.add_edge(station_one, station_two, dist=inf_distance)
    adjacency_matrix = nx.to_pandas_adjacency(G, weight="dist", dtype=float)
    embedding = MDS(n_components=2, dissimilarity='precomputed', max_iter=500)
    distance_embedding = embedding.fit_transform(adjacency_matrix)
    return adjacency_matrix, distance_embedding

def geo_distance_function(vect_one, vect_two):
    diffs = [
        vect_one.loc[index] - vect_two.loc[index]
        for index in vect_one.index
    ]
    sq_diffs = [power(diff) for diff in diffs]
    return sqrt(sum(sq_diffs))

def is_valid_station_pair(station_one, station_two, station_pairs):
    return (
        ((station_one, station_two) in station_pairs) or
        ((station_two, station_one) in station_pairs)
    )

def gen_month_col(df):
    df["month"] = pd.DatetimeIndex(df["date"]).month
    return df

def lat_long_to_xy(df):
    p = Proj(
        proj='utm', zone=10,
        ellps='WGS84', preserve_units=False
    )
    latitudes = []
    longitudes = []
    for station in df["station"].unique():
        latitudes.append(df[df["station"] == station].iloc[0]["latitude"])
        longitudes.append(df[df["station"] == station].iloc[0]["longitude"])
    latitudes = np.array(latitudes)
    longitudes = np.array(longitudes)
    x, y = p(longitudes, latitudes)
    projection_embedding = pd.DataFrame()
    projection_embedding["x"] = x
    projection_embedding["y"] = y 
    return projection_embedding

def rigid_transform_2D(A, B):
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    # if num_rows != 3:
    #     raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    # if num_rows != 3:
    #     raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.values.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        # this might be the only place 3-D expected.
        Vt[2,:] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    return R, t

    
def dist_main(num_days=False):
    full_df = pd.read_csv("test_small.csv")

    full_df["date"] = pd.to_datetime(full_df["date"])
    full_df = filter_df(full_df)
    full_df = full_df.sort_values(by=["station", "date"])
    full_df = gen_month_col(full_df)
    
    for month, df in full_df.groupby("month"):
        stations = clip_stations(df, num_days=num_days)
        station_pairs = list(stations.keys())
        adjacency_matrix, distance_embedding = gen_graph_dist_embedding(stations, df)
        projection_embedding = lat_long_to_xy(df)
        R, t = rigid_transform_2D(distance_embedding, projection_embedding)
        print(f"month {month}")
        err = (((R @ distance_embedding) * t) - projection_embedding).sum()
        avg_size_of_x = projection_embedding['x'].mean()
        avg_size_of_y = projection_embedding['y'].mean()
        print(f"x error: {err['x']/avg_size_of_x}")
        print(f"y error: {err['y']/avg_size_of_y}")
# I need to sort by the stations to make sure we are doing a correct projection from temperature pairwise distances
# to projections.  Otherwise this doesn't work.

if __name__ == '__main__':
    dist_main(num_days=False)

# top one accuracy 0.5352941176470588
# within one mile 0.9941176470588236

# to dos?:
# histogram approach
# elucid distance again, all points
# elucid distance again, 7 day moving average
# 