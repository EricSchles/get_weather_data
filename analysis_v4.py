import pandas as pd
import networkx as nx
from scipy.stats import pearsonr, spearmanr


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

def power(x):
    return pow(x, 2)

def sqrt(x):
    return pow(x, 0.5)

def distance_function(vect_one, vect_two):
    diffs = vect_one - vect_two
    sq_diffs = power(diffs)
    return sqrt(sq_diffs.sum())
    
def align_stations(station_one, station_two):
    dates_one = station_one["date"]
    dates_two = station_two["date"]
    intersection = set(dates_one.values).intersection(dates_two.values)
    station_one = station_one[station_one["date"].isin(intersection)]
    station_two = station_two[station_two["date"].isin(intersection)]
    return station_one, station_two

def clip_stations(final_df):
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
            if station_one_res.shape[0] == 0 or station_two_res.shape[0] == 0:
                continue
            stations[(station_one, station_two)] = (station_one_res, station_two_res)
    return stations
        
def gen_graph(stations, final_df):
    G = nx.Graph()
    [G.add_node(station) for station in final_df["station"].unique()]
    for station_pair in stations:
        station_one, station_two = station_pair
        station_one_df, station_two_df = stations[station_pair]
        try:
            corr, pvalue = spearmanr(
                station_one_df["TMAX"], station_two_df["TMAX"]
            )
            G.add_edge(station_one, station_two, corr=corr, pvalue=pvalue)
        except:
            print(station_one, station_two)
        
    return G

def get_closest(node, nodes, graph, threshold_pvalue, threshold_corr):
    distances = []
    for node_j in nodes:
        if node == node_j:
            continue
        try:
            distances.append((node_j,
                graph[(node, node_j)]["pvalue"],
                graph[(node, node_j)]["corr"]
            ))
        except:
            continue
            #print("failed on ", node, node_j)
    if distances == []:
        return distances
    else:
        return [
            distance for distance in distances
            if (distance[1] < threshold_pvalue) and (distance[2] > threshold_corr)
        ]
    
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
        
def get_geographic_closest(stations_pairs, final_df):
    closest_nodes = {}
    geo_distances = {}
    for station_one, tmp_one in final_df.groupby("station"):
        distances = []
        for station_two, tmp_two in final_df.groupby("station"):
            if is_valid_station_pair(station_one, station_two, station_pairs):
                distances.append((station_one, station_two, geo_distance_function(
                    tmp_one.iloc[0][["latitude", "longitude"]],
                    tmp_two.iloc[0][["latitude", "longitude"]]
                )))
        distances = sorted(distances, key=lambda t: t[2])
        station_a, station_b, distance = distances[1]
        closest_nodes[station_a] = station_b
        # 0.014 is approximately 1 square mile
        geo_distances[station_a] = [distance[1] for distance in distances if distance[2] < 0.014]
        if len(geo_distances[station_a]) == 0:
            geo_distances[station_a] = [station_b]
    return closest_nodes, geo_distances

if __name__ == '__main__':
    df = pd.read_csv("test_small.csv")

    df["date"] = pd.to_datetime(df["date"])
    df = filter_df(df)
    df = df.sort_values(by=["station", "date"])

    stations = clip_stations(df)
    station_pairs = list(stations.keys())
    G = gen_graph(stations, df)
    graph = dict(G.edges)
    nodes = list(G.nodes)

    closest_nodes = {}
    for node in nodes:
        res = get_closest(node, nodes, graph, 0.05, 0.8)
        if res == []:
            continue
        closest_nodes[node] = res

    geo_closest_nodes, geo_distances = get_geographic_closest(station_pairs, df)

    # geo_distances contains everything geo_closest_nodes does
    # check if keys - 
    # check if values - 
    top_one_correct_count = 0
    within_one_square_mile_correct_count = 0
    total_count_a = 0

    for geo_node in geo_closest_nodes:
        total_count_a += 1
        try:
            nodes = [elem[0] for elem in closest_nodes[geo_node]]
            if geo_closest_nodes[geo_node] in nodes:
                top_one_correct_count += 1
        except:
            continue

    print("top one accuracy", top_one_correct_count/total_count_a)

    total_count_b = 0
    for geo_node in geo_distances:
        total_count_b += 1
        try:
            nodes = [elem[0] for elem in closest_nodes[geo_node]]
            if len([geo_dist for geo_dist in geo_distances if geo_dist in nodes]) > 0:
                within_one_square_mile_correct_count += 1
        except:
            continue

    print("within one mile", within_one_square_mile_correct_count/total_count_b)

# top one accuracy 0.5352941176470588
# within one mile 0.9941176470588236

# to dos?:
# histogram approach
# elucid distance again, all points
# elucid distance again, 7 day moving average
# 
