import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.preprocessing import StandardScaler, MinMaxScaler

def prepare_data(training_data, new_data):
    my_data = new_data.copy()

    num_of_sibling_median = training_data['num_of_siblings'].median()
    PCR3_median = training_data['PCR_03'].median()

    my_data['num_of_siblings'] = my_data['num_of_siblings'].fillna(num_of_sibling_median)
    my_data['PCR_03'] = my_data['PCR_03'].fillna(PCR3_median)

    my_data['SpecialProperty'] = my_data["blood_type"].isin(["O+", "B+"])
    my_data = my_data.drop('blood_type', axis=1)

    all_features = training_data.columns.tolist() 
    pcr_features = [feature for feature in all_features if feature.startswith("PCR")]

    list_to_standard_scale = [feature for feature in pcr_features 
                          if int(feature.split("_")[1]) in [3, 4, 5, 6, 7, 9, 10]]

    list_to_min_max = [feature for feature in pcr_features 
                   if int(feature.split("_")[1]) in [1, 2, 8]]
    
    features_to_standard_scale = my_data[list_to_standard_scale]
    features_to_min_max = my_data[list_to_min_max]
    
    
    standard_scaler = StandardScaler()
    minmax_scaler = MinMaxScaler(feature_range=(-1, 1))

    standard_scaler.fit(training_data[list_to_standard_scale])
    minmax_scaler.fit(training_data[list_to_min_max])
    

    standard_scaled_features = pd.DataFrame(standard_scaler.transform(features_to_standard_scale),
                                           columns=list_to_standard_scale,
                                           index=my_data.index)
    min_max_scaled_features = pd.DataFrame(minmax_scaler.transform(features_to_min_max),
                                           columns=list_to_min_max,
                                           index=my_data.index)
    
    my_data[list_to_standard_scale] = standard_scaled_features
    my_data[list_to_min_max] = min_max_scaled_features

    return my_data