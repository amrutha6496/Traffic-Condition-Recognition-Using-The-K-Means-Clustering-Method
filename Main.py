# Import libraries
from modules.plots import *
from modules.map import *
import pandas as pd
import numpy as np
from scipy import stats      #statstics fn are located in the sub pachage scipy.stats
from sklearn.cluster import KMeans
import folium           #Interactive maps
import xgboost as xgb  #provides the gradient boosting 

def FeatureEng(train):
    #Create new features for date/time & fig4  weekday
    train['pickup_datetime'] = pd.to_datetime(train.pickup_datetime)
    train['pickup_weekday_name'] = train['pickup_datetime'].dt.weekday_name
    train['pickup_weekday'] = train['pickup_datetime'].dt.weekday
    train['pickup_hour_weekofyear'] = train['pickup_datetime'].dt.weekofyear
    train['pickup_hour'] = train['pickup_datetime'].dt.hour
    
    #Calculate distance & map
    train['distance_haversine'] = haversine_array(train['pickup_latitude'].values, 
                                             train['pickup_longitude'].values, 
                                             train['dropoff_latitude'].values, 
                                             train['dropoff_longitude'].values)

    train['distance_manhattan'] = manhattan_distance(train['pickup_latitude'].values, 
                                                     train['pickup_longitude'].values, 
                                                     train['dropoff_latitude'].values, 
                                                     train['dropoff_longitude'].values)
    #Location clustering
    coords = np.vstack((train[['pickup_latitude', 'pickup_longitude']].values,
                        train[['dropoff_latitude', 'dropoff_longitude']].values))
    sample_ind = np.random.permutation(len(coords))[:5000]
    kmeans = KMeans(n_clusters=100).fit(coords[sample_ind])
    
    train['pickup_cluster'] = kmeans.predict(train[['pickup_latitude', 'pickup_longitude']])
    train['dropoff_cluster'] = kmeans.predict(train[['dropoff_latitude', 'dropoff_longitude']])
    
    # Return dataframe
    return(train)

# Read train set
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Feature Engineering
train['log_trip_duration'] = np.log(train['trip_duration'].values + 1)    
city_long_border = (-74.03, -73.75)
city_lat_border = (40.63, 40.85)
train = train[(train.pickup_latitude > city_lat_border[0]) & (train.pickup_latitude < city_lat_border[1])]
train = train[(train.dropoff_latitude > city_lat_border[0]) & (train.dropoff_latitude < city_lat_border[1])]
train = train[(train.pickup_longitude > city_long_border[0]) & (train.pickup_longitude < city_long_border[1])]
train = train[(train.dropoff_longitude > city_long_border[0]) & (train.dropoff_longitude < city_long_border[1])]

# For both train and test
train = FeatureEng(train)
test = FeatureEng(test)

# Visualization           
LonLatPlot(train)                                                            #6  Lat_Lon_Plot          
ViolinPlot(train, 'log_trip_duration','passenger_count', 'vendor_id')
Hist(train['log_trip_duration'], 'Log (Duration)', 'Log Trip Duration His')  #1   Log Trip Duration His
CountPlot(train, 'pickup_weekday_name','Day of Week', 4, 2)                  #2  Day of Week_Count_Plot
PivotPlot(train, 'log_trip_duration', 'pickup_hour', 'pickup_weekday')       #7  pickup_hour_and_pickup_weekday_Pivot
Time_Distance_Plot(train)                                                    #8  Time_distance_pivot

# Calculate distance and speed
train['avg_speed_h'] = train['distance_haversine'] / train['trip_duration'] * 3600# in the unit of km/hr
train['avg_speed_m'] = train['distance_manhattan'] / train['trip_duration'] * 3600# in the unit of km/hr
SpeedPlot(train,'pickup_weekday','avg_speed_h','Weekday')
SpeedPlot(train,'pickup_hour','avg_speed_h','Hour of Day')

# Map Locations of all pickups
MapPlot(train)
MapPlot(train[:10000], 'pickup_cluster', legend_plot = False)

# Create a simple leafllet map
locations = train[['pickup_latitude', 'pickup_longitude']]
locationlist = locations.values.tolist()[:300]
map_object = folium.Map(location=[40.767937,-73.982155], tiles='CartoDB dark_matter', zoom_start=12)
#marker_cluster = folium.MarkerCluster().add_to(map_object)
for point in range(0, len(locationlist)):
    folium.Marker(locationlist[point]).add_to(map_object)
folium.Map.save(map_object, "Leaflet_Map.html")

# Exclude feature not for model development
feature_names = list(train.columns)
do_not_use_for_training = ['id', 'log_trip_duration', 'pickup_datetime', 'dropoff_datetime', 'trip_duration', 
                           'pickup_weekday_name','pickup_date', 'avg_speed_h', 'avg_speed_m','store_and_fwd_flag']
feature_names = [f for f in train.columns if f not in do_not_use_for_training]

# Examine features
feature_stats = pd.DataFrame({'feature': feature_names})
feature_stats['train_mean'] = np.nanmean(train[feature_names].values, axis=0)
feature_stats['train_std'] = np.nanstd(train[feature_names].values, axis=0)
feature_stats['train_nan'] = np.mean(np.isnan(train[feature_names].values), axis=0)

## Start training the XGB - eXtreme Gradient Boosting
# Note, the parameters should be further tuned
xgb_pars = {'eta': 0.3, 'colsample_bytree': 0.3, 'max_depth': 10,
            'subsample': 0.5, 'lambda': 1, 'booster' : 'gbtree', 'silent':0,
            'eval_metric': 'rmse', 'objective': 'reg:linear'}
Xtr = train[feature_names]
Ytr = train['log_trip_duration']
dtrain = xgb.DMatrix(Xtr, label=Ytr) #xgboost dataset

# Use CV to determind number of rounds
max_num_round = 100
xgb_cv = xgb.cv(xgb_pars, dtrain, max_num_round, nfold=5,
       callbacks=[xgb.callback.print_evaluation(show_stdv=True)])

#Use the best number of rounds from CV to train the model again
selected_num_round = 92
xgb_model = xgb.train(xgb_pars, dtrain, selected_num_round)

# Examine feature importance & fig 2
feature_importance_dict = xgb_model.get_fscore()
feature_importance_table = pd.DataFrame({'feature_name': list(feature_importance_dict.keys()), 
                                         'importance': list(feature_importance_dict.values())})
feature_importance_table.sort_values(by ='importance', ascending=False, inplace= True)
PlotFeatureImp(feature_importance_table)

# Apply model on test dataset
Xte = test[feature_names]
dtest = xgb.DMatrix(Xte) #xgboost dataset
Yte = xgb_model.predict(dtest)
test['log_trip_duration'] = Yte
test['trip_duration'] = np.exp(Yte) - 1
test[['id', 'trip_duration']].to_csv('sam.csv', index=False)
Two_Hist_Plot(train, test)


