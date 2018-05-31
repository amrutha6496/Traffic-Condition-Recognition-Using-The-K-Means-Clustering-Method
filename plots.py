import seaborn as sns

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

def Hist(dataset, xlabel, filename):  #Log Trip Duration His
    sns.set_style("darkgrid")
    sns.distplot(dataset, kde=True, rug=False)
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('./fig/'+filename+'.png',dpi=100)  

def CountPlot(dataset, column, label, s, a):   #Day of Week_Count_Plot
    sns.set_style("darkgrid")
    sns_plot = sns.factorplot(x=column,data=dataset,kind='count', palette="muted",size = s, aspect = a)
    plt.xlabel(label)
    plt.ylabel('Frequency')
    plt.tight_layout()
    sns_plot.savefig('./fig/'+label + '_Count_Plot.png',dpi=100)

def MapPlot(dataset, color = None, legend_plot = True):
    sns.set_style("darkgrid")
    city_long_border = (-74.03, -73.75)
    city_lat_border = (40.63, 40.85)
    if color is None:
        sns_plot = sns.lmplot('pickup_longitude', 'pickup_latitude', data=dataset,fit_reg=False, scatter_kws={"s": 2.5}, legend =legend_plot)
    else:
        sns_plot = sns.lmplot('pickup_longitude', 'pickup_latitude', data=dataset,hue = color, fit_reg=False, scatter_kws={"s": 2.5}, legend =legend_plot)
    #sns.plt.xlim(city_long_border)
    #sns.plt.ylim(city_lat_border)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    sns_plot.savefig('./fig/3_Map_Plot_PickUp.png',dpi=100)

def SpeedPlot(dataset,column,speed,label):
    sns.set_style("darkgrid")
    sns.pointplot(x=column, y=speed, data=dataset)
    plt.xlabel(label)
    plt.ylabel('Speed (km/hr)')
    plt.savefig('./fig/4_Speed_By_'+label+'.png',dpi=100)
 
def PlotFeatureImp(feature_importance_table):
    sns.set_style("darkgrid")
    sns.factorplot(x="importance", y="feature_name",data=feature_importance_table, kind="bar")
    plt.savefig('./fig/5_Feature_Imp_XGB.png',dpi=100)
    

def LonLatPlot(train):
    sns.set(style="darkgrid", palette="muted")
    f, axes = plt.subplots(2,2,figsize=(12, 12), sharex = False, sharey = False)#
    sns.despine(left=True) # if true, remove the ax
    sns.distplot(train['pickup_latitude'].values, color="m",bins = 100, ax=axes[0,0])
    sns.distplot(train['pickup_longitude'].values, color="g",bins =100, ax=axes[0,1])
    sns.distplot(train['dropoff_latitude'].values, color="m",bins =100, ax=axes[1,0])
    sns.distplot(train['dropoff_longitude'].values, color="g",bins =100, ax=axes[1,1])
    axes[0, 0].set_title('pickup_latitude')
    axes[0, 1].set_title('pickup_longitude')
    axes[1, 0].set_title('dropoff_latitude')
    axes[1, 1].set_title('dropoff_longitude')
    plt.setp(axes, yticks=[])
    plt.tight_layout()
    plt.savefig('./fig/6_Lat_Lon_Plot.png',dpi=100)

def ViolinPlot(train, y_, row_, hue_):
    sns.set(style="darkgrid")
    sns.violinplot(x=row_, 
                   y=y_, 
                   hue=hue_, data=train, split=True,
                   inner="quart")

def PivotPlot(train, y_, row_, col_):   #pickup_hour_and_pickup_weekday_Pivot 
    sns.set(style="darkgrid")
    pivot_table=pd.pivot_table(train, index=row_, columns=col_, values=y_, aggfunc=np.mean)
    sns.heatmap(pivot_table)
    plt.tight_layout()
    plt.savefig('./fig/'+ row_ + '_and_' + col_ + '_Pivot.png',dpi=100)
 
#Time_Distance_Plot   
def Time_Distance_Plot(train): 
    sample_ind = np.random.permutation(len(train))[:5000]
    sns.lmplot(x='distance_haversine', y='log_trip_duration', data = train.iloc[sample_ind], scatter_kws={"s": 10})   
    plt.savefig('./fig/8_Time_Distance_Pivot.png',dpi=100)

def Two_Hist_Plot(train, test):
    sns.set_style("darkgrid")
    sns.kdeplot(train['log_trip_duration'], shade=True, label = 'Train', color = 'b')
    sns.kdeplot(test['log_trip_duration'], shade=True, label = 'Test', color = 'r')
    plt.legend()
    plt.xlabel('Log(Duration)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('./fig/9_Two_His_Plot.png',dpi=100)            
