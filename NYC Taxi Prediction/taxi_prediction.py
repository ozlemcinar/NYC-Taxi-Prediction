import pandas as pd
from sklearn.cluster import KMeans
from collections import Counter
import numpy as np
from sklearn import preprocessing
from numpy import genfromtxt
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestRegressor
import pydot
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import sys
import csv

if __name__ == "__main__":

    print("\ncompiling project takes a long time, thanks for waiting\n")
    #taking data to dataframe
    taxi_df = pd.read_csv('green_tripdata_2016-04.csv')
    
    #used to calculate randomforest score
    y = taxi_df.Pickup_latitude
    X = taxi_df.drop('Pickup_latitude', axis=1)
    X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.33,random_state=42)

    #scaler_X = preprocessing.StandardScaler().fit_transform(X_train)
    #scaler_y = preprocessing.StandardScaler().fit_transform(y_train)
    #X_train_scaled = scaler_X.transform(X_train)
    #y_train_scaled = scaler_y.transform(y_train)
    
    #X1 = preprocessing.StandardScaler().fit_transform(X)
    #used to convert date to string
    le = preprocessing.LabelEncoder()

    #le.fit(taxi_df['VendorID'])

    #train_x = pd.get_dummies(taxi_df['Pickup_latitude'])

    #train_features, test_features, train_labels, test_labels = train_test_split(train_x, labels, test_size = 0.25, random_state = 42)
    # Instantiate model with 1000 decision trees
    #feature_list = list(train_x.columns)

    rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)

    # Import tools needed for visualization
    #rf.fit(train_features, train_labels);

    # Pull out one tree from the forest
    #tree = rf.n_estimators
    # Import tools needed for visualization

    #export_graphviz(tree, out_file = 'tree.dot', feature_names = feature_list, rounded = True, precision = 1)
    # Use dot file to create a graph
    #(graph, ) = pydot.graph_from_dot_file('tree.dot')
    # Write graph to a png file
    #graph.write_png('tree.png')


    sa = np.array(taxi_df)
    #print(sa[:,7])
    
    #taking pickup longtitude and pickup latitude to X dataframe
    arr = []
    i = 0
    j = 1
    #print(sa[0:2,7])
    while  i < len(sa):
        abc =sa[i:j,7]
        defg =sa[i:j,8]
        tfs =[abc,defg]
        #print(tfs[0])
        my_string_date = str(tfs[0]).replace('[', '')
        my_string_date_1 = str(my_string_date).replace(']', '')
        my_string_date_2 = str(tfs[1]).replace('[', '')
        my_string_date_3 = str(my_string_date_2).replace(']', '')
        arr.append([my_string_date_1,my_string_date_3])
        i+=1
        j+=1
    #for i in arr:
     #   print(i)
    newcsv = pd.DataFrame(arr)
    newcsv.columns = ['Pickup_longitude','Pickup_latitude']
    
    #calculating kmeans analysis
    cluster = KMeans(n_clusters=20).fit(newcsv).predict(newcsv)
    counter = Counter(cluster).values()
    topPickup = list(reversed((np.array(counter).argsort())))
    #print(top5_pickup_index)
    
    #print(pd.DataFrame(cluster))
    newcsvLabel = newcsv.join(pd.DataFrame(cluster))
    
    newcsvLabel.columns = ['Pickup_longitude', 'Pickup_latitude', 'Cluster']
    dataf1 = newcsvLabel[newcsvLabel['Cluster'].apply(lambda x:x in topPickup )]
    
    #writing output fo new csv file
    dataf1.to_csv('locationForPickup.csv', header=0)
    
    print("First process is completed\n")
    
    
    sss1 = np.array(taxi_df)
    
    # taking lpep_pickup_datetime
    arr1 = []
    k1 = 0
    k2 = 1
    #print(sa[0:2,7])
    while  k1 < len(sss1):
        abc =sss1[k1:k2,1]
        #print(tfs[0])
        my_string_date = str(abc).replace("'", '')
        my_string_date_1 = str(my_string_date).replace(']', '')
        my_string_date_2 =str(my_string_date_1).replace('[', '')
        arr1.append(my_string_date_2)
        k1+=1
        k2+=1
    
    X1 = pd.DataFrame(arr1)
    
    # taking Lpep_dropoff_datetime
    arr2 = []
    k3 = 0
    k4 = 1
    #print(sa[0:2,7])
    while  k3 < len(sss1):
        abc =sss1[k3:k4,2]
        #print(tfs[0])
        my_string_date = str(abc).replace("'", '')
        my_string_date_1 = str(my_string_date).replace(']', '')
        my_string_date_2 =str(my_string_date_1).replace('[', '')
        arr2.append(my_string_date_2)
        k3+=1
        k4+=1
    #for i in arr2:
     #   print(i)
    X2 = pd.DataFrame(arr2)
    
    #converting time and store them in X3
    k5=0
    arr3 = []
    while k5 < len(X2):
        r1 = datetime.strptime(X2[0][k5],'%Y-%m-%d %H:%M:%S')
        r2 = datetime.strptime(X1[0][k5],'%Y-%m-%d %H:%M:%S')
        arr3.append([r1,r2])
        #z1 = (r1 - r2)
        k5+=1

    X3 = pd.DataFrame(arr3)
    #print(X3)
    k6 = 0
    arr4 = []
    X4 = pd.DataFrame()
    #calculating trip_time
    while k6< len(X3):
        cikartma = (X3[0][k6] - X3[1][k6]) / np.timedelta64(1,'M')
        arr4.append(cikartma)
        k6+=1
    #print(cikartma)
    X4 = pd.DataFrame(arr4)
    arr5 = []
    k7 = 0
    k8 = 1
    #print(sa[0:2,7])
    #calculating fare_per_time
    while  k7 < len(sss1):
        abc =sss1[k7:k8,18]
        arr5.append(abc)
        k7+=1
        k8+=1
    k9 = 0
    arr6 = []
    while k9<len(sss1):
        if arr5[k9]>0 and arr4[k9]>0:
            bolum = (arr5[k9]/arr4[k9])
            arr6.append(bolum)
        k9+=1
    X5 = pd.DataFrame(arr6)
    
    
    longitude = taxi_df.loc[:, ['Pickup_longitude']]
    latitude = taxi_df.loc[:, ['Pickup_latitude']]
    newcsv = pd.concat([longitude, latitude, X5], axis=1)
    newcsv.columns = ['longitude', 'latitude', 'fare_per_time']
    newcsv = newcsv[np.isfinite(newcsv['fare_per_time'])]
    #print X
    newcsv = newcsv[(newcsv['latitude'] != 0) & (newcsv['longitude'] != 0)]
    newcsv = newcsv.reset_index(drop=1)
    # kmeans analysis
    
    #randomforest analysis
    X1 = preprocessing.StandardScaler().fit_transform(newcsv)
    clf = RandomForestRegressor()
    clf = preprocessing.StandardScaler().fit_transform(X1)
    #print(clf)

    #X_train_scaled = scaler_X.transform(X_train)

    #X_test_scaled = scaler_X.transform(X1)
    #y_test_scaled = scaler_y.transform(y_test)
    #print(X1)
    
    #kmeans analysis
    k = 10
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X1)
    cluster = kmeans.predict(X1)
    
    #score analyis
    r2_score(clf,X1)
    newcsvLabel = pd.DataFrame(newcsv).join(pd.DataFrame(cluster))
    
    newcsvLabel.columns = ['longitude', 'latitude', 'fare_per_time', 'Cluster']
    clusterAndFare = newcsvLabel.groupby(['Cluster'], sort=False)['fare_per_time'].mean()
    clusterAndFare = clusterAndFare.sort_index()
    topMoney = np.array(clusterAndFare).argsort()[-5:]
    topLucrative = list(reversed(topMoney))
    
    dataf2 = newcsvLabel[newcsvLabel['Cluster'].apply(lambda x: x in topLucrative)]
    #storing dataset in in csv file
    dataf2.to_csv('locationForLucrative.csv', header=0)
    
    
    

    


    

