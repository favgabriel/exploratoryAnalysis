import sqlite3
import csv
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

con = sqlite3.connect("gorcery.db")

#con.execute("create table if not exists t (Member_number,Date, itemDescription);")
# con.execute("drop table t;")
# con.commit()
# with open('Groceries_dataset.csv','r') as fin:
#     dr = csv.DictReader(fin)
#     to_db = [(i['Member_number'],i['Date'],i['itemDescription']) for i in dr]
#     print(to_db)
# con.executemany("insert into t (Member_number,Date, itemDescription) values(?,?,?);",to_db)
# con.commit()

#recency =con.execute("select Member_number as id, round(julianDay()-max(Date)) as recency from t group by Member_number").fetchall()
#frequency =con.execute("select Member_number as id, count(Member_number) as frequency from t group by Member_number").fetchall()
#monetary =con.execute("select Member_number as id, sum(Member_number) as monetary from t group by Member_number").fetchall()

vn = pd.read_sql("select Member_number as id, round(julianDay()-max(Date)) as recency, count(Member_number) as frequency, sum(Member_number) as monetary from t group by Member_number",con)
#print(vn.shape)
recency_score=pd.qcut(vn['recency'],q=3,labels=range(1,4))
frequency_score=pd.qcut(vn['frequency'],q=3,labels=range(1,4))
monetary_score = pd.qcut(vn['monetary'],q=3,labels=range(1,4))
vn = vn.assign(R = recency_score.values).assign(F = frequency_score.values).assign(M= monetary_score.values)
def rfm_score(row):
    return str(row['R'])+str(row['F'])+str(row['M'])
vn["RFM_Score"] = vn.apply(rfm_score,axis=1)

vn['segment']=""

best = list(vn.loc[vn['RFM_Score']=='333'].index)
lost_cheap = list(vn.loc[vn['RFM_Score']=='111'].index)
lost = list(vn.loc[vn['RFM_Score']=='133'].index)
lost_almost = list(vn.loc[vn['RFM_Score']=='233'].index)

for i in vn.index:
    if i in lost_cheap:
        vn.segment.iloc[i] = 'lost Cheap Customers'
    elif i in lost:
        vn.segment.iloc[i] = 'lost Customer'
    elif i in best:
        vn.segment.iloc[i] = 'best Customer'
    elif i in lost_almost:
        vn.segment.iloc[i] = 'Almost lost'
    else:
        vn.segment.iloc[i] = 'others'

loyal = list(vn.loc[vn['F']==3].index)
loyal2 = []

for i in loyal:
    if i not in best and i not in lost_cheap and i not in lost_almost and i not in lost:
        loyal2.append(i)

for i in vn.index:
    if i in loyal2:
        vn.segment.iloc[i]='loyal customers'

big = list(vn.loc[vn['M']==3].index)
big2 = []

for i in big:
    if i not in best and i not in lost_cheap and i not in lost_almost and i not in lost:
        big2.append(i)

for i in vn.index:
    if i in big2:
        vn.segment.iloc[i]='big spenders'

def check_skew(df_skew,column):
    skew = stats.skew(df_skew[column])
    skewtest = stats.skewtest(df_skew[column])
    plt.title('Distribution'+ column)
    sns.distplot(df_skew[column])
    print("{}'s: Skew: {}, : {}".format(column,skew,skewtest))
    return

print(vn.head())

#plt.figure(figsize=(9,9))

# plt.subplot(3,1,1)
# check_skew(vn,'recency')
#
# plt.subplot(3,1,2)
# check_skew(vn,'frequency')
#
# plt.subplot(3,1,3)
# check_skew(vn,'monetary')
# plt.tight_layout()
#plt.show()

vn.drop('segment', inplace=True, axis=1)

scaler = StandardScaler()
scaler.fit(vn)

rfm_tabled_scaled = scaler.transform(vn)

rfm_tabled_scaled = pd.DataFrame(rfm_tabled_scaled,columns=vn.columns)
print(rfm_tabled_scaled.head())

from scipy.spatial.distance import cdist
distortions = []
inertias = []
mapping1 = {}
mapping2 = {}
K = range(1,10)

for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(rfm_tabled_scaled)

    distortions.append(sum(np.min(cdist(rfm_tabled_scaled,kmeanModel.cluster_centers_,'euclidean'),axis=1))/rfm_tabled_scaled.shape[0])
    inertias.append(kmeanModel.inertia_)
    mapping1[k] = sum(np.min(cdist(rfm_tabled_scaled,kmeanModel.cluster_centers_,'euclidean'),axis=1))/rfm_tabled_scaled.shape[0]
    mapping2[k] = kmeanModel.inertia_

#plt.subplot(3,1,3)
#plt.plot(K, inertias, 'bx-')
#plt.xlabel('Values of k')
#plt.ylabel('Inertia')
#plt.title('The elbow method')

def snake_plot(normalised_df, df_rfm_kmeans, df_rfm_orig):
    normalised_df = pd.DataFrame(normalised_df, index=vn.index, columns=vn.columns)
    normalised_df['Cluster'] = df_rfm_kmeans['Cluster']
    df_melt = pd.melt(normalised_df.reset_index(), id_vars=['id','Cluster'],value_vars=['recency','frequency','monetary'],var_name='Metric', value_name='Value')
    plt.xlabel('Metric')
    plt.ylabel('Value')
    sns.pointplot(data=df_melt,x='Metric', y='Value', hue='Cluster')
    return


def kmeans(normalised_df, clusers_number, original_df):
    kmeans = KMeans(n_clusters=clusers_number,random_state=1)
    kmeans.fit(normalised_df)
    cluster_label = kmeans.labels_
    df_new = original_df.assign( Cluster = cluster_label)
    model = TSNE(random_state=1)
    transformed = model.fit_transform(df_new)

    plt.title('Flattened Graph of {} clusters'.format(clusers_number))
    sns.scatterplot(x=transformed[:,0], y=transformed[:,1],hue=cluster_label, style=cluster_label, palette='Set1')
    return df_new

def rfm_values(df):
    df_new = df.groupby(['Cluster']).agg({'recency':'mean',
                                          'frequency':'mean',
                                          'monetary':['mean','count']}).round(0)
    return df_new

plt.figure(figsize=(9,9))

plt.subplot(3,1,1)
df_rfm_k3 = kmeans(rfm_tabled_scaled,3,vn)

plt.subplot(3,1,2)
df_rfm_k4 = kmeans(rfm_tabled_scaled,4,vn)

plt.subplot(3,1,3)
df_rfm_k5 = kmeans(rfm_tabled_scaled,5,vn)

plt.tight_layout()


plt.figure(figsize=(10,10))

plt.subplot(3,1,1)
df_rfm_3 = snake_plot(rfm_tabled_scaled,df_rfm_k3,vn)

plt.subplot(3,1,2)
df_rfm_4 = snake_plot(rfm_tabled_scaled,df_rfm_k4,vn)

plt.subplot(3,1,3)
df_rfm_5 = snake_plot(rfm_tabled_scaled,df_rfm_k5,vn)

plt.tight_layout()

print(rfm_values(df_rfm_k4))

plt.show()
