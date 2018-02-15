import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

zipMap = {'10003-8623': '10003'}
zillowDS = pd.read_csv(r'G:\dean\proj\2bedroom.csv')
zillowDS['RegionName'] = zillowDS['RegionName'].astype(str)
airbnbDS = pd.read_csv(r'G:\dean\proj\listings.csv', low_memory=False)
airbnbDS['zipcode'] = airbnbDS['zipcode'].astype(str)
airbnbDS['zipcode'] = airbnbDS.apply(lambda x: zipMap.get(x['zipcode'],x['zipcode']), axis = 1)
zipcodeDS = pd.read_excel(r'G:\dean\proj\NYC Zipcodes.xlsx')
zipcodeDS['ZipCode'] = zipcodeDS['ZipCode'].astype(str)

airbnbListings = airbnbDS[['id', 'last_scraped', 'summary', 'description', 'neighborhood_overview','notes', 'transit', 
                     'street', 'state', 'city','neighbourhood', 'neighbourhood_cleansed', 'neighbourhood_group_cleansed', 
                     'zipcode', 'market', 'smart_location', 'country_code', 'country', 'latitude', 
                     'longitude', 'property_type', 'room_type', 'bedrooms', 'amenities', 'price', 'weekly_price', 
                     'monthly_price',  'minimum_nights', 'maximum_nights', 'calendar_updated', 'has_availability',
                     'availability_30', 'availability_60', 'availability_90', 'availability_365']]
# Filter only New York States
airbnbListings = airbnbListings[(airbnbListings['state'].isin(['NY', 'New York']))]
airbnbListings['state'] = 'NY'
#Merge AirBnB and Zillow Datasets on Zipcode
airbnbListings = pd.merge(airbnbListings, zipcodeDS,left_on = 'zipcode', right_on = 'ZipCode', how = 'inner')
# Filter only New York City
airbnbListings['city_mod'] = 'New York'
#Remove the $ in Price column 
airbnbListings['price_numeric'] = airbnbListings.apply(lambda x: x['price'].strip('$'), axis = 1).apply(pd.to_numeric, errors='coerce')
#Replace NaN values with 0
airbnbListings = airbnbListings.fillna(0)
zillowDS = zillowDS.fillna(0)

#Filtering only 2 Bed Room Properties
airbnbListings = airbnbListings[airbnbListings['bedrooms'] == 2.0]

location_home = airbnbListings[['latitude', 'longitude']]

conditions = [
    (airbnbListings['room_type'] == 'Entire home/apt'),
    (airbnbListings['room_type'] == 'Private room'),
    (airbnbListings['room_type'] == 'Shared room')]
choices = ['blue', 'red', 'yellow']
location_home['color'] = np.select(conditions, choices, default='black')
location_home = location_home.drop_duplicates()
location_home.head()


%matplotlib inline
matplotlib.style.use('ggplot')
import os
from bokeh.io import output_file, show
from bokeh.models import (
  GMapPlot, GMapOptions, ColumnDataSource, Circle, Range1d, PanTool, WheelZoomTool, BoxSelectTool
)

map_options = GMapOptions(lat=40.7128, lng=-74.0060, map_type="roadmap", zoom=11)

plot = GMapPlot(x_range=Range1d(), y_range=Range1d(), map_options=map_options)
plot.title.text = "New York City Geographical Clusters based on Property Type"


plot.api_key = "AIzaSyCyDz9QHIZurVQwTK7HOZIuL1UnErgOOcI"
source = ColumnDataSource(
    data=dict(
        lat_home=location_home['latitude'],
        lon_home=location_home['longitude'],
        colors = location_home['color']
    )
)

circle = Circle(x="lon_home", y="lat_home", size=4, fill_color="colors", fill_alpha=0.8, line_color=None)

plot.add_glyph(source, circle)

plot.add_tools(PanTool(), WheelZoomTool(), BoxSelectTool())
show(plot)

roomProperty_DF = airbnbListings.groupby(['property_type','room_type']).price_numeric.mean()
roomProperty_DF = roomProperty_DF.reset_index()
roomProperty_DF=roomProperty_DF.sort_values('price_numeric',ascending=[0])
roomProperty_DF

import seaborn as sns

plt.figure(figsize=(12,12))
ax = plt.axes()
sns.heatmap(roomProperty_DF.groupby([
        'property_type', 'room_type']).price_numeric.mean().unstack(),annot=True, fmt=".0f",cmap="RdYlGn")
ax.set_title('Price break down by Room Type and Propery Type')

#Analyzing Room Type & Zip Code
roomType_DF = airbnbListings.groupby(['zipcode','room_type']).price_numeric.mean()
roomType_DF = roomType_DF.reset_index()
roomType_DF=roomType_DF.sort_values('price_numeric',ascending=[0])
roomType_DF

plt.figure(figsize=(25,25))
ax = plt.axes()
sns.heatmap(roomType_DF.groupby([
        'zipcode', 'room_type']).price_numeric.mean()[:35].unstack(),annot=True, fmt=".0f", cmap="RdYlGn")
ax.set_title('Price break down by Room Type and Zip Code for Top 35 Zip Codes')

plt.figure(figsize=(25,25))
ax = plt.axes()
sns.heatmap(roomType_DF.groupby([
        'zipcode', 'room_type']).price_numeric.mean()[-34:].unstack(),annot=True, fmt=".0f", cmap="RdYlGn")
ax.set_title('Price break down by Room Type and Zip Code for Bottom 35 Zip Codes - Risky Investment')

matplotlib.style.use('ggplot')

zipCodePrice.plot(kind='bar', 
           x=zipCodePrice.index.values,
           y='mean',
           color = '#66c2ff', 
           figsize =(25,15), 
           title = 'New York City Mean Price', 
           legend = False)

plt.ylabel('Mean Price')

matplotlib.style.use('ggplot')

zipCodePrice.plot(kind='bar', 
           x=zipCodePrice.index.values,
           y='count',
           color = '#66c2ff', 
           figsize =(25,15), 
           title = 'New York City Listings Count', 
           legend = False)

plt.ylabel('Listings Count')

#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.);

sns.set(style="ticks")
plt.subplots(figsize=(20,15))
#sns.boxplot(x="day", y="total_bill", hue="sex", data=tips, palette="PRGn")
ax = sns.boxplot(x="zipcode", y="price_numeric", data=airbnbListings)
plt.setp(ax.get_xticklabels(),rotation=90)
sns.despine(offset=10, trim=True)

summaryDS = airbnbListings[['summary','price_numeric']]
summaryDS = summaryDS.sort_values('price_numeric',ascending=[0])
top100DS = summaryDS.head(100)
top100DS.head()


from nltk.corpus import stopwords
import string
import nltk

words=''
for index,row in top100DS.iterrows():
    words = words +str(row['summary'])
    
string_punctuation = string.punctuation
ignoreChar=['\r','\n','',' ',"'s"]
nums=['0','1','2','3','4','5','6','7','8','9']
summary_data=nltk.word_tokenize(words)
words_only = [l.lower() for l in summary_data if l not in string_punctuation if l not in ignoreChar if l not in nums]
filtered_data=[word for word in words_only if word not in stopwords.words('english')] 
wnl = nltk.WordNetLemmatizer() 
final_data=[wnl.lemmatize(data) for data in filtered_data]
final_words=' '.join(final_data)
final_words[:50]

from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

wordcloud = WordCloud(width = 1000, height = 700).generate(final_words)
plt.figure(figsize=(18,12))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

#Analyzing Amenities
import re

amenitiesDF = airbnbListings[['amenities','price_numeric','id',]]
amenitiesDFTopper = amenitiesDF.sort_values('price_numeric',ascending=[0])
amenitiesDFtop=amenitiesDFTopper.head(30)
allemenities = ''
for index,row in amenitiesDFtop.iterrows():
    p = re.sub('[^a-zA-Z]+',' ', row['amenities'])
    allemenities+=p

allemenities_data=nltk.word_tokenize(allemenities)
filtered_data=[word for word in allemenities_data if word not in stopwords.words('english')] 
wnl = nltk.WordNetLemmatizer() 
allemenities_data=[wnl.lemmatize(data) for data in filtered_data]
allemenities_words=' '.join(allemenities_data)

from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

wordcloud = WordCloud(width = 1000, height = 700).generate(allemenities_words)
plt.figure(figsize=(18,12))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

df['zip']=df['latitude'].astype(str)+ ','
df['zip']=df['zip'] + df['longitude'].astype(str)

def find_zipcode(x):
    #print(x)
    x1= x.split(',')
    #print(x1,type(x1))
    rs=search.by_coordinate(float(x1[0]), float(x1[1]), radius=1, returns=1)
    #print(len(rs))
    if len(rs)>=1:
        #print(rs[0]['Zipcode'])
        return rs[0]['Zipcode']
    else:
        return 0
df['zipcoode']=df['zip'].apply(find_zipcode)

df_final_15=df[['zipcoode','price']]
df_final_15.columns=['zipcode','price_15']
df_final_15['zipcode']=(df_final_15['zipcode'].astype(int))
df_final_15.sort_values('zipcode',ascending=0)

#GroupBy Zipcode and Mean Price
df_final_15_Group=df_final_15.groupby(['zipcode']).price_15.mean()
pd2=pd.DataFrame(df_final_15_Group)
pd2=pd2.reset_index(0)
pd2['zipcode']=pd2['zipcode'].astype(int)
pd2
#Reading 2016 Airbnb listings data
df_2016=pd.read_csv(r'G:\dean\proj\Airbnb\Jan2016listings.csv')

df_2016['zip']=df_2016['latitude'].astype(str)+ ','
df_2016['zip']=df_2016['zip'] + df_2016['longitude'].astype(str)

def find_zipcode(x):
    #print(x)
    x1= x.split(',')
    #print(x1,type(x1))
    rs=search.by_coordinate(float(x1[0]), float(x1[1]), radius=1, returns=1)
    #print(len(rs))
    if len(rs)>=1:
        #print(rs[0]['Zipcode'])
        return rs[0]['Zipcode']
    else:
        return 0
df_2016['zipcoode']=df['zip'].apply(find_zipcode)

df_final_16=df_2016[['zipcoode','price']]
df_final_16.columns=['zipcode','price_16']
df_final_16.fillna(0).astype(int)
df_final_16['zipcode']=df_final_16['zipcode'].astype(float)
df_final_16.sort_values('zipcode',ascending=0)
df_final_16_Group=df_final_16.groupby(['zipcode']).price_16.mean()
#df_final_16.head(2)
pd1=pd.DataFrame(df_final_16_Group)
pd1=pd1.reset_index(0)
pd1['zipcode']=pd1['zipcode'].astype(int)
pd1

#Merging 2015 & 2016 Airbnb pricing data
Final_Prices=pd.merge(pd1,pd2,on='zipcode',how='inner')
Final_Prices=Final_Prices[Final_Prices['zipcode']!=0]
Final_Prices

df_final_17=airbnbListings[['ZipCode','price']]
df_final_17.columns=['zipcode','price_17']
df_final_17['price_17'] = df_final_17.apply(lambda x: x['price_17'].strip('$'), axis = 1).apply(pd.to_numeric, errors='coerce')
df_final_17_Group=df_final_17.groupby(['zipcode']).price_17.mean()
#df_final_16.head(2)
pd3=pd.DataFrame(df_final_17_Group)
pd3=pd3.reset_index(0)
pd3['zipcode']=pd1['zipcode'].astype(int)
pd3.head(2)
pd3=pd3[pd3.zipcode!=0]
pd3

airbnbListingsFinal['price_15_16_change']=(airbnbListingsFinal['price_16']-airbnbListingsFinal['price_15'])/airbnbListingsFinal['price_15']
airbnbListingsFinal['price_16_17_change']=(airbnbListingsFinal['price_17']-airbnbListingsFinal['price_16'])/airbnbListingsFinal['price_16']
airbnbListingsFinal['price_15_17_change']=(airbnbListingsFinal['price_17']-airbnbListingsFinal['price_15'])/airbnbListingsFinal['price_15']
airbnbListingsFinal['zipcode']=airbnbListingsFinal['zipcode'].astype(str)

zillowDSFinalAnnual
zillow_Year_Prices = pd.DataFrame(zillowDSFinalAnnual)
zillow_Year_Prices.columns=['ZipCode',         '1996',         '1997',         '1998',         '1999',
               '2000',         '2001',         '2002',         '2003',         '2004',
               '2005',         '2006',         '2007',         '2008',         '2009',
               '2010',         '2011',         '2012',         '2013',         '2014',
               '2015',         '2016',         '2017']
zillow_Year_Prices = zillow_Year_Prices[['ZipCode','2015','2016','2017']]
zillow_Year_Prices=zillow_Year_Prices.sort_values('ZipCode',ascending=1)
len((zillow_Year_Prices['ZipCode'].unique()))

zillowDS = zillowDS[zillowDS['State']=='NY']
airbnbListingsFinal['Total Investment_2017'] = 365 *0.75* airbnbListingsFinal['price_17']
airbnbListingsFinal['Total Investment_2016'] = 365 *0.75* airbnbListingsFinal['price_16']
airbnbListingsFinal['Total Investment_2015'] = 365 *0.75* airbnbListingsFinal['price_15']
zillowDSFinal = zillowDS.drop(['RegionID','City', 'State', 'Metro', 'CountyName', 'SizeRank' ], 1)
zillowDSFinal = zillowDSFinal.set_index('RegionName')
zillowDSFinal.columns = pd.to_datetime(zillowDSFinal.columns).to_period('M')
zillowDSFinalTS = zillowDSFinal.transpose()
zillowDSFinalAnnual = pd.DataFrame()
zillowDSFinalAnnual = zillowDSFinalTS.resample('A').mean()[-3:]
number_of_years = zillowDSFinalAnnual.index
zillowDSFinalAnnual = zillowDSFinalAnnual.transpose().reset_index()
airbnb_price = pd.merge(airbnbListingsFinal, zillowDSFinalAnnual,left_on = 'zipcode', right_on = 'RegionName', how = 'inner')
roi_ds = pd.DataFrame()
roi_ds['zipcode'] = airbnb_price['zipcode']
for each in number_of_years:
    roi_year = 'ROI_'+str(each)
    investment_Year = 'Total Investment_' + str(each)
    roi_ds[roi_year] = airbnb_price[investment_Year].div(airbnb_price[each]).replace(np.inf, 0)
roi_ds = roi_ds.set_index('zipcode')
roi_ds = roi_ds.transpose()
roi_ds
colors = ['red', 'blue','green', 'cyan', 'black','yellow', 'pink', 'purple', 'magenta', 
                   'maroon', 'brown','olive', 'lime', 'navy','lavender', 'coral', 'teal', 'violet',
                   'darkcyan', 'crimson','indigo', 'royalblue'] 


plt.figure(figsize=(15,15))
count = 0
for each in roi_ds.columns:
    plt.plot(range(len(roi_ds[each].values)),roi_ds[each].values, color = colors[count], marker='o', label = each)
    count = count + 1
plt.title('Return on Investment on 2 Bed Room Properties over the years (2015-2017)',  fontsize=16)
plt.ylabel('Return on Investment', fontsize=16)
plt.xticks(range(len(roi_ds[each].values)),[2015,2016,2017])
plt.legend(loc='upper right')