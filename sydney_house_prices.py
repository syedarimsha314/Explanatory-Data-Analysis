#!/usr/bin/env python
# coding: utf-8

# In[4]:


#Part 1 -Importing Libraries
import numpy as np
import math
import pandas as pd
#For data visualization
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns


# In[5]:


#Part 2 - Importing dataset
data = pd.read_csv(r"G:\Other computers\My Laptop\Gsolar_Rimsha\Rimsha_AdditionalPractice\Projects\House pricing\sydney_house_prices.csv")
suburbs = pd.read_csv(r"G:\Other computers\My Laptop\Gsolar_Rimsha\Rimsha_AdditionalPractice\Projects\House pricing\sydney_suburbs.csv")


# In[6]:


#Part - 3 Data Analysis
#Gathering information about dataset1 - data
print(data.describe())
data.shape
data.columns
data.info()
print("\n Number of rows: {0:,d} \n Number of features: {1:10,d}".format(data.shape[0],data.shape[1]))


# In[7]:


#Gathering information about dataset2 - suburbs
print(suburbs.describe())
suburbs.shape
suburbs.columns
suburbs.info()
print("\n Number of rows: {0:,d} \n Number of features: {1:3,d}".format(suburbs.shape[0],suburbs.shape[1]))


# In[8]:


#Counting missing data for each feature
data.isnull().sum()


# In[9]:


#Visualizing missing values
print("Visualizing missing data")
msno.bar(data)


# In[56]:


# Counting unique values for each feature of the data set
count= pd.DataFrame()
for i in data.columns:
    l = len(data[i].value_counts())
    new_row = {'feature' : i, 'total_unique_values' : l}
    count = count.append(new_row, ignore_index = True)
print(count)
print(data['suburb'].describe())


# In[10]:


print('Counting records for each unique value per feature')
for i in data.columns:
    print(data[i].value_counts().to_frame())


# In[11]:


#Findings
#1. 199,504 data points (records), 9 features
#2. Incorrect datatypes existing [Date] [bed] [car]
#3. Missing values existing [bed] [car]
#4. Inconsistent feature name
#5. Outliers: [sellPrice] [bed] [bath] [car]
#6. Other things to consider when doing analysis:
#7. House is the most sold property type
#8. Number of car spaces almost range between (1,10), most common is 1 or 2


# In[12]:


#Part 4 - Data Cleaning
#Creating new data frame "cdf" to store cleaned data:
#Step-1: Removing property type other than House and dropping column Proptype 
cdf = data.drop(index=(data[data.propType!='house'].index), columns='propType')

#Step-2: Remove postalCode >=3000 (keep Sydney only)
cdf.drop(cdf[cdf.postalCode >=3000].index, inplace=True)

#Step-3: Rename column for consistency
cdf.rename(columns={'Date':'date', 'Id':'id'}, inplace=True)

#Step-4 Correct data types
cdf['date'] = pd.to_datetime(cdf['date'])

#Step-5 # Handling Null values: replace null values with 0 ([bed], [car])
cdf = cdf.fillna(0)

#Step-6 # convert to best possible data types 
cdf = cdf.convert_dtypes()

cdf.info()


# In[13]:


#Rechecking if there's any missing values
cdf.isnull().sum()


# In[14]:


#Results after cleaning
#1. 170,105 data points left after cleaning
#2. All data types corrected (1 datetime, 1 string, 8 interger)
#3. Missing values are replaced by 0


# In[15]:


# Visualizing data with different bin size to explore distribution of data.
fig, ax = plt.subplots(5, 3, figsize=(15,10)) #5 Columns and 3 rows
col = ['bed','bath', 'car']
title = ['bedrooms', 'bathrooms', 'car spaces']

for j,c,t in zip(range(3),col,title):
    for i,s in zip(range(5),range(5,31,5)): #bin range is from 5 to 31(exclusive) and no. of steps are 5.
        ax[i,j].hist(c, data=cdf, bins=s, edgecolor="red", color="lightblue") 
        ax[i,j].text(10,20000,t + " / bin size=" + str(s) ,fontsize=10) #10 & 20000 are position of text 
        


# In[16]:


#Findings

#1. Most data distributed in range (0,10) for number of bedrooms, (0,5) for number of bathrooms, and (0,10) for number of car spaces.
#2. Those data points that are out of these ranges will be removed for analysis purpose.


# In[17]:


# Removing outliers [bed] (drop any bed>10)
bef = len(cdf)
cdf.drop(cdf[cdf.bed >10].index, inplace=True)


# In[18]:


aft = len(cdf)
print("Number of data points before: {0:,d}\nTotal data points removed: {2:,d}\nAfter removing [bed]: {1:,d}".format(bef, aft, bef-aft))


# In[19]:


# Removing outliers [bath] (drop any [bath]>5)
bef = len(cdf)
cdf.drop(cdf[cdf.bath >5].index, inplace=True)


# In[20]:


aft = len(cdf)
print("Number of data points before: {0:,d}\nTotal data points removed: {2:,d}\nAfter removing [bath]: {1:,d}".format(bef, aft, bef-aft))


# In[21]:


# Removing outliers [car] (drop any [car]>10)
bef = len(cdf)
cdf.drop(cdf[cdf.car >10].index, inplace=True)


# In[29]:


aft = len(cdf)
print("Number of data points before: {0:,d}\nTotal data points removed: {2:,d}\nAfter removing [car]: {1:,d}".format(bef, aft, bef-aft))


# In[147]:


var = cdf[['bed','bath', 'car']]
for v in var:
    print('Number of sale transactions per unique value for [', v,']')
    print(cdf[v].value_counts().to_frame())


# In[148]:


# Scatterplot to visualize house price over period

plt.scatter(cdf.date, cdf.sellPrice, marker='o', c='b',edgecolor='r', alpha=0.5)
plt.title("Sydney housing price over years")
plt.grid()
plt.show()


# In[149]:


#There are houses that are sold at prices extremely higher than others, these may affect the analysis results. To decide how to handle the outliers [sellPrice], I will try diffent approaches
#Approach no 1 - Calculate percentile to see data range and skewness: create df named p_cdf
#Approach no 2 - removing data by using Interquantile technique: create df named IQR_cdf


# In[38]:


# Calculate percentile of sell price to see data range, and spot outliers if any 
qr = [0.0005, 0.0010, 0.25, 0.5, 0.75, 0.9, 0.999, 0.9999]
print("Percentiles of house price")
for q in qr:
    print("{0:5,.2f} percentile: {1:15,.0f}".format(q*100, cdf.sellPrice.quantile(q)))
l=0.0009
u=0.9999
lq = cdf.sellPrice.quantile(l)
uq = cdf.sellPrice.quantile(u)
print("\nFindings\n{0}% of Sydney houses price between ${1:,.0f} and ${2:,.0f}".format((u-l)*100, lq, uq))


# In[39]:


# keeping only points between 0.09 - 99.99 percentitle
p_cdf = cdf.loc[(cdf.sellPrice>=lq) & (cdf.sellPrice<=uq)]


# In[40]:


p_cdf[['sellPrice']].describe().round(decimals = 0).T #T - transpose


# In[ ]:


# Calculate number of data points removed
bef = len(cdf)
aft = len(p_cdf)
print("Number of data points before: {0:,d}\nTotal data points removed: {2:,d} (or {3:,.2f}%)\nAfter removing [sellPrice] using percentile: {1:,d}"
      .format(bef, aft, bef-aft, (bef-aft)*100/bef))


# In[30]:


#Approach - 2 Define function to remove outliers
def remove_outlier_IQR(df):
    Q1=df.quantile(0.25)
    Q3=df.quantile(0.75)
    IQR=Q3-Q1
    df_final=df[~((df<(Q1-1.5*IQR)) | (df>(Q3+1.5*IQR)))]
    return df_final

def return_final(df,col):
    df_outlier_removed=remove_outlier_IQR(col)
    ind_diff=df.index.difference(df_outlier_removed.index)

    for i in range(0, len(ind_diff),1):
        df_final=df.drop([ind_diff[i]])
        df=df_final
    return df 


# In[31]:


# create new df named IQR_cdf
IQR_cdf = cdf
# apply return_final function to remove outliers
IQR_cdf=return_final(IQR_cdf, IQR_cdf['sellPrice'])


# In[34]:


print(IQR_cdf[['sellPrice']].describe().round(decimals = 0).T)


# In[35]:


# Calculate number of data points removed
bef = len(cdf)
aft = len(IQR_cdf)
print("Number of data points before: {0:,d}\nTotal data points removed: {2:,d} (or {3:,.2f}%)\nAfter removing [sellPrice] using IQR: {1:,d}"
      .format(bef, aft, bef-aft, (bef-aft)*100/bef))
print("IQR removed {0:,d} data points, roughly {1:,.2f}% of total data.".format(bef-aft,(bef-aft)*100/bef))


# In[36]:


cdf[['sellPrice']].describe().round(decimals = 0).T


# In[41]:


#Using scatterplot to visualize two approaches, compring how data distribute

fig, ax = plt.subplots(3, figsize=(10, 9), constrained_layout = True, sharex=True)

ax[0].scatter(cdf.date, cdf.sellPrice, marker='o', c='b',edgecolor='r', alpha=0.5)
ax[0].set_title("Sydney housing price over years before removing")

ax[1].scatter(p_cdf.date, p_cdf.sellPrice, marker='o', c='cyan',edgecolor='r', alpha=0.5)
ax[1].set_title("Sydney housing price over years using percentile")

ax[2].scatter(IQR_cdf.date, IQR_cdf.sellPrice, marker='o', c='cyan',edgecolor='r', alpha=0.5)
ax[2].set_title("Sydney housing price over years using IQR")

plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




