# %%
import pandas as pd
import numpy as np
import datetime as dt

file_path = r"C:\Users\EIPLPC038\Documents\Invoice Product Folder\Invoice wise detail_2026-08-18.csv"

df = pd.read_csv(file_path, encoding="cp1252")

print("File loaded successfully!")
print("Shape:", df.shape)
display(df.head())


# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn: model selection, preprocessing, metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity

from scipy.sparse import hstack, csr_matrix
from sklearn.cluster import KMeans

import pickle
import json

# Styling
sns.set_theme(style='whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
pd.set_option('display.float_format', lambda x: f'{x:,.4f}')

print("✅ All imports successful!")

# %%

print(df.tail(10))

# %%
df.describe()

# %%
df.shape

# %%
df.index

# %%
df.size

# %%
df.columns

# %%
df.info()

# %% [markdown]
# DATA PREPROCESSING

# %%
df.isnull()

# %%
df.isnull().sum()

# %% [markdown]
# MODE IMPUTATION

# %%
df['State Name'] = df['State Name'].fillna(df['State Name'].mode()[0])



# %%
df['Customer GSTIN'] = df['Customer GSTIN'].fillna(df['Customer GSTIN'].mode()[0])



# %%
df['Contact Name'] = df['Contact Name'].fillna(df['Contact Name'].mode()[0])



# %%
df['Accouting Doc Type'] = df['Accouting Doc Type'].fillna(df['Accouting Doc Type'].mode()[0])


# %%
df['Customer State Name'] = df['Customer State Name'].fillna(df['Customer State Name'].mode()[0])



# %% [markdown]
# Mean Imputation

# %%
df['Accouting Doc'] = df['Accouting Doc'].fillna(df['Accouting Doc'].mean())


# %%
df['LR No'] = df['LR No'].fillna(df['LR No'].mode()[0])


# %% [markdown]
# MEDIAN IMPUTATION

# %%
df['Accouting Doc'] = df['Accouting Doc'].fillna(df['Accouting Doc'].median())


# %%
df['Delivery No'] = df['Delivery No'].fillna(df['Delivery No'].median())


# %%
df.isnull().sum()

# %% [markdown]
# OUTLIER DETECTION AND TREATMENT

# %%
# Select numeric columns
numeric_cols = df.select_dtypes(include=np.number).columns

# Detect outliers using IQR
outlier_counts = {}

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = ((df[col] < lower_bound) | (df[col] > upper_bound))

    # Sum = number of outliers
    outlier_counts[col] = outliers.sum()

# Display outlier count for each column
outlier_counts = pd.Series(outlier_counts)

print(outlier_counts)


# %%
print(df.select_dtypes(include=np.number).columns.tolist())


# %%
import pandas as pd
import numpy as np

# ============================================================
# STEP 1: Select numerical columns
# ============================================================

numeric_columns = df.select_dtypes(include=np.number).columns

print("Numerical columns:")
print(list(numeric_columns))


# ============================================================
# STEP 2: Function to calculate outliers using IQR
# ============================================================

def count_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = (
        (data[column] < lower_bound) |
        (data[column] > upper_bound)
    )
    
    return outliers.sum(), lower_bound, upper_bound


# ============================================================
# STEP 3: OUTLIERS BEFORE CAPPING
# ============================================================

before_results = []

for col in numeric_columns:
    
    outlier_count, lower_bound, upper_bound = count_outliers(df, col)
    
    before_results.append({
        'Column': col,
        'Lower Bound': lower_bound,
        'Upper Bound': upper_bound,
        'Outliers Before Capping': outlier_count
    })

before_df = pd.DataFrame(before_results)

print("\n========== BEFORE CAPPING ==========")
display(before_df)


# ============================================================
# STEP 4: IQR OUTLIER CAPPING
# ============================================================

for col in numeric_columns:
    
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Cap lower outliers
    df[col] = df[col].clip(lower=lower_bound)
    
    # Cap upper outliers
    df[col] = df[col].clip(upper=upper_bound)


print("\nOutlier capping completed successfully!")


# ============================================================
# STEP 5: OUTLIERS AFTER CAPPING
# ============================================================

after_results = []

for col in numeric_columns:
    
    outlier_count, lower_bound, upper_bound = count_outliers(df, col)
    
    after_results.append({
        'Column': col,
        'Lower Bound': lower_bound,
        'Upper Bound': upper_bound,
        'Outliers After Capping': outlier_count
    })

after_df = pd.DataFrame(after_results)

print("\n========== AFTER CAPPING ==========")
display(after_df)


# ============================================================
# STEP 6: COMBINED BEFORE & AFTER RESULTS
# ============================================================

comparison = before_df[['Column', 'Outliers Before Capping']].merge(
    after_df[['Column', 'Outliers After Capping']],
    on='Column'
)

print("\n========== OUTLIER TREATMENT SUMMARY ==========")
display(comparison)


# ============================================================
# STEP 7: CHECK WHETHER ALL OUTLIERS ARE REMOVED
# ============================================================

total_outliers_after = comparison['Outliers After Capping'].sum()

print("\nTotal outliers before capping:",
      comparison['Outliers Before Capping'].sum())

print("Total outliers after capping:",
      total_outliers_after)

if total_outliers_after == 0:
    print("\nSUCCESS: All outliers have been capped. Zero outliers remain.")
else:
    print("\nSome values are still being identified as outliers.")


# %%
print(outlier_count)

# %% [markdown]
# ### Handling Duplicate Entries

# %%
print("Number of duplicate rows:", df.duplicated().sum())



# %%
#Remove duplicate rows
df.drop_duplicates(inplace=True)

# Verify
print("Duplicate rows remaining:", df.duplicated().sum())
print("New shape:", df.shape)


# %%
categorical_columns = [
    'State Name',
    'Part Groups'
    'Contact Name'
]

# %%
categorical_columns = [
    col for col in categorical_columns
    if col in df.columns
]

print("Using categorical columns:")
print(categorical_columns)

# %%
encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=True
)

categorical_matrix = encoder.fit_transform(
    df[categorical_columns]
)

print("Categorical matrix shape:", categorical_matrix.shape)

# %%
tfidf = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

material_matrix = tfidf.fit_transform(
    df["Material Desc"]
    .fillna("")
    .astype(str)
)

print(
    "Material TF-IDF matrix shape:",
    material_matrix.shape
)

# %%
# Inspect The vocabulary
feature_names = tfidf.get_feature_names_out()

print("Number of vocabulary terms:")
print(len(feature_names))

print("\nFirst 50 terms:")
print(feature_names[:50])

# %%
# Combining Categorical adn Text Vectors Using Hstack
combined_matrix = hstack([
    categorical_matrix,
    material_matrix
])

combined_matrix = csr_matrix(
    combined_matrix
)

print(
    "Final vector matrix shape:",
    combined_matrix.shape
)

# %%
query_index = 0

query_vector = combined_matrix[
    query_index
]

# %%
similarities = cosine_similarity(
    query_vector,
    combined_matrix
).flatten()

# %%
print(similarities[:10])

# %%
top_indices = (
    similarities
    .argsort()[::-1]
)

# %%
top_indices = top_indices[
    top_indices != query_index
]

# %%
top_indices = top_indices[:10]

# %% [markdown]
# ## Exploratory Data Analysis


# %%
df_sales = df.copy()


# %%
import pandas as pd
import matplotlib.pyplot as plt

# Make a copy if df_sales already exists
df_sales = df.copy()

# Convert Posting Date to datetime
df_sales['Posting Date'] = pd.to_datetime(
    df_sales['Posting Date'],
    errors='coerce'
)

# Convert Total VALUE to numeric as well
df_sales['Total VALUE'] = pd.to_numeric(
    df_sales['Total VALUE'],
    errors='coerce'
)

# Check the conversion
print(df_sales[['Posting Date', 'Total VALUE']].dtypes)

# Create plots
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

# Revenue by month
df_sales.groupby(
    df_sales['Posting Date'].dt.month
)['Total VALUE'].sum().plot(
    kind='bar',
    ax=axes[0],
    color='#4C72B0'
)

axes[0].set_title('Revenue by Month')
axes[0].set_xlabel('Month')
axes[0].set_ylabel('Revenue')

plt.tight_layout()
plt.show()


# %% [markdown]
# So there are three major stories here:
# Growth → sudden drop → recovery/peak → decline
# 
# From Month 3 through Month 7, revenue increases almost every month:
# 
# $15K → $23K → $38K → $58K → $79K
# 
# That's a very positive trend.
# 
# But then:
# Month 7 → Month 8: $79K → $41K
# That's almost a 48% drop.
# 
# After the Month 8 decline:
# $41K → $102K → $118K
# 
# That's a dramatic recovery.
# 
# In fact, Month 10 is approximately 3× Month 8's revenue.
# 
# This could indicate:
# 
# A successful marketing campaign
# Seasonal demand
# A major customer/order
# New product launch
# Pricing changes

# %% [markdown]
# ## 05. Top-Selling Products — by Part Group and Material Desc
# This is the core deliverable: which **Part Group**s and which **Material Desc**s actually move.

# %%
top_by_group = (df_sales.groupby('Part Group')
                .agg(Total_Qty=('Quantity','sum'), Total_Revenue=('Total VALUE','sum'), Orders=('Bill Doc','nunique'))
                .reset_index().sort_values('Total_Qty', ascending=False))
top_by_group

# %%
df.columns = df.columns.str.strip()

print(df.columns.tolist())

# %%
df.iloc[2:5]

# %%
top_by_desc = (df_sales.groupby(['Part Group','Material Desc'])
               .agg(Total_Qty=('Quantity','sum'), Total_Revenue=('Total VALUE','sum'), Orders=('Bill Doc','nunique'),
                    Customers=('Customer','nunique'))
               .reset_index().sort_values('Total_Qty', ascending=False))

print("Top 15 products by quantity sold:")
top_by_desc.head(15)

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

top_by_group.set_index('Part Group')['Total_Qty'].plot(kind='barh', ax=axes[0], color='#C44E52')
axes[0].set_title('Total Quantity Sold by Part Group')
axes[0].invert_yaxis()

top15 = top_by_desc.head(15).sort_values('Total_Qty')
axes[1].barh(top15['Material Desc'], top15['Total_Qty'], color='#8172B2')
axes[1].set_title('Top 15 Products by Quantity Sold (Material Desc)')

plt.tight_layout()
plt.show()

# %% [markdown]
# 1. Total Quantity Sold by Part Group (Left Chart) BRAKE PARTS: Highest selling category by a massive margin (over 2,000,000 units).  
# BRAKE FLUID: Second highest category (approximately 550,000 units).  
# LUBES: Third highest category (approximately 150,000 units). 
#  OTHERS: Minor category (approximately 100,000 units).
#  EMS, AC GAS, SUSPENSION: Minimal to negligible sales volumes relative to brakes.
# 
# 2. Top 15 Products by Quantity Sold (Right Chart) Top Performer: BRAKE FLUID 1/4 LTR DOT3 (Non Petroleum) is the highest-selling individual product (over 200,000 units).  
# Runner Up: BRAKE FLUID 100ml DOT3 (Non Petroleum) is second (approximately 150,000 units). 
#  Key Components: SHOE KIT and BRAKE FLUID 1/4 Lt DOT4 follow closely, both exceeding 100,000 units.Volume Baseline: The lowest item in the top 15 (POWER STEERING OIL BOTTLE 500ml) still accounts for nearly 50,000 units.

# %%
top_by_revenue = top_by_desc.sort_values('Total_Revenue', ascending=False).head(15)
top_by_revenue[['Part Group','Material Desc','Total_Qty','Total_Revenue','Orders','Customers']]

# %% [markdown]
# ---
# # 09. Time Feature Engineering

# %%
# Make sure Posting Date is datetime
df_sales['Posting Date'] = pd.to_datetime(
    df_sales['Posting Date'],
    errors='coerce'
)

# Create date-based features
df_sales['Year'] = df_sales['Posting Date'].dt.year
df_sales['MonthNum'] = df_sales['Posting Date'].dt.month
df_sales['Quarter'] = df_sales['Posting Date'].dt.quarter
df_sales['DayOfWeek'] = df_sales['Posting Date'].dt.dayofweek

# Use Int64 instead of int because missing dates produce <NA>
df_sales['WeekOfYear'] = (
    df_sales['Posting Date']
    .dt.isocalendar()
    .week
    .astype('Int64')
)

df_sales['IsWeekend'] = (
    df_sales['DayOfWeek']
    .isin([5, 6])
    .astype('Int64')
)

# Display results
display(
    df_sales[
        [
            'Posting Date',
            'Year',
            'MonthNum',
            'Quarter',
            'DayOfWeek',
            'WeekOfYear',
            'IsWeekend'
        ]
    ].head()
)


# %% [markdown]
# # 10. Customer Features (RFM + purchase breadth)

# %%
snapshot_date = df_sales['Posting Date'].max() + pd.Timedelta(days=1)

cust_feat = df_sales.groupby('Contact Name').agg(
    Recency=('Posting Date', lambda x: (snapshot_date - x.max()).days),
    Frequency=('Bill Doc', 'nunique'),
    Monetary=('Total VALUE', 'sum'),
    Unique_Products=('Material Desc', 'nunique'),
    Unique_PartGroups=('Part Group', 'nunique'),
    Avg_Order_Value=('Total VALUE', 'mean'),
    Tenure_Days=('Posting Date', lambda x: (x.max() - x.min()).days)
).reset_index()

print(cust_feat.shape)
cust_feat.sort_values('Monetary', ascending=False).head(10)

# %% [markdown]
# # Product Features

# %%
# Convert numerical columns to numeric
df_sales['Quantity'] = pd.to_numeric(
    df_sales['Quantity'],
    errors='coerce'
)

df_sales['Total VALUE'] = pd.to_numeric(
    df_sales['Total VALUE'],
    errors='coerce'
)

# Check the data types
print(df_sales[['Quantity', 'Total VALUE']].dtypes)


# %%
prod_feat = df_sales.groupby(
    ['Material Desc', 'Part Group']
).agg(
    Total_Qty=('Quantity', 'sum'),
    Total_Revenue=('Total VALUE', 'sum'),
    Num_Orders=('Bill Doc', 'nunique')
).reset_index()

display(prod_feat.head())


# %% [markdown]
# This code block aggregates your raw transactional dataset (df_sales) at the individual product level to build structured features for your product analysis
# It extracts three essential business metrics (Volume, Revenue, and Transaction Count) needed to calculate metrics like average price per unit or order velocity.

# %% [markdown]
# # SALES Features (trend/growth)

# %%
monthly = df_sales.groupby('MonthNum').agg(Qty=('Quantity', 'sum'), Revenue=('Total VALUE', 'sum')).reset_index()
monthly['Revenue_MoM_Growth_%'] = monthly['Revenue'].pct_change() * 100

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(monthly['MonthNum'], monthly['Revenue'], marker='o', color='#4C72B0')
ax.set_title('Monthly Revenue Trend')
ax.set_xlabel('Month')
ax.set_ylabel('Revenue')
plt.tight_layout()
plt.show()

monthly

# %%
import matplotlib.pyplot as plt

# Calculate revenue by Part Group
part_group_revenue = df_sales.groupby('Part Group')['Total VALUE'].sum()

# Create pie chart
fig, ax = plt.subplots(figsize=(8, 6))

ax.pie(
    part_group_revenue,
    labels=part_group_revenue.index,
    autopct='%1.1f%%',
    startangle=90
)

ax.set_title('Revenue Distribution by Part Group')

plt.tight_layout()
plt.show()


# %% [markdown]
# ## Customer–Product Matrix
# 

# %%
cp_matrix = df_sales.pivot_table(index='Contact Name', columns='Material Desc', values='Quantity',
                                  aggfunc='sum', fill_value=0)
print("Customer-Product matrix shape:", cp_matrix.shape)
cp_matrix.iloc[:5, :6]

# %% [markdown]
# ## Market Basket Analysis
# `mlxtend` isn't available in this environment, so association rules (support / confidence / lift) are computed
# directly from invoice co-occurrence — restricted to the top 60 products by volume to keep the pair-count
# combinatorics tractable. This is the standard Apriori metric set, just computed by hand.

# %% [markdown]
# ### Confidence
# 
# Shows the probability of buying one product when another is purchased.
# 
# For example:
# 
# Confidence A → B = 70%
# 
# Meaning:
# 
# Among customers/invoices containing Product A, 70% also containe

# %%
from collections import Counter
from itertools import combinations

TOP_N_FOR_BASKET = 60
top_products_for_basket = prod_feat.sort_values('Total_Qty', ascending=False).head(TOP_N_FOR_BASKET)['Material Desc'].tolist()

baskets = (df_sales[df_sales['Material Desc'].isin(top_products_for_basket)]
           .groupby('Bill Doc')['Material Desc'].apply(set))

n_baskets = len(baskets)
item_counts = Counter()
pair_counts = Counter()
for basket in baskets:
    for item in basket:
        item_counts[item] += 1
    for a, b in combinations(sorted(basket), 2):
        pair_counts[(a, b)] += 1

MIN_COOCCURRENCE = 5
rules = []
for (a, b), cnt in pair_counts.items():
    if cnt < MIN_COOCCURRENCE:
        continue
    support = cnt / n_baskets
    conf_a_to_b = cnt / item_counts[a]
    conf_b_to_a = cnt / item_counts[b]
    lift = support / ((item_counts[a] / n_baskets) * (item_counts[b] / n_baskets))
    rules.append((a, b, cnt, support, conf_a_to_b, conf_b_to_a, lift))

rules_df = pd.DataFrame(rules, columns=['Product_A', 'Product_B', 'CoCount', 'Support',
                                         'Confidence_A_to_B', 'Confidence_B_to_A', 'Lift'])
rules_df = rules_df.sort_values('Lift', ascending=False)

print(f"{n_baskets} invoices considered, {len(rules_df)} rules with co-occurrence >= {MIN_COOCCURRENCE}")
rules_df.head(15)

# %%
# ============================================================
# SAVE MARKET BASKET ANALYSIS RESULTS AS CSV
# ============================================================

file_name = "Market Basket Analysis.csv"

rules_df.to_csv(
    file_name,
    index=False,
    encoding="utf-8-sig"
)

print(f"CSV file successfully created: {file_name}")

# Display the top results
display(rules_df.head(15))

# %% [markdown]
# #  Product Similarity
# Two complementary similarity signals, both via **cosine similarity**:
# 
# 1. **Content-based** — `TfidfVectorizer` on `Material Desc + Part Group` text, cosine similarity between TF-IDF vectors. Captures "these products *sound* alike" (same family, size variants).
# 2. **Collaborative** — cosine similarity between product columns of the customer-product matrix. Captures "these products are *bought by the same customers*", independent of naming.
# 
# Both feed the hybrid recommender in section 20.

# %%
pip install nltk

# %%
# --- Content-based (TF-IDF + cosine similarity) ---
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

prod_feat['text'] = prod_feat['Material Desc'].astype(str) + ' ' + prod_feat['Part Group'].astype(str)

tfidf = TfidfVectorizer(stop_words='english', max_features=2000)
tfidf_matrix = tfidf.fit_transform(prod_feat['text'])

content_sim = cosine_similarity(tfidf_matrix)
content_sim_df = pd.DataFrame(content_sim, index=prod_feat['Material Desc'], columns=prod_feat['Material Desc'])

print("Content similarity matrix:", content_sim_df.shape)

def similar_products(name, sim_df, top_n=5):
    if name not in sim_df.index:
        return f"'{name}' not found"
    sims = sim_df[name].drop(index=name).sort_values(ascending=False).head(top_n)
    return sims.rename('Similarity').to_frame()

print("Content-similar to 'SHOE KIT':")
similar_products('SHOE KIT', content_sim_df)

# %% [markdown]
# Similarity Search For Month

# %%
print(df.columns.tolist())


# %%
month_order = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
]

df = df[df["Month"].isin(month_order)].copy()


# %%
df = df[df["Month"].isin(month_order)].copy()


# %%
df['Month'][0]

# %%
df['Month'][5]

# %% [markdown]
# Determining Cosine Similarity

# %%
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Month × Part Group matrix
month_part = pd.pivot_table(
    df,
    index="Month",
    columns="Part Group",
    values="Quantity",
    aggfunc="sum",
    fill_value=0
)

# Calculate cosine similarity between months
month_similarity = cosine_similarity(month_part)

# Convert to DataFrame
month_similarity_df = pd.DataFrame(
    month_similarity,
    index=month_part.index,
    columns=month_part.index
)

month_similarity_df


# %%
# ============================================
# PART GROUP vs MONTH SIMILARITY
# January to July
# ============================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------
# 1. Make a fresh copy of your ORIGINAL df
# --------------------------------------------
data = df.copy()

# Clean column names
data.columns = (
    data.columns
    .astype(str)
    .str.strip()
)

print("YOUR ACTUAL COLUMNS:")
print(data.columns.tolist())


# --------------------------------------------
# 2. Find Part Group and Month columns
# --------------------------------------------

part_col = None
month_col = None

for col in data.columns:
    if str(col).strip().lower() == "part group":
        part_col = col

    if str(col).strip().lower() == "month":
        month_col = col

if part_col is None:
    raise ValueError(
        "I cannot find 'Part Group'. Your columns are:\n"
        + str(data.columns.tolist())
    )

if month_col is None:
    raise ValueError(
        "I cannot find 'Month'. Your columns are:\n"
        + str(data.columns.tolist())
    )


# --------------------------------------------
# 3. Find numeric columns automatically
# --------------------------------------------

numeric_cols = data.select_dtypes(
    include="number"
).columns.tolist()

print("\nNUMERIC COLUMNS:")
print(numeric_cols)

if len(numeric_cols) == 0:
    raise ValueError(
        "There are no numeric columns available for similarity calculation."
    )


# --------------------------------------------
# 4. Display numeric columns so you can choose
# --------------------------------------------

print("\nChoose the metric you want to compare.")

for i, col in enumerate(numeric_cols):
    print(f"{i}: {col}")


# --------------------------------------------
# 5. AUTOMATICALLY SELECT A NUMERIC COLUMN
# --------------------------------------------
# We avoid 'Total Qty' completely.
#
# The first numeric column is selected.
#
# You can later replace this with:
# value_col = "your_actual_column_name"

value_col = numeric_cols[0]

print("\nUSING THIS COLUMN FOR SIMILARITY:")
print(value_col)


# --------------------------------------------
# 6. Clean Month values
# --------------------------------------------

data[month_col] = (
    data[month_col]
    .astype(str)
    .str.strip()
    .str[:3]
    .str.title()
)

months = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul"
]

data = data[
    data[month_col].isin(months)
].copy()


# --------------------------------------------
# 7. Create Part Group × Month matrix
# --------------------------------------------

monthly_matrix = pd.pivot_table(
    data,
    index=part_col,
    columns=month_col,
    values=value_col,
    aggfunc="sum",
    fill_value=0
)

# Force correct Jan → Jul order
monthly_matrix = monthly_matrix.reindex(
    columns=months,
    fill_value=0
)

print("\n================================")
print("PART GROUP × MONTH MATRIX")
print("================================")

display(monthly_matrix)


# --------------------------------------------
# 8. Similarity BETWEEN PART GROUPS
# --------------------------------------------

part_similarity = cosine_similarity(
    monthly_matrix
)

part_similarity_df = pd.DataFrame(
    part_similarity,
    index=monthly_matrix.index,
    columns=monthly_matrix.index
)

print("\n================================")
print("PART GROUP SIMILARITY")
print("================================")

display(
    part_similarity_df.round(3)
)


# --------------------------------------------
# 9. Similarity BETWEEN MONTHS
# --------------------------------------------

month_similarity = cosine_similarity(
    monthly_matrix.T
)

month_similarity_df = pd.DataFrame(
    month_similarity,
    index=monthly_matrix.columns,
    columns=monthly_matrix.columns
)

print("\n================================")
print("MONTH SIMILARITY")
print("================================")

display(
    month_similarity_df.round(3)
)


# --------------------------------------------
# 10. PART GROUP SIMILARITY HEATMAP
# --------------------------------------------

plt.figure(figsize=(11, 9))

sns.heatmap(
    part_similarity_df,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    vmin=0,
    vmax=1,
    square=True
)

plt.title(
    f"Part Group Similarity ({value_col})\n"
    "January–July"
)

plt.xlabel("Part Group")
plt.ylabel("Part Group")

plt.tight_layout()
plt.show()


# --------------------------------------------
# 11. MONTH SIMILARITY HEATMAP
# --------------------------------------------

plt.figure(figsize=(9, 7))

sns.heatmap(
    month_similarity_df,
    annot=True,
    fmt=".2f",
    cmap="YlGnBu",
    vmin=0,
    vmax=1,
    square=True
)

plt.title(
    f"Month-to-Month Similarity ({value_col})"
)

plt.xlabel("Month")
plt.ylabel("Month")

plt.tight_layout()
plt.show()


# %%
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Customer × Product interaction matrix
cp_matrix = pd.crosstab(
    df["Contact Name"],
    df["Material"]
)

print("Customer-Product matrix:", cp_matrix.shape)

# Product × Product collaborative similarity
collab_sim = cosine_similarity(cp_matrix.T)

collab_sim_df = pd.DataFrame(
    collab_sim,
    index=cp_matrix.columns,
    columns=cp_matrix.columns
)

print("Collaborative similarity matrix:", collab_sim_df.shape)

# %%
cp_matrix = pd.crosstab(
    df["Contact Name"],
    df["Part Group"]
)

collab_sim = cosine_similarity(cp_matrix.T)

collab_sim_df = pd.DataFrame(
    collab_sim,
    index=cp_matrix.columns,
    columns=cp_matrix.columns
)

# %%
print("cp_matrix:", cp_matrix.shape)
print("collab_sim_df:", collab_sim_df.shape)

print(cp_matrix.head())

# %%
# --- Collaborative (co-purchase pattern via customer-product matrix) ---
collab_sim = cosine_similarity(cp_matrix.T)
collab_sim_df = pd.DataFrame(collab_sim, index=cp_matrix.columns, columns=cp_matrix.columns)

print("Collaborative similarity matrix:", collab_sim_df.shape)
print("\nCollaboratively-similar to 'SHOE KIT' (bought alongside / instead-of by the same customers):")
similar_products('SHOE KIT', collab_sim_df)


# %%
# --- Collaborative (co-purchase pattern via customer-product matrix) ---
collab_sim = cosine_similarity(cp_matrix.T)
collab_sim_df = pd.DataFrame(collab_sim, index=cp_matrix.columns, columns=cp_matrix.columns)

print("Collaborative similarity matrix:", collab_sim_df.shape)
print("\nCollaboratively-similar to 'SHOE KIT' (bought alongside / instead-of by the same customers):")
similar_products('GREEN COOLANT BOTTLE (1 LTR)', collab_sim_df)

# %% [markdown]
# # 16. Customer Segmentation
# K-Means on log-scaled RFM features. Silhouette score is checked across k=2..6 to guide the choice —
# k=2 scores highest (a simple active/inactive split) but k=4 gives more commercially actionable segments
# (e.g. distinguishing "high value, infrequent" from "high value, frequent"), so k=4 is used going forward.
# Both are shown so the tradeoff is explicit rather than picked silently.

# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------
# 1. Clean date column
# --------------------------------------------------
data['Posting Date'] = pd.to_datetime(
    data['Posting Date'],
    errors='coerce'
)

# --------------------------------------------------
# 2. Clean monetary column
# --------------------------------------------------
# Convert Total VALUE from text/Arrow string to numeric
data['Total VALUE'] = (
    data['Total VALUE']
    .astype(str)
    .str.replace(',', '', regex=False)
    .str.replace('₹', '', regex=False)
    .str.strip()
)

data['Total VALUE'] = pd.to_numeric(
    data['Total VALUE'],
    errors='coerce'
)

# --------------------------------------------------
# 3. Clean customer and transaction identifiers
# --------------------------------------------------
data['Customer No'] = data['Customer No'].astype('string').str.strip()
data['Bill Doc'] = data['Bill Doc'].astype('string').str.strip()

# --------------------------------------------------
# 4. Remove unusable rows
# --------------------------------------------------
rfm_data = data.dropna(
    subset=['Customer No', 'Posting Date', 'Total VALUE']
).copy()

# Remove blank customer numbers
rfm_data = rfm_data[
    rfm_data['Customer No'].notna() &
    (rfm_data['Customer No'].str.len() > 0)
].copy()

# --------------------------------------------------
# 5. Reference date
# --------------------------------------------------
reference_date = (
    rfm_data['Posting Date'].max()
    + pd.Timedelta(days=1)
)

# --------------------------------------------------
# 6. Create customer-level RFM features
# --------------------------------------------------
cust_feat = (
    rfm_data.groupby('Customer No')
    .agg(
        Recency=(
            'Posting Date',
            lambda x: (reference_date - x.max()).days
        ),
        Frequency=(
            'Bill Doc',
            'nunique'
        ),
        Monetary=(
            'Total VALUE',
            'sum'
        )
    )
    .reset_index()
)

# --------------------------------------------------
# 7. Make absolutely sure Monetary is numeric
# --------------------------------------------------
cust_feat['Monetary'] = pd.to_numeric(
    cust_feat['Monetary'],
    errors='coerce'
)

cust_feat['Frequency'] = pd.to_numeric(
    cust_feat['Frequency'],
    errors='coerce'
)

cust_feat['Recency'] = pd.to_numeric(
    cust_feat['Recency'],
    errors='coerce'
)

# --------------------------------------------------
# 8. Remove invalid RFM records
# --------------------------------------------------
cust_feat = cust_feat.dropna(
    subset=['Recency', 'Frequency', 'Monetary']
).copy()

cust_feat = cust_feat[
    cust_feat['Monetary'] > 0
].copy()

# --------------------------------------------------
# 9. Create RFM matrix
# --------------------------------------------------
rfm = cust_feat[
    ['Recency', 'Frequency', 'Monetary']
].copy()

# --------------------------------------------------
# 10. Log transform
# --------------------------------------------------
rfm_log = rfm.apply(
    lambda x: np.log1p(x)
)

# --------------------------------------------------
# 11. Standardize
# --------------------------------------------------
scaler = StandardScaler()

rfm_scaled = scaler.fit_transform(rfm_log)

print("RFM created successfully")
print("Customers:", len(cust_feat))
print("\nRFM summary:")
print(rfm.describe())

print("\nFirst 5 customers:")
print(cust_feat.head())


# %% [markdown]
# # 17. Demand Prediction
# Aggregate to **product × month**, predict next quantity from prior-month lag features. Time-based split
# (train on Feb–Apr, test on May–Jun; Jan has no lag so it's dropped) — this avoids leaking future months into
# training, which a random split would do.
# 
# Four models compared: Linear Regression, Ridge, Random Forest, Gradient Boosting.
# (`xgboost` isn't available in this environment; Gradient Boosting is scikit-learn's closest equivalent and is
# included specifically as the strongest candidate.)

# %%
#Product-level features
prod_feat = (
    df_sales
    .groupby(['Material Desc', 'Part Group'])
    .agg(
        Total_Qty=('Quantity', 'sum'),
        Total_Revenue=('Total VALUE', 'sum'),
        Num_Orders=('Bill Doc', 'nunique')
    )
    .reset_index()
)

print("prod_feat created successfully!")
display(prod_feat.head())
print(prod_feat.columns.tolist())


# %%
pm = (
    df_sales
    .groupby(['Material Desc', 'MonthNum'])
    .agg(
        Qty=('Quantity', 'sum'),
        Revenue=('Total VALUE', 'sum')
    )
    .reset_index()
    .sort_values(['Material Desc', 'MonthNum'])
)

# Previous month's quantity and revenue
pm['Qty_Lag1'] = (
    pm.groupby('Material Desc')['Qty']
    .shift(1)
)

pm['Rev_Lag1'] = (
    pm.groupby('Material Desc')['Revenue']
    .shift(1)
)

# Merge product-level features
pm = pm.merge(
    prod_feat[
        ['Material Desc', 'Total_Qty', 'Total_Revenue', 'Num_Orders']
    ],
    on='Material Desc',
    how='left'
)

# Remove rows where previous-month quantity is unavailable
pm_model = pm.dropna(
    subset=['Qty_Lag1']
).copy()
print("pm shape:", pm.shape)
print("pm_model shape:", pm_model.shape)
display(pm_model.head())


# %% [markdown]
# ## Section 4: Feature Scaling with StandardScaler
# 
# ⚠️ **THIS IS CRITICAL FOR YOUR MODELS TO WORK WELL**
# 
# Without scaling:
# - Features at different scales (MonthNum ~1-6, Qty_Lag1 ~millions)
# - Gradient descent converges poorly or fails
# - Ridge/Lasso/SGD all underperform
# 
# With StandardScaler:
# - All features mean=0, std=1
# - Gradient descent converges quickly
# - Regularization (alpha parameter) works fairly across all features
# 

# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# --------------------------------------------------
# 1. Create modeling dataframe
# --------------------------------------------------
pm_model = data.copy()

# --------------------------------------------------
# 2. Clean dates
# --------------------------------------------------
pm_model['Posting Date'] = pd.to_datetime(
    pm_model['Posting Date'],
    errors='coerce'
)

# --------------------------------------------------
# 3. Clean numeric columns
# --------------------------------------------------
numeric_cols = [
    'Quantity',
    'Net Amount',
    'Total VALUE',
    'Rate'
]

for col in numeric_cols:
    if col in pm_model.columns:
        pm_model[col] = (
            pm_model[col]
            .astype(str)
            .str.replace(',', '', regex=False)
            .str.replace('₹', '', regex=False)
            .str.strip()
        )
        pm_model[col] = pd.to_numeric(
            pm_model[col],
            errors='coerce'
        )

# --------------------------------------------------
# 4. Create Month
# --------------------------------------------------
pm_model['Month'] = pm_model['Posting Date'].dt.to_period('M')

# Remove rows without required information
pm_model = pm_model.dropna(
    subset=['Month', 'Part Group', 'Quantity']
).copy()

# --------------------------------------------------
# 5. Convert Part Group to string
# --------------------------------------------------
pm_model['Part Group'] = (
    pm_model['Part Group']
    .astype('string')
    .str.strip()
)

# --------------------------------------------------
# 6. Encode Part Group
# --------------------------------------------------
le_group = LabelEncoder()

pm_model['PartGroup_enc'] = le_group.fit_transform(
    pm_model['Part Group']
)

print("pm_model created successfully")
print("Rows:", len(pm_model))
print("Part Groups:", pm_model['Part Group'].nunique())
print(pm_model[['Month', 'Part Group', 'PartGroup_enc', 'Quantity']].head())


# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# ============================================================
# 1. START FROM RAW DATA
# ============================================================

pm_model = data.copy()

# Date
pm_model['Posting Date'] = pd.to_datetime(
    pm_model['Posting Date'],
    errors='coerce'
)

# Numeric columns
numeric_cols = [
    'Quantity',
    'Net Amount',
    'Total VALUE',
    'Rate'
]

for col in numeric_cols:
    if col in pm_model.columns:
        pm_model[col] = (
            pm_model[col]
            .astype(str)
            .str.replace(',', '', regex=False)
            .str.replace('₹', '', regex=False)
            .str.strip()
        )
        pm_model[col] = pd.to_numeric(
            pm_model[col],
            errors='coerce'
        )

# Clean Part Group
pm_model['Part Group'] = (
    pm_model['Part Group']
    .astype('string')
    .str.strip()
)

# ============================================================
# 2. CREATE MONTH
# ============================================================

pm_model['Month'] = (
    pm_model['Posting Date']
    .dt.to_period('M')
)

# Remove invalid rows
pm_model = pm_model.dropna(
    subset=['Month', 'Part Group', 'Quantity']
).copy()

# ============================================================
# 3. MONTHLY AGGREGATION BY PART GROUP
# ============================================================

monthly = (
    pm_model
    .groupby(['Part Group', 'Month'])
    .agg(
        Qty=('Quantity', 'sum'),
        Revenue=('Total VALUE', 'sum'),
        Orders=('Bill Doc', 'nunique'),
        Num_Customers=('Customer No', 'nunique'),
        Avg_Rate=('Rate', 'mean')
    )
    .reset_index()
)

# ============================================================
# 4. SORT BEFORE CREATING LAGS
# ============================================================

monthly = monthly.sort_values(
    ['Part Group', 'Month']
).reset_index(drop=True)

# ============================================================
# 5. CREATE TIME FEATURES
# ============================================================

monthly['MonthNum'] = (
    monthly['Month'].dt.month
)

# ============================================================
# 6. CREATE LAG FEATURES
# ============================================================

monthly['Qty_Lag1'] = (
    monthly.groupby('Part Group')['Qty']
    .shift(1)
)

monthly['Qty_Lag2'] = (
    monthly.groupby('Part Group')['Qty']
    .shift(2)
)

monthly['Revenue_Lag1'] = (
    monthly.groupby('Part Group')['Revenue']
    .shift(1)
)

monthly['Orders_Lag1'] = (
    monthly.groupby('Part Group')['Orders']
    .shift(1)
)

# ============================================================
# 7. MOVING AVERAGE
# ============================================================

monthly['Qty_MA2'] = (
    monthly.groupby('Part Group')['Qty']
    .transform(
        lambda x: x.shift(1).rolling(
            window=2,
            min_periods=2
        ).mean()
    )
)

# ============================================================
# 8. TOTAL_QTY
# ============================================================

# Total quantity for the Part Group up to the previous month
monthly['Total_Qty'] = (
    monthly.groupby('Part Group')['Qty']
    .transform(
        lambda x: x.shift(1).cumsum()
    )
)

# ============================================================
# 9. ENCODE PART GROUP
# ============================================================

le_group = LabelEncoder()

monthly['PartGroup_enc'] = (
    le_group.fit_transform(
        monthly['Part Group']
    )
)

# ============================================================
# 10. FINAL MODEL DATAFRAME
# ============================================================

pm_model = monthly.copy()

FEATURES = [
    'MonthNum',
    'Qty_Lag1',
    'Qty_Lag2',
    'Revenue_Lag1',
    'Orders_Lag1',
    'Avg_Rate',
    'Orders',
    'Num_Customers',
    'Total_Qty',
    'PartGroup_enc',
    'Qty_MA2'
]

TARGET = 'Qty'

# ============================================================
# 11. CHECK REQUIRED COLUMNS
# ============================================================

missing = [
    col for col in FEATURES + [TARGET]
    if col not in pm_model.columns
]

if missing:
    raise ValueError(
        f"Missing columns: {missing}"
    )

# ============================================================
# 12. REMOVE ROWS WHERE LAGS ARE NOT AVAILABLE
# ============================================================

pm_model = pm_model.dropna(
    subset=FEATURES + [TARGET]
).copy()

# ============================================================
# 13. CREATE X AND y
# ============================================================

X = pm_model[FEATURES].copy()
y = pm_model[TARGET].copy()

print("Model preparation successful!")
print("pm_model shape:", pm_model.shape)
print("X shape:", X.shape)
print("y shape:", y.shape)

print("\nFeatures:")
print(X.columns.tolist())

print("\nSample:")
print(pm_model[
    ['Part Group', 'Month', 'Qty'] + FEATURES
].head())


# %%
# 🔑 APPLY STANDARDSCALER
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=FEATURES, index=X.index)

print("\n✅ After Scaling (mean≈0, std≈1):")
print(X_scaled.describe())

# %% [markdown]
# #  Product Priority Score
# A weighted, business-facing score for inventory/promotion prioritization — combines how much revenue a product
# drives, how many distinct customers buy it (reach), and how often it's ordered (frequency). Weights are a
# starting point (revenue-led); adjust to match actual commercial priorities.

# %%
# ============================================================
# 1. Recreate product-level features
# ============================================================

prod_feat = df_sales.groupby(
    ['Material Desc', 'Part Group']
).agg(
    Total_Qty=('Quantity', 'sum'),
    Total_Revenue=('Total VALUE', 'sum'),
    Num_Orders=('Bill Doc', 'nunique')
).reset_index()


# ============================================================
# 2. Calculate number of unique customers per product
# ============================================================

customer_count = df_sales.groupby(
    'Material Desc'
)['Customer'].nunique().reset_index()

customer_count = customer_count.rename(
    columns={'Customer': 'Num_Customers'}
)


# ============================================================
# 3. Merge customer count into prod_feat
# ============================================================

prod_feat = prod_feat.merge(
    customer_count,
    on='Material Desc',
    how='left'
)


# ============================================================
# 4. Check the resulting columns
# ============================================================

print("prod_feat columns:")
print(prod_feat.columns.tolist())

display(prod_feat.head(10))


# %%
# ============================================================
# CREATE PRODUCT PRIORITY DATA
# ============================================================

# Check required columns
print(df.columns.tolist())

# Clean numeric columns
df["Quantity"] = pd.to_numeric(
    df["Quantity"],
    errors="coerce"
).fillna(0)

df["Total VALUE"] = (
    df["Total VALUE"]
    .astype(str)
    .str.replace("₹", "", regex=False)
    .str.replace(",", "", regex=False)
    .str.strip()
)

df["Total VALUE"] = pd.to_numeric(
    df["Total VALUE"],
    errors="coerce"
).fillna(0)


# ============================================================
# CREATE PRIORITY DATAFRAME
# ============================================================

priority = (
    df
    .groupby(
        ["Material", "Material Desc"],
        as_index=False
    )
    .agg(
        Total_Revenue=("Total VALUE", "sum"),
        Num_Customers=("Contact Name", "nunique"),
        Num_Orders=("Material", "size"),
        Total_Quantity=("Quantity", "sum")
    )
)

print("✅ priority created successfully")
print("Shape:", priority.shape)

display(priority.head(10))

# %%
print(priority["Total_Revenue"].head(10))

# %%
priority["Total_Revenue"] = (
    priority["Total_Revenue"]
    .astype(str)
    .str.replace(",", "", regex=False)
    .str.replace("₹", "", regex=False)
    .str.strip()
)

priority["Total_Revenue"] = pd.to_numeric(
    priority["Total_Revenue"],
    errors="coerce"
)

# %%
# ============================================================
# FIND AVAILABLE NUMERIC COLUMNS
# ============================================================

possible_numeric_columns = [
    "Total_Revenue",
    "Total_Quantity",
    "Frequency",
    "Num_Customers",
    "Num_Orders"
]

numeric_columns = [
    col
    for col in possible_numeric_columns
    if col in priority.columns
]

print("Available numeric columns:")
print(numeric_columns)

# %%
# ============================================================
# CONVERT AVAILABLE COLUMNS TO NUMERIC
# ============================================================

for col in numeric_columns:

    priority[col] = (
        priority[col]
        .astype(str)
        .str.replace("₹", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )

    priority[col] = pd.to_numeric(
        priority[col],
        errors="coerce"
    ).fillna(0)

print("\nData types after conversion:")
print(priority[numeric_columns].dtypes)

# %%
# ============================================================
# NORMALIZATION FUNCTION
# ============================================================

def normalize(s):

    s = pd.to_numeric(
        s,
        errors="coerce"
    ).fillna(0)

    min_value = s.min()
    max_value = s.max()

    if max_value == min_value:
        return pd.Series(
            1.0,
            index=s.index
        )

    return (
        (s - min_value)
        /
        (max_value - min_value)
    )
    # ============================================================
# INDIVIDUAL SCORES
# ============================================================

priority["Revenue_Score"] = normalize(
    priority["Total_Revenue"]
)

priority["Reach_Score"] = normalize(
    priority["Num_Customers"]
)

priority["Frequency_Score"] = normalize(
    priority["Num_Orders"]
)

# %%
# ============================================================
# FINAL PRIORITY SCORE
# ============================================================

priority["Priority_Score"] = (
    0.50 * priority["Revenue_Score"]
    + 0.30 * priority["Reach_Score"]
    + 0.20 * priority["Frequency_Score"]
)

priority = (
    priority
    .sort_values(
        "Priority_Score",
        ascending=False
    )
    .reset_index(drop=True)
)

display(priority.head(20))

# %%
variables_to_check = [
    "df",
    "cross_output",
    "final_recommendations",
    "priority"
]

for var in variables_to_check:
    print(f"{var}: {var in globals()}")

# %%
file_name = "product_priority_scores.csv"

priority_output.to_csv(
    file_name,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 14. DISPLAY RESULT
# ============================================================

print(f"CSV file successfully created: {file_name}")

display(priority_output.head(10))

# %% [markdown]
# #  Recommender System — Item-Based Collaborative Filtering
# For a given customer, score every product they haven't bought by its average cosine similarity (from the
# collaborative similarity matrix in section 15) to the products they *have* bought, then rank.

# %%
print("cust_feat customer values:")
print(cust_feat.iloc[:5, 0])

print("\ncp_matrix index:")
print(cp_matrix.index[:5])

# %% [markdown]
# Below code creates a customer/contact-level summary by grouping all transactions by Contact Name.
# Frequency counts how many transactions/records each contact has.
# TotalQty calculates the total quantity purchased by each contact.
# TotalValue calculates the total sales value generated by each contact.
# The resulting cust_feat table gives a quick view of each contact's purchasing activity and business value

# %%
cust_feat = (
    df.groupby('Contact Name')
      .agg(
          Frequency=('Contact Name', 'size'),
          TotalQty=('Quantity', 'sum'),
          TotalValue=('Total VALUE', 'sum')
      )
)

print(cust_feat.head())

# %%
sample_customer = (
    cust_feat
    .sort_values('Frequency', ascending=False)
    .index[2]
)

print("Sample customer:", sample_customer)

# %%
sample_customer = (
    cust_feat
    .sort_values('Frequency', ascending=False)
    .index[11]
)

print("Sample customer:", sample_customer)

# %%
sample_customer = (
    cust_feat
    .sort_values('Frequency', ascending=False)
    .index[36]
)

print("Sample customer:", sample_customer)

# %%
# ============================================================
# CREATE CUSTOMER-PRODUCT MATRIX
# ============================================================

cp_matrix = pd.pivot_table(
    df,
    index="Contact Name",
    columns="Material",
    values="Quantity",
    aggfunc="sum",
    fill_value=0
)

print("✅ Customer-Product Matrix created")
print("Shape:", cp_matrix.shape)

display(cp_matrix.head())

# %%
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# ============================================================
# PRODUCT-PRODUCT COLLABORATIVE SIMILARITY
# ============================================================

collab_sim = cosine_similarity(cp_matrix.T)

collab_sim_df = pd.DataFrame(
    collab_sim,
    index=cp_matrix.columns,
    columns=cp_matrix.columns
)

print("✅ Collaborative Similarity Matrix created")
print("Shape:", collab_sim_df.shape)

# %%
# ============================================================
# COLLABORATIVE RECOMMENDATION FUNCTION
# ============================================================

def recommend_collaborative(customer, top_n=5):

    # Check whether customer exists
    if customer not in cp_matrix.index:
        print(f"Customer '{customer}' not found.")
        return pd.Series(dtype=float)

    # Products already purchased
    bought = cp_matrix.loc[customer]

    bought_products = (
        bought[bought > 0]
        .index
        .tolist()
    )

    # Check if customer purchased anything
    if len(bought_products) == 0:
        print("No purchase history available.")
        return pd.Series(dtype=float)

    # Keep only products available in similarity matrix
    bought_products = [
        product
        for product in bought_products
        if product in collab_sim_df.index
    ]

    # Calculate average similarity score
    scores = (
        collab_sim_df
        .loc[bought_products]
        .mean(axis=0)
    )

    # Remove already purchased products
    scores = scores.drop(
        index=bought_products,
        errors="ignore"
    )

    # Return Top-N recommendations
    return (
        scores
        .sort_values(ascending=False)
        .head(top_n)
    )


print("✅ recommend_collaborative() function created successfully")

# %%
print("Sample customer:", sample_customer)
print("Customer exists in cp_matrix:",
      sample_customer in cp_matrix.index)

# %%
cp_matrix = pd.crosstab(
    df['Contact Name'],
    df['Material']
)

# %%
print("Sample customer:", sample_customer)
print("Type:", type(sample_customer))

# %%
print(cp_matrix.index[:10])

# %%
sample_customer = cp_matrix.index[0]

print("Sample customer:", sample_customer)

# %%
recommendations = recommend_collaborative(
    sample_customer,
    top_n=10
)

print(recommendations)

# %%
def recommend_collaborative(customer, top_n=5):

    # ------------------------------------------------
    # 1. Check if customer exists
    # ------------------------------------------------

    if customer not in cp_matrix.index:
        print(f"❌ Customer '{customer}' does not exist in cp_matrix.")
        print("\nExample valid customers:")
        print(cp_matrix.index[:5].tolist())

        return pd.Series(dtype=float)

    # ------------------------------------------------
    # 2. Get customer's purchased products
    # ------------------------------------------------

    bought = cp_matrix.loc[customer]

    bought_products = (
        bought[bought > 0]
        .index
        .tolist()
    )

    # ------------------------------------------------
    # 3. Check purchase history
    # ------------------------------------------------

    if len(bought_products) == 0:

        print("❌ This customer has no purchase history.")

        return pd.Series(dtype=float)

    # ------------------------------------------------
    # 4. Calculate similarity scores
    # ------------------------------------------------

    valid_products = [
        product
        for product in bought_products
        if product in collab_sim_df.index
    ]

    if len(valid_products) == 0:

        print("❌ No valid products found in similarity matrix.")

        return pd.Series(dtype=float)

    scores = (
        collab_sim_df
        .loc[valid_products]
        .mean(axis=0)
    )

    # ------------------------------------------------
    # 5. Remove already purchased products
    # ------------------------------------------------

    scores = scores.drop(
        index=bought_products,
        errors="ignore"
    )

    # ------------------------------------------------
    # 6. Return Top-N recommendations
    # ------------------------------------------------

    return (
        scores
        .sort_values(ascending=False)
        .head(top_n)
    )

# %%
print("sample_customer =", sample_customer)
print("sample_customer type =", type(sample_customer))
print("\nFirst customers in cp_matrix:")
print(cp_matrix.index[:10].tolist())

# %%
def get_similar_products(
    row_index,
    top_n=10,
    similarity_threshold=0.0
):

    # Get query vector
    query_vector = combined_matrix[row_index]

    # Calculate cosine similarity
    similarity_scores = cosine_similarity(
        query_vector,
        combined_matrix
    ).flatten()

    # Create result DataFrame
    result = feature_df.copy()

    result["Similarity_Score"] = similarity_scores

    # Remove query item itself
    result = result[
        result.index != row_index
    ]

    # Apply threshold
    result = result[
        result["Similarity_Score"] >= similarity_threshold
    ]

    # Sort by similarity
    result = (
        result
        .sort_values(
            "Similarity_Score",
            ascending=False
        )
        .head(top_n)
    )

    return result

# %% [markdown]
# #  Hybrid Recommender
# Blends content-based similarity (product naming/description) with collaborative similarity (co-purchase
# behaviour) into one similarity matrix, weighted 50/50 by default. Content-based covers new/low-history
# products (cold start); collaborative captures real buying patterns TF-IDF can't see. `HYBRID_WEIGHT` controls
# the mix — raise it toward 1 to lean more on product description similarity, lower it to lean on purchase
# behaviour.

# %%
print("sample_customer:", sample_customer)
print("Type:", type(sample_customer))

print("\ncp_matrix index name:")
print(cp_matrix.index.name)

print("\nFirst 10 valid customers:")
print(cp_matrix.index[:10].tolist())

print("\nDoes sample_customer exist?")
print(sample_customer in cp_matrix.index)

# %%
HYBRID_WEIGHT = 0.5  # weight on content-based similarity; (1 - HYBRID_WEIGHT) goes to collaborative

# Align both matrices on the same product set/order before blending
common_products = [p for p in prod_feat['Material Desc'] if p in cp_matrix.columns]
content_aligned = content_sim_df.loc[common_products, common_products]
collab_aligned = collab_sim_df.loc[common_products, common_products]

hybrid_sim_df = HYBRID_WEIGHT * content_aligned + (1 - HYBRID_WEIGHT) * collab_aligned
print("Hybrid similarity matrix:", hybrid_sim_df.shape)

def recommend_hybrid(customer, top_n=5, exclude_bought=True):
    """Recommend top_n products for a customer using the blended content+collaborative similarity."""
    bought = cp_matrix.loc[customer]
    bought_products = bought[bought > 0].index.tolist()
    bought_products = [p for p in bought_products if p in hybrid_sim_df.index]
    if not bought_products:
        return pd.Series(dtype=float, name='Score')
    scores = hybrid_sim_df.loc[bought_products].mean(axis=0)
    if exclude_bought:
        scores = scores.drop(index=bought_products, errors='ignore')
    return scores.sort_values(ascending=False).head(top_n).rename('Score')

print(f"Hybrid recommendations for {sample_customer}:")
recommend_hybrid(sample_customer)

# %% [markdown]
# # Final Recommendations
# Put it together: one function that returns ready-to-use, ranked product recommendations for any customer,
# using the hybrid similarity, with `Priority_Score` as a tie-breaker among close scores (so that, all else
# equal, the recommender nudges toward the products worth pushing commercially).

# %%
[x for x in globals() if 'recommend' in x.lower()]


# %%
cp_matrix = pd.crosstab(
    df["Contact Name"],
    df["Part Group"]
)

print(cp_matrix.shape)
print(cp_matrix.head())

# %%
# Clean Quantity
df["Quantity"] = pd.to_numeric(
    df["Quantity"],
    errors="coerce"
).fillna(0)

# Clean Total VALUE
df["Total VALUE"] = (
    df["Total VALUE"]
    .astype(str)
    .str.replace("₹", "", regex=False)
    .str.replace(",", "", regex=False)
    .str.strip()
)

df["Total VALUE"] = pd.to_numeric(
    df["Total VALUE"],
    errors="coerce"
).fillna(0)

# %%
# ============================================================
# CREATE CUSTOMER FEATURES
# ============================================================

cust_feat = (
    df
    .groupby("Contact Name")
    .agg(
        Frequency=("Contact Name", "size"),
        TotalQty=("Quantity", "sum"),
        Monetary=("Total VALUE", "sum")
    )
    .reset_index()
)

print("✅ Customer features created")

display(cust_feat.head())

# %%
# ============================================================
# TOP CONTACTS BY TOTAL VALUE
# ============================================================

top_contacts = (
    df
    .groupby(
        "Contact Name",
        as_index=False
    )["Total VALUE"]
    .sum()
    .sort_values(
        "Total VALUE",
        ascending=False
    )
    .head(5)
)

display(top_contacts)

# %% [markdown]
#  month-level recommendation

# %%
rec_data = df.copy()


# %%
DEALER_COL = "Contact Name"
MONTH_COL = "Month"
PART_COL = "Part Group"
QTY_COL = "Qty"


# %% [markdown]
# Below Cell analyzes each Dealer's historical Part Group purchases by month using Quantity.
# It calculates cosine similarity between months to find which previous months have purchasing patterns similar to the current month.
# It uses those month similarities as weights to predict which Part Groups each dealer is likely to purchase.
# It removes Part Groups the dealer has already purchased in the current month.
# Finally, it ranks the remaining Part Groups by Estimated Quantity and produces dealer-specific recommendations.

# %%
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# 1. SETTINGS
# ============================================================

DEALER_COL = "Contact Name"
MONTH_COL = "Month"
PART_COL = "Part Group"
QTY_COL = "Quantity"

# Month order
month_order = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
]

# ============================================================
# 2. CLEAN DATA
# ============================================================

data = df.copy()



# Make quantity numeric
data[QTY_COL] = pd.to_numeric(
    data[QTY_COL],
    errors="coerce"
)

data = data.dropna(subset=[QTY_COL])

# Standardize month order
data[MONTH_COL] = pd.Categorical(
    data[MONTH_COL],
    categories=month_order,
    ordered=True
)

# ============================================================
# 3. AGGREGATE PURCHASES
# ============================================================

purchase = (
    data
    .groupby(
        [DEALER_COL, MONTH_COL, PART_COL],
        observed=True
    )[QTY_COL]
    .sum()
    .reset_index()
)

# ============================================================
# 4. DETERMINE CURRENT MONTH
# ============================================================

current_month = pd.Timestamp.today().strftime("%b")

print("Current month:", current_month)

# Safety check
if current_month not in month_order:
    raise ValueError(
        f"{current_month} is not present in month_order"
    )

# ============================================================
# 5. CREATE MONTH × PART GROUP MATRIX
# ============================================================

month_part = (
    purchase
    .pivot_table(
        index=MONTH_COL,
        columns=PART_COL,
        values=QTY_COL,
        aggfunc="sum",
        fill_value=0,
        observed=True
    )
)

# Make sure all months exist
month_part = month_part.reindex(month_order, fill_value=0)

# ============================================================
# 6. CALCULATE MONTH SIMILARITY
# ============================================================

month_similarity = cosine_similarity(month_part)

month_similarity_df = pd.DataFrame(
    month_similarity,
    index=month_part.index,
    columns=month_part.index
)

print("\nMonth similarity:")
display(month_similarity_df.round(3))

# ============================================================
# 7. GET PREVIOUS MONTHS
# ============================================================

current_idx = month_order.index(current_month)

previous_months = month_order[:current_idx]

if len(previous_months) == 0:
    raise ValueError(
        "There are no previous months available for the current month."
    )

# Similarity of current month to previous months
similarities = (
    month_similarity_df
    .loc[current_month, previous_months]
    .sort_values(ascending=False)
)

print("\nSimilarity with previous months:")
display(similarities)

# ============================================================
# 8. CALCULATE WEIGHTS
# ============================================================

# Ignore negative similarities
similarities = similarities.clip(lower=0)

# If all similarities are zero, use equal weights
if similarities.sum() == 0:
    weights = pd.Series(
        1 / len(similarities),
        index=similarities.index
    )
else:
    weights = similarities / similarities.sum()

print("\nMonth weights:")
display(weights.round(3))

# ============================================================
# 9. CREATE DEALER × MONTH × PART GROUP MATRIX
# ============================================================

dealer_month_part = (
    purchase
    .pivot_table(
        index=[DEALER_COL, MONTH_COL],
        columns=PART_COL,
        values=QTY_COL,
        aggfunc="sum",
        fill_value=0,
        observed=True
    )
)

# ============================================================
# 10. RECOMMEND PRODUCTS FOR EACH DEALER
# ============================================================
recommendations = []

dealers = purchase[DEALER_COL].unique()

for dealer in dealers:

    # Products already purchased in current month
    current_purchase = purchase[
        (purchase[DEALER_COL] == dealer) &
        (purchase[MONTH_COL] == current_month)
    ]

    purchased_current = set(
        current_purchase[PART_COL]
    )

    # Historical purchases for this dealer
    dealer_history = purchase[
        (purchase[DEALER_COL] == dealer) &
        (purchase[MONTH_COL].isin(previous_months))
    ]

    # Skip dealers with no historical purchases
    if dealer_history.empty:
        continue

    # Estimate expected quantity for each Part Group
    part_scores = {}

    for _, row in dealer_history.iterrows():

        part = row[PART_COL]
        month = row[MONTH_COL]
        qty = row[QTY_COL]

        # Similarity/weight of this historical month
        similarity_weight = weights.get(month, 0)

        if part not in part_scores:
            part_scores[part] = 0

        part_scores[part] += qty * similarity_weight

    # Remove products already purchased in current month
    for part in purchased_current:
        part_scores.pop(part, None)

    # Store recommendations
    for part, estimated_qty in part_scores.items():

        if estimated_qty > 0:

            recommendations.append({
                DEALER_COL: dealer,
                "Current Month": current_month,
                PART_COL: part,
                "Estimated Quantity": round(
                    estimated_qty, 2
                )
            })


# ============================================================
# CREATE RECOMMENDATION DATAFRAME
# ============================================================

recommendations_df = pd.DataFrame(
    recommendations
)


# ============================================================
# RANK RECOMMENDATIONS FOR EACH DEALER
# ============================================================

if not recommendations_df.empty:

    recommendations_df["Rank"] = (
        recommendations_df
        .groupby(DEALER_COL)["Estimated Quantity"]
        .rank(
            method="first",
            ascending=False
        )
    )

    recommendations_df = (
        recommendations_df
        .sort_values(
            [DEALER_COL, "Rank"]
        )
        .reset_index(drop=True)
    )


# ============================================================
# DISPLAY
# ============================================================

display(recommendations_df)


# ============================================================
# 11. FINAL RECOMMENDATION DATAFRAME
# ============================================================

recommendations_df = pd.DataFrame(
    recommendations
)

# ============================================================
# 12. RANK PRODUCTS WITHIN EACH DEALER
# ============================================================

if not recommendations_df.empty:

    recommendations_df["Rank"] = (
        recommendations_df
        .groupby(DEALER_COL)["Estimated Quantity"]
        .rank(
            method="first",
            ascending=False
        )
    )

    recommendations_df = (
        recommendations_df
        .sort_values(
            [DEALER_COL, "Rank"]
        )
        .reset_index(drop=True)
    )

# ============================================================
# 13. SHOW RESULT
# ============================================================

display(recommendations_df.head(50))


# %% [markdown]
# The Below converts each month (Jan–Jun) into a numeric recency value, where later months get higher weights.
# RecencyWeight gives more importance to recent purchases than older purchases.
# WeightedQty calculates Quantity × RecencyWeight, increasing the influence of recent purchases.
# top_n_per_contact() groups data by Contact Name and ranks recommendations by a selected Score.
# It returns the top N recommendations for each Contact Name, with the highest-scoring items first.

# %%
# recency weight: Jan=1 ... Jun=6, so more recent months matter more
month_map = {
    'Jan': 1,
    'Feb': 2,
    'Mar': 3,
    'Apr': 4,
    'May': 5,
    'Jun': 6
}

df['MonthNum'] = df['Month'].map(month_map)
df['RecencyWeight'] = df['MonthNum']
df['WeightedQty'] = df['Quantity'] * df['RecencyWeight']

def top_n_per_contact(agg_df, n=3, score_col='Score'):
    """Return top-n rows per Contact Name, ranked by score_col desc."""
    return (agg_df.sort_values(['Contact Name', score_col], ascending=[True, False])
                  .groupby('Contact Name').head(n).reset_index(drop=True))

# %%
df['RecencyWeight'] = df['MonthNum']
df['WeightedQty'] = df['Quantity'] * df['RecencyWeight']

# %% [markdown]
# This code calculates each Contact Name's affinity toward each Part Group using quantity, value, frequency, and recency-weighted quantity.
# TotalQty, TotalValue, and Freq measure the contact's overall purchasing behavior for each Part Group.
# WeightedQty gives more importance to recent purchases using the recency weights calculated earlier.
# Score normalizes each Part Group's WeightedQty within the contact, making the strongest Part Group score 1.0.
# Finally, top_n_per_contact() selects the top 3 Part Groups for every Contact Name based on this affinity score.

# %%
# ---- Part-Group-level affinity ----
pg_affinity = (df.groupby(['Contact Name', 'Part Group'])
               .agg(TotalQty=('Quantity', 'sum'),
                    TotalValue=('Total VALUE', 'sum'),
                    Freq=('Month', 'count'),
                    WeightedQty=('WeightedQty', 'sum'))
               .reset_index())

pg_affinity['Score'] = (pg_affinity.groupby('Contact Name')['WeightedQty']
                         .transform(lambda x: x / x.max()))

TOP_N_PART_GROUPS = 3
top_part_groups = top_n_per_contact(pg_affinity, n=TOP_N_PART_GROUPS)
print(top_part_groups.shape)
top_part_groups.head(9)

# %% [markdown]
# This code calculates material-level affinity for each Contact Name, going one level deeper than Part Group.
# It measures each material's Total Quantity, Total Value, purchase Frequency, and Recency-Weighted Quantity.
# Score normalizes WeightedQty within each Contact, so the contact's most strongly preferred material gets a score of 1.0.
# top_n_per_contact() then ranks the materials by Score for each Contact Name.
# Finally, it returns the top 5 materials for every Contact, along with their Part Group and Material Description.

# %%
# ---- Material-level affinity (finer-grained than Part Group) ----
mat_affinity = (df.groupby(['Contact Name', 'Part Group', 'Material', 'Material Desc'])
                .agg(TotalQty=('Quantity', 'sum'),
                     TotalValue=('Total VALUE', 'sum'),
                     Freq=('Month', 'count'),
                     WeightedQty=('WeightedQty', 'sum'))
                .reset_index())

mat_affinity['Score'] = (mat_affinity.groupby('Contact Name')['WeightedQty']
                          .transform(lambda x: x / x.max()))

TOP_N_MATERIALS = 5
top_materials = top_n_per_contact(mat_affinity, n=TOP_N_MATERIALS)
print(top_materials.shape)
top_materials.head(10)

# %% [markdown]
# ## Cross-state majority / trending items
# 
# For every `(Part Group, Material)` combination, compute each state's **share of national value**. If one state accounts for **> 50%** of a material's total sales value, treat that material as *"strongly associated with that state"* (a regional favorite / trending item).
# 
# We then recommend those state-majority items to contacts based in **other** states, **excluding** materials the contact has already purchased — this is a cross-sell / market-expansion recommendation ("customers in State X love this — you haven't tried it yet").

# %%
print(df[['Quantity', 'Total VALUE']].head(10))
print(df[['Quantity', 'Total VALUE']].dtypes)

# %% [markdown]
# This code calculates which states dominate the sales of each Material, using Total VALUE as the basis.
# StateShare represents a material's percentage contribution from each state to its overall sales value.
# A material is classified as a majority-state item when one state's share is greater than 50%.
# contact_state identifies each Contact Name's most frequently transacted state using the mode of Customer State Name.
# The final majority_state_items table identifies materials with strong state-level concentration, which can later be used for cross-state recommendation

# %%
MAJORITY_THRESHOLD = 0.50

print("MAJORITY_THRESHOLD =", MAJORITY_THRESHOLD)

# %%
# --------------------------------------------------
# Convert numeric columns
# --------------------------------------------------

df['Quantity'] = pd.to_numeric(
    df['Quantity'],
    errors='coerce'
)

df['Total VALUE'] = (
    df['Total VALUE']
    .astype(str)
    .str.replace('₹', '', regex=False)
    .str.replace(',', '', regex=False)
    .str.strip()
)

df['Total VALUE'] = pd.to_numeric(
    df['Total VALUE'],
    errors='coerce'
)


# --------------------------------------------------
# State-level aggregation
# --------------------------------------------------

state_mat = (
    df.groupby(
        ['Part Group', 'Material', 'State Name']
    )
    .agg(
        StateQty=('Quantity', 'sum'),
        StateValue=('Total VALUE', 'sum')
    )
    .reset_index()
)


# --------------------------------------------------
# Calculate state share
# --------------------------------------------------

group_total = (
    state_mat
    .groupby(['Part Group', 'Material'])['StateValue']
    .transform('sum')
)

state_mat['StateShare'] = (
    state_mat['StateValue'] / group_total
)


# --------------------------------------------------
# Majority-state items
# --------------------------------------------------

majority_state_items = (
    state_mat[
        state_mat['StateShare'] > MAJORITY_THRESHOLD
    ]
    .copy()
)

majority_state_items = majority_state_items.sort_values(
    'StateValue',
    ascending=False
)
MAJORITY_THRESHOLD = 0.5  # a state must hold > 50% of a material's national value to count as 'majority'
TOP_N_CROSS_STATE = 10

def safe_mode(s):
    m = s.mode()
    return m.iloc[0] if len(m) else 'Unknown'

# each contact's home state = the state they transact from most often
contact_state = df.groupby('Contact Name')['Customer State Name'].agg(safe_mode).to_dict()

# state-level sales per material
state_mat = (df.groupby(['Customer State Name', 'Part Group', 'Material', 'Material Desc'])
             .agg(StateQty=('Quantity', 'sum'), StateValue=('Total VALUE', 'sum'))
             .reset_index())

state_mat['StateShare'] = (state_mat.groupby(['Part Group', 'Material'])['StateValue']
                            .transform(lambda x: x / x.sum()))

majority_state_items = state_mat[state_mat['StateShare'] > MAJORITY_THRESHOLD].copy()
majority_state_items = majority_state_items.sort_values('StateValue', ascending=False)
print(f'{len(majority_state_items)} materials have a clear state-majority (> {MAJORITY_THRESHOLD:.0%} of national value)')
majority_state_items.head(10)

# %% [markdown]
# Below code finds materials that are popular in other states but have not yet been purchased by a particular Contact.
# contact_purchased creates a set of materials that each Contact has already bought, so those materials can be excluded.
# For each Contact, the code identifies their home state and searches for majority-state materials from different states.
# It excludes already-purchased materials and selects the top 3 cross-state materials by StateValue.
# The final cross_state_df gives Contact → Home State → Trending State → Part Group → Material, enabling cross-state product recommendations.

# %%
# materials each contact has already bought (to exclude from cross-state recs)
contact_purchased = df.groupby('Contact Name')['Material'].apply(set).to_dict()

cross_state_recs = []
for contact, home_state in contact_state.items():
    already_bought = contact_purchased.get(contact, set())
    candidates = majority_state_items[
        (majority_state_items['Customer State Name'] != home_state) &
        (~majority_state_items['Material'].isin(already_bought))
    ].sort_values('StateValue', ascending=False).head(TOP_N_CROSS_STATE)

    for _, r in candidates.iterrows():
        cross_state_recs.append({
            'Contact Name': contact,
            'Home State': home_state,
            'Trending In State': r['Customer State Name'],
            'Part Group': r['Part Group'],
            'Material': r['Material'],
            'Material Desc': r['Material Desc'],
            'StateShare': round(r['StateShare'], 3),
        })

cross_state_df = pd.DataFrame(cross_state_recs)
print(cross_state_df.shape)
cross_state_df.head(10)

# %% [markdown]
# ##  Combine into one ranked recommendation table
# 
# Merge the two recommenders into a single table per contact, tagged with a `Reason` so it's clear *why* each item was recommended:
# - `Personal Repeat Purchase` — from Recommender 1 (their own history)
# - `Cross-State Trending` — from Recommender 2 (popular elsewhere, not yet tried)
# 
# Personal-history items are ranked first (stronger signal — proven demand), followed by cross-state suggestions.

# %% [markdown]
# Below code combines two recommendation strategies: personal repeat purchases and cross-state trending materials.
# personal_recs recommends each Contact's top materials based on recency-weighted purchase behavior.
# cross_recs recommends popular materials from other states that the Contact has not necessarily purchased, using StateShare as the score.
# Both recommendation types are combined into final_recommendations, with a Reason and Detail explaining why each item was recommended.
# Finally, recommendations are ranked separately for each Contact and recommendation type, producing one consolidated recommendation table.

# %%
# ============================================================
# CREATE cross_state_df
# ============================================================

cross_state_df = pd.DataFrame(cross_state_recs)


# ============================================================
# CREATE CROSS-STATE RECOMMENDATIONS
# ============================================================

cross_recs = cross_state_df[
    [
        'Contact Name',
        'Part Group',
        'Material',
        'Material Desc',
        'StateShare',
        'Trending In State'
    ]
].copy()


# Rename StateShare → Score
cross_recs = cross_recs.rename(
    columns={
        'StateShare': 'Score'
    }
)


# Convert score from decimal to percentage
cross_recs['Score'] = (
    cross_recs['Score'] * 100
).round(2)


# Add recommendation reason
cross_recs['Reason'] = 'Cross-State Trending'


# Check result
display(cross_recs.head(10))

# %%
# ============================================================
# CREATE CROSS-STATE DATAFRAME
# ============================================================

cross_state_df = pd.DataFrame(cross_state_recs)

print("cross_state_df created successfully!")
print("\nColumns available:")
print(cross_state_df.columns.tolist())

print("\nFirst 5 rows:")
display(cross_state_df.head())

# %%
personal_recs = top_materials[['Contact Name', 'Part Group', 'Material', 'Material Desc', 'Score']].copy()
personal_recs['Reason'] = 'Personal Repeat Purchase'
personal_recs['Detail'] = 'Top ' + str(TOP_N_MATERIALS) + ' by recency-weighted quantity'

cross_recs = cross_state_df[['Contact Name', 'Part Group', 'Material', 'Material Desc', 'StateShare']].copy()
cross_recs = cross_recs.rename(columns={'StateShare': 'Score'})
cross_recs['Reason'] = 'Cross-State Trending'
cross_recs = cross_recs.merge(cross_state_df[['Contact Name', 'Material', 'Trending In State']],
                               on=['Contact Name', 'Material'], how='left')
cross_recs['Detail'] = 'Majority-share in ' + cross_recs['Trending In State']
cross_recs = cross_recs.drop(columns=['Trending In State'])

final_recommendations = pd.concat([personal_recs, cross_recs], ignore_index=True)
final_recommendations['Rank'] = (final_recommendations
                                  .sort_values(['Contact Name', 'Reason', 'Score'], ascending=[True, True, False])
                                  .groupby('Contact Name').cumcount() + 1)
final_recommendations = final_recommendations.sort_values(
    ['Contact Name', 'Reason', 'Score'], ascending=[True, True, False]).reset_index(drop=True)

print('Total recommendation rows:', len(final_recommendations))
final_recommendations.head(15)

# %% [markdown]
# Below code selects the first Contact Name from the final recommendation dataset as an example contact.
# It then filters final_recommendations to show all recommendations belonging to that contact.
# The output includes both Personal Repeat Purchase and Cross-State Trending recommendations.
# This lets you inspect which materials are recommended, their scores, reasons, details, and ranks for one specific contact.
# In short, it is a quick way to validate and understand the recommendations for an individual contact.

# %%
# Export Final Recommendations
OUTPUT_PATH = 'august_recommendations.csv'
final_recommendations.to_csv(OUTPUT_PATH, index=False)
print(f'Saved {len(final_recommendations)} recommendation rows to {OUTPUT_PATH}')
final_recommendations.head(20)

# %%
%whos

# %%
print("df exists:", "df" in globals())
print("cust_feat exists:", "cust_feat" in globals())
print("cp_matrix exists:", "cp_matrix" in globals())
print("cross_output exists:", "cross_output" in globals())
print("final_recommendations exists:", "final_recommendations" in globals())

# %%
final_recommendations = df.copy()

# %%
# ============================================================
# CHECK AVAILABLE COLUMNS
# ============================================================

print("Current columns:")
print(final_recommendations.columns.tolist())


# ============================================================
# POSSIBLE SCORE COLUMN NAMES
# ============================================================

possible_score_columns = [
    "Score",
    "StateShare",
    "WeightedQty",
    "Total_Quantity",
    "TotalQty",
    "Quantity"
]


# ============================================================
# FIND AN AVAILABLE SCORE COLUMN
# ============================================================

score_source = None

for col in possible_score_columns:
    if col in final_recommendations.columns:
        score_source = col
        break


# ============================================================
# CREATE SCORE COLUMN
# ============================================================

if score_source is not None:

    print(f"\nUsing '{score_source}' as the source for Score")

    final_recommendations["Score"] = pd.to_numeric(
        final_recommendations[score_source],
        errors="coerce"
    )

else:

    print("\nERROR: No suitable score column was found.")
    print("\nAvailable columns are:")
    print(final_recommendations.columns.tolist())

# %%
# ============================================================
# MAKE SURE SCORE IS NUMERIC
# ============================================================

final_recommendations["Score"] = pd.to_numeric(
    final_recommendations["Score"],
    errors="coerce"
)

# Remove rows where Score is missing
final_recommendations = final_recommendations.dropna(
    subset=["Score"]
)


# ============================================================
# SORT RECOMMENDATIONS
# ============================================================

final_recommendations = final_recommendations.sort_values(
    by=["Contact Name", "Score"],
    ascending=[True, False]
)


# ============================================================
# CREATE CONTINUOUS RANK FOR EACH CONTACT
# ============================================================

final_recommendations["Rank"] = (
    final_recommendations
    .groupby("Contact Name")
    .cumcount()
    + 1
)


# ============================================================
# DISPLAY RESULT
# ============================================================

final_recommendations.head(20)

# %%
final_recommendations["Score"] = pd.to_numeric(
    final_recommendations["Score"],
    errors="coerce"
)

final_recommendations["Score"] = (
    final_recommendations["Score"]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0)
)

# %%
# ============================================================
# 1. MAKE SURE PERSONAL RECOMMENDATIONS HAVE REASON
# ============================================================

personal_recs = personal_recs.copy()

personal_recs["Reason"] = "Personal Repeat Purchase"


# ============================================================
# 2. MAKE SURE CROSS-STATE RECOMMENDATIONS HAVE REASON
# ============================================================

cross_recs = cross_recs.copy()

cross_recs["Reason"] = "Cross-State Trending"


# ============================================================
# 3. COMBINE BOTH RECOMMENDATION TYPES
# ============================================================

final_recommendations = pd.concat(
    [personal_recs, cross_recs],
    ignore_index=True,
    sort=False
)


# ============================================================
# 4. CHECK REQUIRED COLUMNS
# ============================================================

print("Available columns:")
print(final_recommendations.columns.tolist())

# %% [markdown]
# ## salesperson-friendly presentation layer

# %%
# creating The Rank
final_recommendations["Rank"] = (
    final_recommendations["Score"]
    .rank(
        method="first",
        ascending=False
    )
    .astype(int)
)

# %%
# ============================================================
# 2. SALESPERSON RECOMMENDATION FUNCTION
# ============================================================

def salesperson_recommendations(
    contact_name,
    top_personal=5,
    top_cross_state=5
):
    
    # --------------------------------------------------------
    # Filter Contact
    # --------------------------------------------------------
    
    contact_data = final_recommendations[
        final_recommendations["Contact Name"] == contact_name
    ].copy()
    
    if contact_data.empty:
        print(f"No recommendations found for: {contact_name}")
        return None
    
    
    # --------------------------------------------------------
    # PERSONAL REPEAT PURCHASE
    # --------------------------------------------------------
    
    personal = (
        contact_data[
            contact_data["Reason"] == "Personal Repeat Purchase"
        ]
        .sort_values("Score", ascending=False)
        .head(top_personal)
        .copy()
    )
    
    
    personal["Rank"] = range(1, len(personal) + 1)
    
    
    personal_output = personal[
        [
            "Rank",
            "Part Group",
            "Material Desc",
            "Score"''
            "Total_Qty",
            "Total_Revenue"
        ]
    ].copy()
    
    
    personal_output["Score"] = (
        personal_output["Score"]
        .round(2)
    )
    
    
    # --------------------------------------------------------
    # CROSS-STATE RECOMMENDATIONS
    # --------------------------------------------------------
    
    cross = (
        contact_data[
            contact_data["Reason"] == "Cross-State Trending"
        ]
        .sort_values("Score", ascending=False)
        .head(top_cross_state)
        .copy()
    )
    
    
    cross["Rank"] = range(1, len(cross) + 1)
    
    
    # Extract state from Detail
    cross["Trending State"] = (
        cross["Detail"]
        .str.replace(
            "Majority-share in ",
            "",
            regex=False
        )
    )
    
    
    cross_output = cross[
        [
            "Rank",
            "Part Group",
            "Material Desc",
            "Score",
            "Trending State"
        ]
    ].copy()
    
    
    cross_output["Score"] = (
        cross_output["Score"]
        .round(2)
    )
    
    
    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------
    
    print("\n" + "=" * 80)
    print(f"CONTACT: {contact_name}")
    print("=" * 80)
    
    
    print("\nHIGH-CONFIDENCE REPEAT PURCHASES")
    print("-" * 80)
    
    if personal_output.empty:
        print("No personal repeat-purchase recommendations.")
    else:
        display(personal_output)
    
    
    print("\nNEW MARKET OPPORTUNITIES")
    print("-" * 80)
    
    if cross_output.empty:
        print("No cross-state recommendations.")
    else:
        display(cross_output)
    
    
    return {
        "personal": personal_output,
        "cross_state": cross_output
    }

# %%
def salesperson_recommendations(
    contact_name,
    top_n=10
):

    # Get recommendations for selected contact
    contact_data = final_recommendations[
        final_recommendations["Contact Name"] == contact_name
    ].copy()

    # Check whether contact exists
    if contact_data.empty:
        print(f"No recommendations found for {contact_name}")
        return None


    # Sort by score
    contact_data = contact_data.sort_values(
        "Score",
        ascending=False
    ).head(top_n)


    # Create rank
    contact_data["Rank"] = range(
        1,
        len(contact_data) + 1
    )


    # Select available columns
    preferred_columns = [
        "Rank",
        "Part Group",
        "Material",
        "Material Desc",
        "Score",
        "Total_QTY",
        "Total_Revenue"
    ]

    available_columns = [
        col for col in preferred_columns
        if col in contact_data.columns
    ]


    output = contact_data[
        available_columns
    ].copy()


    return output

# %%
# ============================================================
# RECREATE CROSS OUTPUT FROM cross_recs
# ============================================================

cross_output = cross_recs.copy()

print("cross_output created successfully!")

print("\nColumns available:")
print(cross_output.columns.tolist())

display(cross_output.head())

# %%
# ============================================================
# REBUILD FINAL RECOMMENDATIONS
# ============================================================

# -------------------------------
# PERSONAL RECOMMENDATIONS
# -------------------------------

personal_output = personal_recs.copy()

personal_output["Reason"] = "Personal Repeat Purchase"


# -------------------------------
# CROSS-STATE RECOMMENDATIONS
# -------------------------------

cross_state_output = cross_output.copy()

cross_state_output["Reason"] = "Cross-State Trending"


# -------------------------------
# COMBINE BOTH
# -------------------------------

final_recommendations = pd.concat(
    [personal_output, cross_state_output],
    ignore_index=True,
    sort=False
)


# ============================================================
# MAKE SCORE NUMERIC
# ============================================================

final_recommendations["Score"] = pd.to_numeric(
    final_recommendations["Score"],
    errors="coerce"
)


# Remove rows where Score is missing
final_recommendations = final_recommendations.dropna(
    subset=["Score"]
)


# ============================================================
# CONVERT SCORE TO 0–100
# ============================================================

max_score = final_recommendations["Score"].max()

if max_score <= 1:
    
    final_recommendations["Score Percentage"] = (
        final_recommendations["Score"] * 100
    )

else:
    
    final_recommendations["Score Percentage"] = (
        final_recommendations["Score"]
    )


# Keep values between 0 and 100
final_recommendations["Score Percentage"] = (
    final_recommendations["Score Percentage"]
    .clip(0, 100)
    .round(2)
)


# Display Part Group recommendations with percentage scores
part_group_scores = final_recommendations[
    [
        "Contact Name",
        "Part Group",
        "Score Percentage",
    ]
].copy()




# ============================================================
# SORT RESULTS
# ============================================================

final_recommendations = final_recommendations.sort_values(
    by=["Contact Name", "Score"],
    ascending=[True, False]
).reset_index(drop=True)


# ============================================================
# CREATE CONTINUOUS RANK
# ============================================================

final_recommendations["Rank"] = (
    final_recommendations
    .groupby("Contact Name")
    .cumcount()
    + 1
)


print(final_recommendations.columns.tolist())

display(final_recommendations.head(10))
top_by_desc = (df_sales.groupby(['Part Group','Material Desc'])
               .agg(Total_Qty=('Quantity','sum'), Total_Revenue=('Total VALUE','sum'), Orders=('Bill Doc','nunique'),
                    Customers=('Customer','nunique'))
               .reset_index().sort_values('Total_Qty', ascending=False))

print("Top 15 products by quantity sold:")
top_by_desc.head(15)

# %%
# Check available variables related to cross-state recommendations

variables_to_check = [
    "cross_state_df",
    "cross_state_recs",
    "cross_recs",
    "cross_output",
    "majority_state_items"
]

for var in variables_to_check:
    print(f"{var}: {var in globals()}")

# %%
# ============================================================
# 1. CALCULATE TOTAL QTY AND TOTAL VALUE
#    FOR EVERY PART GROUP + MATERIAL DESCRIPTION
# ============================================================

product_stats = (
    df_sales
    .groupby(
        [
            'Part Group',
            'Material Desc'
        ],
        as_index=False
    )
    .agg(
        Total_Qty=('Quantity', 'sum'),
        Total_Value=('Total VALUE', 'sum'),
        Orders=('Bill Doc', 'nunique'),
        Customers=('Customer', 'nunique')
    )
)


# ============================================================
# 2. SORT PRODUCT STATISTICS
# ============================================================

product_stats = product_stats.sort_values(
    'Total_Qty',
    ascending=False
)


print("Product statistics created successfully:")

display(product_stats.head(15))


# ============================================================
# 3. REMOVE OLD TOTAL COLUMNS FROM RECOMMENDATIONS
#    (SAFE IF YOU RUN THE CELL MULTIPLE TIMES)
# ============================================================

columns_to_remove = [
    'Total_Qty',
    'Total_Value',
    'TotalQty',
    'TotalRevenue',
    'TotalQty_x',
    'TotalQty_y',
    'TotalRevenue_x',
    'TotalRevenue_y'
]

final_recommendations = final_recommendations.drop(
    columns=[
        col for col in columns_to_remove
        if col in final_recommendations.columns
    ],
    errors='ignore'
)


# ============================================================
# 4. MERGE PRODUCT TOTALS WITH RECOMMENDATIONS
# ============================================================

final_recommendations = final_recommendations.merge(
    product_stats[
        [
            'Part Group',
            'Material Desc',
            'Total_Qty',
            'Total_Value',
            'Orders',
            'Customers'
        ]
    ],
    on=[
        'Part Group',
        'Material Desc'
    ],
    how='left'
)


# ============================================================
# 5. HANDLE MISSING VALUES
# ============================================================

numeric_columns = [
    'Total_Qty',
    'Total_Value',
    'Orders',
    'Customers'
]

for col in numeric_columns:

    final_recommendations[col] = (
        final_recommendations[col]
        .fillna(0)
    )


# ============================================================
# 6. ROUND NUMERIC COLUMNS
# ============================================================

final_recommendations['Total_Qty'] = (
    final_recommendations['Total_Qty']
    .round(2)
)

final_recommendations['Total_Value'] = (
    final_recommendations['Total_Value']
    .round(2)
)


# ============================================================
# 7. DISPLAY RECOMMENDATIONS WITH TOTALS
# ============================================================

display(
    final_recommendations[
        [
            'Contact Name',
            'Rank',
            'Part Group',
            'Material Desc',
            'Total_Qty',
            'Total_Value',
            'Orders',
            'Customers',
            'Score'
        ]
    ].head(30)
)

# %%
# ============================================================
# SALESPERSON RECOMMENDATION FUNCTION
# SCORES DISPLAYED AS PERCENTAGES
# ADDITIONAL BUSINESS METRICS:
# TOTAL QUANTITY + TOTAL REVENUE
# ============================================================

import pandas as pd


def salesperson_recommendations(
    contact_name,
    top_personal=5,
    top_cross_state=5
):

    # --------------------------------------------------------
    # CHECK REQUIRED COLUMNS
    # --------------------------------------------------------

    required_columns = [
        "Contact Name",
        "Reason",
        "Score"
    ]

    missing_columns = [
        col
        for col in required_columns
        if col not in final_recommendations.columns
    ]

    if missing_columns:

        print("ERROR: Missing required columns:")
        print(missing_columns)

        print("\nAvailable columns:")
        print(final_recommendations.columns.tolist())

        return None


    # --------------------------------------------------------
    # FILTER DATA FOR SELECTED CONTACT
    # --------------------------------------------------------

    contact_data = (
        final_recommendations[
            final_recommendations["Contact Name"] == contact_name
        ]
        .copy()
    )


    # --------------------------------------------------------
    # CHECK WHETHER CONTACT EXISTS
    # --------------------------------------------------------

    if contact_data.empty:

        print(f"No recommendations found for: {contact_name}")

        return None


    # --------------------------------------------------------
    # MAKE SCORE NUMERIC
    # --------------------------------------------------------

    contact_data["Score"] = pd.to_numeric(
        contact_data["Score"],
        errors="coerce"
    )

    contact_data = contact_data.dropna(
        subset=["Score"]
    )


    # ========================================================
    # PERSONAL REPEAT PURCHASE RECOMMENDATIONS
    # ========================================================

    personal = (
        contact_data[
            contact_data["Reason"]
            == "Personal Repeat Purchase"
        ]
        .sort_values(
            "Score",
            ascending=False
        )
        .head(top_personal)
        .copy()
    )


    # --------------------------------------------------------
    # CREATE PERSONAL RANK
    # --------------------------------------------------------

    personal = personal.reset_index(drop=True)

    personal["Rank"] = personal.index + 1


    # --------------------------------------------------------
    # CONVERT PERSONAL SCORE TO 0-100
    # --------------------------------------------------------

    if not personal.empty:

        max_score = personal["Score"].max()

        if max_score > 0:

            personal["Score"] = (
                personal["Score"] / max_score * 100
            ).round(2)

        else:

            personal["Score"] = 0.00


    # --------------------------------------------------------
    # ENSURE TOTAL_QTY IS NUMERIC
    # --------------------------------------------------------

    if "Total_Qty" in personal.columns:

        personal["Total_Qty"] = pd.to_numeric(
            personal["Total_Qty"],
            errors="coerce"
        ).fillna(0).round(2)


    # --------------------------------------------------------
    # ENSURE TOTAL_REVENUE IS NUMERIC
    # --------------------------------------------------------

    if "Total_Revenue" in personal.columns:

        personal["Total_Revenue"] = pd.to_numeric(
            personal["Total_Revenue"],
            errors="coerce"
        ).fillna(0).round(2)


    # ========================================================
    # CROSS-STATE TRENDING RECOMMENDATIONS
    # ========================================================

    cross_state = (
        contact_data[
            contact_data["Reason"]
            == "Cross-State Trending"
        ]
        .sort_values(
            "Score",
            ascending=False
        )
        .head(top_cross_state)
        .copy()
    )


    # --------------------------------------------------------
    # CREATE CROSS-STATE RANK
    # --------------------------------------------------------

    cross_state = cross_state.reset_index(drop=True)

    cross_state["Rank"] = cross_state.index + 1


    # --------------------------------------------------------
    # CONVERT CROSS-STATE SCORE TO 0-100
    # --------------------------------------------------------

    if not cross_state.empty:

        max_score = cross_state["Score"].max()

        if max_score > 0:

            cross_state["Score"] = (
                cross_state["Score"] / max_score * 100
            ).round(2)

        else:

            cross_state["Score"] = 0.00


    # --------------------------------------------------------
    # ENSURE TOTAL_QTY IS NUMERIC
    # --------------------------------------------------------

    if "Total_Qty" in cross_state.columns:

        cross_state["Total_Qty"] = pd.to_numeric(
            cross_state["Total_Qty"],
            errors="coerce"
        ).fillna(0).round(2)


    # --------------------------------------------------------
    # ENSURE TOTAL_REVENUE IS NUMERIC
    # --------------------------------------------------------

    if "Total_Revenue" in cross_state.columns:

        cross_state["Total_Revenue"] = pd.to_numeric(
            cross_state["Total_Revenue"],
            errors="coerce"
        ).fillna(0).round(2)


    # ========================================================
    # CREATE PERSONAL OUTPUT
    # ========================================================

    personal_columns = [

        col for col in [

            "Rank",
            "Part Group",
            "Material Desc",
            "Total_Qty",
            "Total_Revenue",
            "Score"

        ]

        if col in personal.columns

    ]


    personal_output = (
        personal[personal_columns]
        .rename(
            columns={
                "Total_Qty": "Total Quantity",
                "Total_Revenue": "Total Revenue",
                "Score": "Recommendation Score"
            }
        )
        .copy()
    )


    # --------------------------------------------------------
    # FORMAT PERSONAL SCORE AS PERCENTAGE
    # --------------------------------------------------------

    if (
        not personal_output.empty
        and "Recommendation Score" in personal_output.columns
    ):

        personal_output["Recommendation Score"] = (
            personal_output["Recommendation Score"]
            .apply(lambda x: f"{x:.2f}%")
        )


    # ========================================================
    # CREATE CROSS-STATE OUTPUT
    # ========================================================

    cross_columns = [

        col for col in [

            "Rank",
            "Part Group",
            "Material Desc",
            "Total_Qty",
            "Total_Revenue",
            "Score",
            "Trending State"

        ]

        if col in cross_state.columns

    ]


    cross_state_output = (
        cross_state[cross_columns]
        .rename(
            columns={
                "Total_Qty": "Total Quantity",
                "Total_Revenue": "Total Revenue",
                "Score": "Recommendation Score"
            }
        )
        .copy()
    )


    # --------------------------------------------------------
    # FORMAT CROSS-STATE SCORE AS PERCENTAGE
    # --------------------------------------------------------

    if (
        not cross_state_output.empty
        and "Recommendation Score" in cross_state_output.columns
    ):

        cross_state_output["Recommendation Score"] = (
            cross_state_output["Recommendation Score"]
            .apply(lambda x: f"{x:.2f}%")
        )


    # ========================================================
    # DISPLAY HEADER
    # ========================================================

    print("\n" + "=" * 80)

    print(f"CONTACT: {contact_name}")

    print("=" * 80)


    # ========================================================
    # DISPLAY PERSONAL RECOMMENDATIONS
    # ========================================================

    print("\nHIGH-CONFIDENCE REPEAT PURCHASES")

    print("-" * 80)


    if personal_output.empty:

        print(
            "No personal repeat-purchase recommendations found."
        )

    else:

        display(personal_output)


    # ========================================================
    # DISPLAY CROSS-STATE RECOMMENDATIONS
    # ========================================================

    print("\nNEW MARKET OPPORTUNITIES")

    print("-" * 80)


    if cross_state_output.empty:

        print(
            "No cross-state recommendations found."
        )

    else:

        display(cross_state_output)


    # ========================================================
    # RETURN RESULTS
    # ========================================================

    return {

        "contact": contact_name,

        "personal_recommendations": personal_output,

        "cross_state_recommendations": cross_state_output

    }

# %%
# ============================================================
# FINAL RECOMMENDATION OUTPUT
# WITH ACTUAL CUMULATIVE SALES FROM ENTIRE INVOICE DATASET
# ============================================================

import pandas as pd
import numpy as np


# ============================================================
# 1. DEFINE COLUMN NAMES
# ============================================================

CONTACT_COL = "Contact Name"
PART_GROUP_COL = "Part Group"
MATERIAL_COL = "Material"
MATERIAL_DESC_COL = "Material Desc"
QTY_COL = "Quantity"
REVENUE_COL = "Total VALUE"


# ============================================================
# 2. CHECK THAT df_sales EXISTS
# ============================================================

if "df_sales" not in globals():
    raise NameError(
        "df_sales is not defined. "
        "Please make sure your original invoice dataset is stored in df_sales."
    )


# ============================================================
# 3. CREATE A COPY OF THE COMPLETE INVOICE DATASET
# ============================================================

invoice_df = df_sales.copy()


# ============================================================
# 4. CONVERT QUANTITY TO NUMERIC
# ============================================================

invoice_df[QTY_COL] = pd.to_numeric(
    invoice_df[QTY_COL],
    errors="coerce"
).fillna(0)


# ============================================================
# 5. CONVERT TOTAL VALUE TO NUMERIC
# ============================================================

invoice_df[REVENUE_COL] = (
    invoice_df[REVENUE_COL]
    .astype(str)
    .str.replace("₹", "", regex=False)
    .str.replace(",", "", regex=False)
    .str.strip()
)

invoice_df[REVENUE_COL] = pd.to_numeric(
    invoice_df[REVENUE_COL],
    errors="coerce"
).fillna(0).round(0)


# ============================================================
# 6. CREATE ACTUAL PRODUCT-LEVEL SALES TOTALS
#
# These totals are calculated from the ENTIRE invoice dataset.
# ============================================================

product_sales_summary = (
    invoice_df
    .groupby(
        [
            PART_GROUP_COL,
            MATERIAL_COL,
            MATERIAL_DESC_COL
        ],
        as_index=False
    )
    .agg(
        Total_Qty=(QTY_COL, "sum"),
        Total_Revenue=(REVENUE_COL, "sum"),
        Total_Orders=("Bill Doc", "nunique")
    )
)


# ============================================================
# 7. ROUND ACTUAL SALES VALUES
# ============================================================

product_sales_summary["Total_Qty"] = (
    product_sales_summary["Total_Qty"]
    .round(2)
)

product_sales_summary["Total_Revenue"] = (
    product_sales_summary["Total_Revenue"]
    .round(2)
)


# ============================================================
# 8. CREATE ACTUAL PART-GROUP LEVEL TOTALS
#
# This gives cumulative totals for the COMPLETE Part Group.
# ============================================================

part_group_sales_summary = (
    invoice_df
    .groupby(
        PART_GROUP_COL,
        as_index=False
    )
    .agg(
        Part_Group_Total_Qty=(QTY_COL, "sum"),
        Part_Group_Total_Revenue=(REVENUE_COL, "sum"),
        Part_Group_Total_Orders=("Bill Doc", "nunique")
    )
)


# ============================================================
# 9. ROUND PART GROUP TOTALS
# ============================================================

part_group_sales_summary[
    "Part_Group_Total_Qty"
] = (
    part_group_sales_summary[
        "Part_Group_Total_Qty"
    ].round(2)
)


part_group_sales_summary[
    "Part_Group_Total_Revenue"
] = (
    part_group_sales_summary[
        "Part_Group_Total_Revenue"
    ].round(2)
)


# ============================================================
# 10. CLEAN OLD SALES COLUMNS
#
# Remove old incorrect Total_Qty / Total_Revenue columns
# before merging fresh values from the complete invoice dataset.
# ============================================================

old_columns_to_remove = [
    "Total_Qty",
   # "Total_Revenue",
    #"Total_Value",
    "Total_Orders",
    "Part_Group_Total_Qty",
    "Part_Group_Total_Revenue",
    "Part_Group_Total_Orders"
]


for col in old_columns_to_remove:

    if col in final_recommendations.columns:

        final_recommendations = (
            final_recommendations
            .drop(columns=[col])
        )


# ============================================================
# 11. MERGE ACTUAL PRODUCT SALES TOTALS
#
# This matches recommendations with their actual cumulative
# sales from the ENTIRE invoice dataset.
# ============================================================

merge_keys = [
    PART_GROUP_COL,
    MATERIAL_COL,
    MATERIAL_DESC_COL
]


# Check which merge columns are available
available_merge_keys = [
    col for col in merge_keys
    if col in final_recommendations.columns
]


# ------------------------------------------------------------
# Preferred merge:
# Part Group + Material + Material Desc
# ------------------------------------------------------------

if len(available_merge_keys) == 3:

    final_recommendations = (
        final_recommendations
        .merge(
            product_sales_summary,
            on=available_merge_keys,
            how="left"
        )
    )


# ------------------------------------------------------------
# Alternative merge:
# Part Group + Material
# ------------------------------------------------------------

elif (
    PART_GROUP_COL in final_recommendations.columns
    and MATERIAL_COL in final_recommendations.columns
):

    product_summary_simple = (
        invoice_df
        .groupby(
            [
                PART_GROUP_COL,
                MATERIAL_COL
            ],
            as_index=False
        )
        .agg(
            Total_Qty=(QTY_COL, "sum"),
            Total_Revenue=(REVENUE_COL, "sum"),
            Total_Orders=("Bill Doc", "nunique")
            
        )
    )

    final_recommendations = (
        final_recommendations
        .merge(
            product_summary_simple,
            on=[
                PART_GROUP_COL,
                MATERIAL_COL
            ],
            how="left"
        )
    )


else:

    raise KeyError(
        "Cannot merge sales totals. "
        "final_recommendations must contain at least "
        "'Part Group' and 'Material'."
    )


# ============================================================
# 12. MERGE COMPLETE PART GROUP TOTALS
# ============================================================

if PART_GROUP_COL in final_recommendations.columns:

    final_recommendations = (
        final_recommendations
        .merge(
            part_group_sales_summary,
            on=PART_GROUP_COL,
            how="left"
        )
    )


# ============================================================
# 13. FILL MISSING SALES VALUES
# ============================================================

sales_columns = [
    "Total_Orders",
    "Part_Group_Total_Qty",
    "Part_Group_Total_Revenue",
    "Part_Group_Total_Orders"
]


for col in sales_columns:

    if col in final_recommendations.columns:

        final_recommendations[col] = (
            pd.to_numeric(
                final_recommendations[col],
                errors="coerce"
            )
            .fillna(0)
            .round(2)
        )


# ============================================================
# 14. MAKE SURE SCORE EXISTS AND IS NUMERIC
# ============================================================

if "Score" not in final_recommendations.columns:

    raise KeyError(
        "'Score' column is not available in final_recommendations."
    )


final_recommendations["Score"] = pd.to_numeric(
    final_recommendations["Score"],
    errors="coerce"
).fillna(0).round(2)


# ============================================================
# 15. DISPLAY IMPORTANT COLUMNS
# ============================================================

print("FINAL RECOMMENDATIONS WITH ACTUAL SALES TOTALS")

display(
    final_recommendations[
        [
            col for col in [
                CONTACT_COL,
                PART_GROUP_COL,
                MATERIAL_COL,
                MATERIAL_DESC_COL,
                "Reason",
                "Score",
                "Total_Qty",
                "Total_Revenue",
                "Total_Orders",
                "Part_Group_Total_Qty",
                "Part_Group_Total_Revenue"
            ]
            if col in final_recommendations.columns
        ]
    ]
    .head(20)
)

# %% [markdown]
# ### Recommendations For Each Contact Name

# %%
# Get all unique contact names
contacts = (
    final_recommendations["Contact Name"]
    .dropna()
    .unique()
)

# Select one contact
example_contact = contacts[0] 

# Generate recommendations
result = salesperson_recommendations(
    contact_name=example_contact,
    top_personal=5,
    top_cross_state=5
)

# %%
# Get all unique contact names
contacts = (
    final_recommendations["Contact Name"]
    .dropna()
    .unique()
)

# Select one contact
example_contact = contacts[1]

# Generate recommendations
result = salesperson_recommendations(
    contact_name=example_contact,
    top_personal=5,
    top_cross_state=5
)

# %%
# Get all unique contact names
contacts = (
    final_recommendations["Contact Name"]
    .dropna()
    .unique()
)

# Select one contact
example_contact = contacts[2]

# Generate recommendations
result = salesperson_recommendations(
    contact_name=example_contact,
    top_personal=5,
    top_cross_state=5
)

# %%
# Get all unique contact names
contacts = (
    final_recommendations["Contact Name"]
    .dropna()
    .unique()
)

# Select one contact
example_contact = contacts[3]

# Generate recommendations
result = salesperson_recommendations(
    contact_name=example_contact,
    top_personal=5,
    top_cross_state=5
)

# %%
# Get all unique contact names
contacts = (
    final_recommendations["Contact Name"]
    .dropna()
    .unique()
)

# Select one contact
example_contact = contacts[4]

# Generate recommendations
result = salesperson_recommendations(
    contact_name=example_contact,
    top_personal=5,
    top_cross_state=5
)

# %%
# Get all unique contact names
contacts = (
    final_recommendations["Contact Name"]
    .dropna()
    .unique()
)

# Select one contact
example_contact = contacts[5]

# Generate recommendations
result = salesperson_recommendations(
    contact_name=example_contact,
    top_personal=5,
    top_cross_state=5
)

# %%
# Get all unique contact names
contacts = (
    final_recommendations["Contact Name"]
    .dropna()
    .unique()
)

# Select one contact
example_contact = contacts[6]

# Generate recommendations
result = salesperson_recommendations(
    contact_name=example_contact,
    top_personal=5,
    top_cross_state=5
)

# %%
# Get all unique contact names
contacts = (
    final_recommendations["Contact Name"]
    .dropna()
    .unique()
)

# Select one contact
example_contact = contacts[7]

# Generate recommendations
result = salesperson_recommendations(
    contact_name=example_contact,
    top_personal=5,
    top_cross_state=5
)

# %%
# Get all unique contact names
contacts = (
    final_recommendations["Contact Name"]
    .dropna()
    .unique()
)

# Select one contact
example_contact = contacts[8]

# Generate recommendations
result = salesperson_recommendations(
    contact_name=example_contact,
    top_personal=5,
    top_cross_state=5
)

# %%
# Get all unique contact names
contacts = (
    final_recommendations["Contact Name"]
    .dropna()
    .unique()
)

# Select one contact
example_contact = contacts[9]

# Generate recommendations
result = salesperson_recommendations(
    contact_name=example_contact,
    top_personal=5,
    top_cross_state=5
)

# %%
# Get all unique contact names
contacts = (
    final_recommendations["Contact Name"]
    .dropna()
    .unique()
)

# Select one contact
example_contact = contacts[10]

# Generate recommendations
result = salesperson_recommendations(
    contact_name=example_contact,
    top_personal=5,
    top_cross_state=5
)

# %%
# Get all unique contact names
contacts = (
    final_recommendations["Contact Name"]
    .dropna()
    .unique()
)

# Select one contact
example_contact = contacts[11]

# Generate recommendations
result = salesperson_recommendations(
    contact_name=example_contact,
    top_personal=5,
    top_cross_state=5
)

# %%
# Get all unique contact names
contacts = (
    final_recommendations["Contact Name"]
    .dropna()
    .unique()
)

# Select one contact
example_contact = contacts[12]

# Generate recommendations
result = salesperson_recommendations(
    contact_name=example_contact,
    top_personal=5,
    top_cross_state=5
)

# %%
# Get all unique contact names
contacts = (
    final_recommendations["Contact Name"]
    .dropna()
    .unique()
)

# Select one contact
example_contact = contacts[13]

# Generate recommendations
result = salesperson_recommendations(
    contact_name=example_contact,
    top_personal=5,
    top_cross_state=5
)

# %%
# Get all unique contact names
contacts = (
    final_recommendations["Contact Name"]
    .dropna()
    .unique()
)

# Select one contact
example_contact = contacts[14]

# Generate recommendations
result = salesperson_recommendations(
    contact_name=example_contact,
    top_personal=5,
    top_cross_state=5
)

# %%
# Get all unique contact names
contacts = (
    final_recommendations["Contact Name"]
    .dropna()
    .unique()
)

# Select one contact
example_contact = contacts[15]

# Generate recommendations
result = salesperson_recommendations(
    contact_name=example_contact,
    top_personal=5,
    top_cross_state=5
)

# %%
# Get all unique contact names
contacts = (
    final_recommendations["Contact Name"]
    .dropna()
    .unique()
)

# Select one contact
example_contact = contacts[16]

# Generate recommendations
result = salesperson_recommendations(
    contact_name=example_contact,
    top_personal=5,
    top_cross_state=5
)

# %%
# Get all unique contact names
contacts = (
    final_recommendations["Contact Name"]
    .dropna()
    .unique()
)

# Select one contact
example_contact = contacts[17]

# Generate recommendations
result = salesperson_recommendations(
    contact_name=example_contact,
    top_personal=5,
    top_cross_state=5
)

# %%
# Get all unique contact names
contacts = (
    final_recommendations["Contact Name"]
    .dropna()
    .unique()
)

# Select one contact
example_contact = contacts[18]

# Generate recommendations
result = salesperson_recommendations(
    contact_name=example_contact,
    top_personal=5,
    top_cross_state=5
)

# %%
# Get all unique contact names
contacts = (
    final_recommendations["Contact Name"]
    .dropna()
    .unique()
)

# Select one contact
example_contact = contacts[19]

# Generate recommendations
result = salesperson_recommendations(
    contact_name=example_contact,
    top_personal=5,
    top_cross_state=5
)

# %%
# Get all unique contact names
contacts = (
    final_recommendations["Contact Name"]
    .dropna()
    .unique()
)

# Select one contact
example_contact = contacts[20]

# Generate recommendations
result = salesperson_recommendations(
    contact_name=example_contact,
    top_personal=5,
    top_cross_state=5
)

# %%
# Get all unique contact names
contacts = (
    final_recommendations["Contact Name"]
    .dropna()
    .unique()
)

# Select one contact
example_contact = contacts[21]

# Generate recommendations
result = salesperson_recommendations(
    contact_name=example_contact,
    top_personal=5,
    top_cross_state=5
)

# %%
# Get all unique contact names
contacts = (
    final_recommendations["Contact Name"]
    .dropna()
    .unique()
)

# Select one contact
example_contact = contacts[22]

# Generate recommendations
result = salesperson_recommendations(
    contact_name=example_contact,
    top_personal=5,
    top_cross_state=5
)

# %%
# Get all unique contact names
contacts = (
    final_recommendations["Contact Name"]
    .dropna()
    .unique()
)

# Select one contact
example_contact = contacts[23]

# Generate recommendations
result = salesperson_recommendations(
    contact_name=example_contact,
    top_personal=5,
    top_cross_state=5
)

# %%
# Get all unique contact names
contacts = (
    final_recommendations["Contact Name"]
    .dropna()
    .unique()
)

# Select one contact
example_contact = contacts[24]

# Generate recommendations
result = salesperson_recommendations(
    contact_name=example_contact,
    top_personal=5,
    top_cross_state=5
)

# %%
# Get all unique contact names
contacts = (
    final_recommendations["Contact Name"]
    .dropna()
    .unique()
)

# Select one contact
example_contact = contacts[25]

# Generate recommendations
result = salesperson_recommendations(
    contact_name=example_contact,
    top_personal=5,
    top_cross_state=5
)

# %%
# Get all unique contact names
contacts = (
    final_recommendations["Contact Name"]
    .dropna()
    .unique()
)

# Select one contact
example_contact = contacts[26]

# Generate recommendations
result = salesperson_recommendations(
    contact_name=example_contact,
    top_personal=5,
    top_cross_state=5
)

# %%
# Get all unique contact names
contacts = (
    final_recommendations["Contact Name"]
    .dropna()
    .unique()
)

# Select one contact
example_contact = contacts[28]

# Generate recommendations
result = salesperson_recommendations(
    contact_name=example_contact,
    top_personal=5,
    top_cross_state=5
)

# %%
# Get all unique contact names
contacts = (
    final_recommendations["Contact Name"]
    .dropna()
    .unique()
)

# Select one contact
example_contact = contacts[29]

# Generate recommendations
result = salesperson_recommendations(
    contact_name=example_contact,
    top_personal=5,
    top_cross_state=5
)

# %%
# Get all unique contact names
contacts = (
    final_recommendations["Contact Name"]
    .dropna()
    .unique()
)

# Select one contact
example_contact = contacts[30]

# Generate recommendations
result = salesperson_recommendations(
    contact_name=example_contact,
    top_personal=5,
    top_cross_state=5
)

# %%
# Get all unique contact names
contacts = (
    final_recommendations["Contact Name"]
    .dropna()
    .unique()
)

# Select one contact
example_contact = contacts[31]

# Generate recommendations
result = salesperson_recommendations(
    contact_name=example_contact,
    top_personal=5,
    top_cross_state=5
)

# %%
# Get all unique contact names
contacts = (
    final_recommendations["Contact Name"]
    .dropna()
    .unique()
)

# Select one contact
example_contact = contacts[32]

# Generate recommendations
result = salesperson_recommendations(
    contact_name=example_contact,
    top_personal=5,
    top_cross_state=5
)

# %%
# Get all unique contact names
contacts = (
    final_recommendations["Contact Name"]
    .dropna()
    .unique()
)

# Select one contact
example_contact = contacts[33]

# Generate recommendations
result = salesperson_recommendations(
    contact_name=example_contact,
    top_personal=5,
    top_cross_state=5
)

# %%
# Get all unique contact names
contacts = (
    final_recommendations["Contact Name"]
    .dropna()
    .unique()
)

# Select one contact
example_contact = contacts[35]

# Generate recommendations
result = salesperson_recommendations(
    contact_name=example_contact,
    top_personal=5,
    top_cross_state=5
)

# %%
# Get all unique contact names
contacts = (
    final_recommendations["Contact Name"]
    .dropna()
    .unique()
)

# Select one contact
example_contact = contacts[34]

# Generate recommendations
result = salesperson_recommendations(
    contact_name=example_contact,
    top_personal=5,
    top_cross_state=5
)

# %%
# Get all unique contact names
contacts = (
    final_recommendations["Contact Name"]
    .dropna()
    .unique()
)

# Select one contact
example_contact = contacts[36]

# Generate recommendations
result = salesperson_recommendations(
    contact_name=example_contact,
    top_personal=5,
    top_cross_state=5
)

# %%
# Get all unique contact names
contacts = (
    final_recommendations["Contact Name"]
    .dropna()
    .unique()
)

# Select one contact
example_contact = contacts[37]

# Generate recommendations
result = salesperson_recommendations(
    contact_name=example_contact,
    top_personal=5,
    top_cross_state=5
)

# %%
# Get all unique contact names
contacts = (
    final_recommendations["Contact Name"]
    .dropna()
    .unique()
)

# Select one contact
example_contact = contacts[38]

# Generate recommendations
result = salesperson_recommendations(
    contact_name=example_contact,
    top_personal=5,
    top_cross_state=5
)

# %%
# Get all unique contact names
contacts = (
    final_recommendations["Contact Name"]
    .dropna()
    .unique()
)

# Select one contact
example_contact = contacts[39]

# Generate recommendations
result = salesperson_recommendations(
    contact_name=example_contact,
    top_personal=5,
    top_cross_state=5
)

# %%
# Get all unique contact names
contacts = (
    final_recommendations["Contact Name"]
    .dropna()
    .unique()
)

# Select one contact
example_contact = contacts[40]

# Generate recommendations
result = salesperson_recommendations(
    contact_name=example_contact,
    top_personal=5,
    top_cross_state=5
)

# %%
# Get all unique contact names
contacts = (
    final_recommendations["Contact Name"]
    .dropna()
    .unique()
)

# Select one contact
example_contact = contacts[41]

# Generate recommendations
result = salesperson_recommendations(
    contact_name=example_contact,
    top_personal=5,
    top_cross_state=5
)

# %%
# Get all unique contact names
contacts = (
    final_recommendations["Contact Name"]
    .dropna()
    .unique()
)

# Select one contact
example_contact = contacts[42]

# Generate recommendations
result = salesperson_recommendations(
    contact_name=example_contact,
    top_personal=5,
    top_cross_state=5
)

# %%
# Get all unique contact names
contacts = (
    final_recommendations["Contact Name"]
    .dropna()
    .unique()
)

# Select one contact
example_contact = contacts[43]

# Generate recommendations
result = salesperson_recommendations(
    contact_name=example_contact,
    top_personal=5,
    top_cross_state=5
)

# %%
# Get all unique contact names
contacts = (
    final_recommendations["Contact Name"]
    .dropna()
    .unique()
)

# Select one contact
example_contact = contacts[44]

# Generate recommendations
result = salesperson_recommendations(
    contact_name=example_contact,
    top_personal=5,
    top_cross_state=5
)

# %%
# Get all unique contact names
contacts = (
    final_recommendations["Contact Name"]
    .dropna()
    .unique()
)

# Select one contact
example_contact = contacts[45]

# Generate recommendations
result = salesperson_recommendations(
    contact_name=example_contact,
    top_personal=5,
    top_cross_state=5
)

# %%
# Get all unique contact names
contacts = (
    final_recommendations["Contact Name"]
    .dropna()
    .unique()
)

# Select one contact
example_contact = contacts[46]

# Generate recommendations
result = salesperson_recommendations(
    contact_name=example_contact,
    top_personal=5,
    top_cross_state=5
)

# %%
# Get all unique contact names
contacts = (
    final_recommendations["Contact Name"]
    .dropna()
    .unique()
)

# Select one contact
example_contact = contacts[47]

# Generate recommendations
result = salesperson_recommendations(
    contact_name=example_contact,
    top_personal=5,
    top_cross_state=5
)

# %%
# Get all unique contact names
contacts = (
    final_recommendations["Contact Name"]
    .dropna()
    .unique()
)

# Select one contact
example_contact = contacts[48]

# Generate recommendations
result = salesperson_recommendations(
    contact_name=example_contact,
    top_personal=5,
    top_cross_state=5
)

# %%
# Get all unique contact names
contacts = (
    final_recommendations["Contact Name"]
    .dropna()
    .unique()
)

# Select one contact
example_contact = contacts[49]

# Generate recommendations
result = salesperson_recommendations(
    contact_name=example_contact,
    top_personal=5,
    top_cross_state=5
)

# %%
# Get all unique contact names
contacts = (
    final_recommendations["Contact Name"]
    .dropna()
    .unique()
)

# Select one contact
example_contact = contacts[50]

# Generate recommendations
result = salesperson_recommendations(
    contact_name=example_contact,
    top_personal=5,
    top_cross_state=5
)

# %%
# ============================================================
# 4. ADD SALESPERSON-FRIENDLY EXPLANATION
# ============================================================

def add_sales_reason(personal, cross_state):
    
    personal = personal.copy()
    cross_state = cross_state.copy()
    
    
    if not personal.empty:
        
        personal["Why Recommended"] = (
            "Previously purchased; "
            "strong recency-weighted buying affinity"
        )
    
    
    if not cross_state.empty:
        
        cross_state["Why Recommended"] = (
            "Strong demand in another state; "
            "potential new-product opportunity"
        )
    
    
    return personal, cross_state

# %%
# ============================================================
# 5. BUSINESS-FRIENDLY OUTPUT
# ============================================================

def create_salesperson_view(contact_name, top_n=5):
    
    contact_data = final_recommendations[
        final_recommendations["Contact Name"] == contact_name
    ].copy()
    
    if contact_data.empty:
        return pd.DataFrame()
    
    
    # --------------------------------------------------------
    # PERSONAL
    # --------------------------------------------------------
    
    personal = (
        contact_data[
            contact_data["Reason"] ==
            "Personal Repeat Purchase"
        ]
        .sort_values("Score", ascending=False)
        .head(top_n)
        .copy()
    )
    
    personal["Recommendation Type"] = (
        "Repeat Purchase Opportunity"
    )
    
    personal["Why Recommended"] = (
        "Previously purchased; strong "
        "recency-weighted buying affinity"
    )
    
    
    # --------------------------------------------------------
    # CROSS STATE
    # --------------------------------------------------------
    
    cross = (
        contact_data[
            contact_data["Reason"] ==
            "Cross-State Trending"
        ]
        .sort_values("Score", ascending=False)
        .head(top_n)
        .copy()
    )
    
    cross["Recommendation Type"] = (
        "New Market Opportunity"
    )
    
    cross["Why Recommended"] = (
        "Strong demand in another state; "
        "potential new-product opportunity"
    )
    
    
    # --------------------------------------------------------
    # COMBINE
    # --------------------------------------------------------
    
    output = pd.concat(
        [personal, cross],
        ignore_index=True
    )
    
    
    # --------------------------------------------------------
    # TRENDING STATE
    # --------------------------------------------------------
    
    output["Trending State"] = (
        output["Detail"]
        .where(
            output["Reason"] ==
            "Cross-State Trending"
        )
        .str.replace(
            "Majority-share in ",
            "",
            regex=False
        )
    )
    
    
    # --------------------------------------------------------
    # FINAL COLUMNS
    # --------------------------------------------------------
    
    output = output[
        [
            "Recommendation Type",
            "Part Group",
            "Material",
            "Material Desc",
            "Score",
            "Trending State",
            "Why Recommended"
        ]
    ].copy()
    
    
    output["Score"] = output["Score"].round(2)
    
    return output

# %%
# ============================================================
# CREATE SALES VIEW FROM FINAL RECOMMENDATIONS
# ============================================================

sales_view = final_recommendations.copy()


# ============================================================
# CREATE RECOMMENDATION TYPE
# ============================================================

sales_view["Recommendation Type"] = (
    sales_view["Reason"]
    .replace({
        "Personal Repeat Purchase":
            "High-Confidence Repeat Purchase",

        "Cross-State Trending":
            "New Market Opportunity"
    })
)


# ============================================================
# CHECK RESULT
# ============================================================

print(sales_view.columns.tolist())

sales_view.head()

# %%
# ============================================================
# CREATE SALES VIEW
# ============================================================

import pandas as pd


# ------------------------------------------------------------
# 1. CREATE SALES VIEW
# ------------------------------------------------------------

sales_view = final_recommendations.copy()


# ------------------------------------------------------------
# 2. CHECK REQUIRED COLUMNS
# ------------------------------------------------------------

required_columns = [
    "Contact Name",
    "Reason",
    "Score"
]


missing_columns = [
    col for col in required_columns
    if col not in sales_view.columns
]


if missing_columns:

    print("ERROR: Missing columns:")
    print(missing_columns)

    print("\nAvailable columns:")
    print(sales_view.columns.tolist())


else:

    # --------------------------------------------------------
    # 3. CREATE RECOMMENDATION TYPE
    # --------------------------------------------------------

    sales_view["Recommendation Type"] = (
        sales_view["Reason"]
        .replace({
            "Personal Repeat Purchase":
                "High-Confidence Repeat Purchase",

            "Cross-State Trending":
                "New Market Opportunity"
        })
    )


    # --------------------------------------------------------
    # 4. CREATE PRIORITY
    # --------------------------------------------------------

    priority_map = {

        "High-Confidence Repeat Purchase": 1,

        "New Market Opportunity": 2
    }


    sales_view["Priority"] = (
        sales_view["Recommendation Type"]
        .map(priority_map)
    )


    # --------------------------------------------------------
    # 5. MAKE SCORE NUMERIC
    # --------------------------------------------------------

    sales_view["Score"] = pd.to_numeric(
        sales_view["Score"],
        errors="coerce"
    )


    # --------------------------------------------------------
    # 6. SORT RECOMMENDATIONS
    # --------------------------------------------------------

    sales_view = (
        sales_view
        .sort_values(
            by=[
                "Contact Name",
                "Priority",
                "Score"
            ],
            ascending=[
                True,
                True,
                False
            ]
        )
        .reset_index(drop=True)
    )


    # --------------------------------------------------------
    # 7. DISPLAY
    # --------------------------------------------------------

    display(
        sales_view.head(20)
    )

# %% [markdown]
# # Hybrid Recommendation System: Model Training & Validation
# 
# **Complete Workflow:** Extract hybrid recommendations → Train models → Validate performance metrics
# 
# **Objective:** Quantify the accuracy and effectiveness of your hybrid recommendation system by:
# 1. Reconstructing the hybrid recommendations from your existing system
# 2. Training classification models on recommended vs. non-recommended products
# 3. Calculating **Accuracy, Precision, Recall, F1-Score, AUC-ROC per Part Group and Material**
# 4. Identifying high-performing vs. low-performing recommendations
# 5. Providing actionable insights for model improvement
# 
# **Data Strategy:** Train on Jan–May, test on June (simulating August validation)
# **Models:** Logistic Regression, Random Forest, Gradient Boosting
# 

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, roc_auc_score,
    classification_report, ConfusionMatrixDisplay
)
from sklearn.metrics.pairwise import cosine_similarity

pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 200)

print('✅ All imports successful')

# %% [markdown]
# ## Build Hybrid Recommendations (from your system)

# %% [markdown]
# **Approach:** Recreate your hybrid recommendation system using:
# 1. **Recency-weighted affinity** — Recent purchases (May, Jun) weighted more than Jan
# 2. **Cross-state trending** — Materials popular in one state recommended to other states
# 3. **Ranking & combination** — Personal repeats ranked first, cross-state suggestions second
# 

# %%
# ========== PREPARE TRAINING DATA ==========
# Use Jan-May for building recommendations (simulating Aug prediction)
TRAIN_MONTHS = [1, 2, 3, 4, 5]
train_df = df[df['MonthNum'].isin(TRAIN_MONTHS)].copy()

print(f'Training data (Jan-May): {len(train_df)} rows')

# ========== RECENCY WEIGHTING ==========
# Recent months weighted more heavily
recency_weight = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
train_df['RecencyWeight'] = train_df['MonthNum'].map(recency_weight)
train_df['WeightedQty'] = train_df['Quantity'] * train_df['RecencyWeight']

print('\n✅ Recency weighting applied (linear 1-5)')

# %%
# ========== RECOMMENDER 1: PERSONALIZED (RECENCY-WEIGHTED) AFFINITY ==========

# Part Group affinity
pg_affinity = (train_df.groupby(['Contact Name', 'Part Group'])
               .agg(TotalQty=('Quantity', 'sum'),
                    TotalValue=('Total VALUE', 'sum'),
                    Freq=('Month', 'count'),
                    WeightedQty=('WeightedQty', 'sum'))
               .reset_index())

pg_affinity['Score'] = (pg_affinity.groupby('Contact Name')['WeightedQty']
                         .transform(lambda x: x / (x.max() + 1e-10)))

TOP_N_PART_GROUPS = 3
top_part_groups = (pg_affinity.sort_values(['Contact Name', 'Score'], ascending=[True, False])
                   .groupby('Contact Name').head(TOP_N_PART_GROUPS).reset_index(drop=True))

print(f'Top {TOP_N_PART_GROUPS} Part Groups per contact: {len(top_part_groups)} rows')

# Material affinity (finer-grained)
mat_affinity = (train_df.groupby(['Contact Name', 'Part Group', 'Material', 'Material Desc'])
                .agg(TotalQty=('Quantity', 'sum'),
                     TotalValue=('Total VALUE', 'sum'),
                     Freq=('Month', 'count'),
                     WeightedQty=('WeightedQty', 'sum'))
                .reset_index())

mat_affinity['Score'] = (mat_affinity.groupby('Contact Name')['WeightedQty']
                          .transform(lambda x: x / (x.max() + 1e-10)))

TOP_N_MATERIALS = 5
top_materials = (mat_affinity.sort_values(['Contact Name', 'Score'], ascending=[True, False])
                 .groupby('Contact Name').head(TOP_N_MATERIALS).reset_index(drop=True))

print(f'Top {TOP_N_MATERIALS} Materials per contact: {len(top_materials)} rows')
print('✅ Personal recommendation affinity scores computed')

# %%
# ========== RECOMMENDER 2: CROSS-STATE TRENDING ==========

def safe_mode(s):
    m = s.mode()
    return m.iloc[0] if len(m) > 0 else 'Unknown'

# Map contacts to their home state
contact_state = train_df.groupby('Contact Name')['Customer State Name'].agg(safe_mode).to_dict()

# State-level material popularity
state_mat = (train_df.groupby(['Customer State Name', 'Part Group', 'Material', 'Material Desc'])
             .agg(StateQty=('Quantity', 'sum'), StateValue=('Total VALUE', 'sum'))
             .reset_index())

state_mat['StateValue'] = pd.to_numeric(state_mat['StateValue'], errors='coerce').astype('float64')
state_mat['StateShare'] = (state_mat.groupby(['Part Group', 'Material'])['StateValue']
                            .transform(lambda x: x / (x.sum() + 1e-10)))

# Flag materials with >50% share in one state (majority signal)
MAJORITY_THRESHOLD = 0.5
majority_state_items = state_mat[state_mat['StateShare'] > MAJORITY_THRESHOLD].copy()
majority_state_items = majority_state_items.sort_values('StateValue', ascending=False)

print(f'{len(majority_state_items)} materials have >50% share in one state (majority signal)')

# Materials each contact has already bought
contact_purchased = train_df.groupby('Contact Name')['Material'].apply(set).to_dict()

# Generate cross-state recommendations
cross_state_recs = []
TOP_N_CROSS_STATE = 3

for contact, home_state in contact_state.items():
    already_bought = contact_purchased.get(contact, set())
    candidates = majority_state_items[
        (majority_state_items['Customer State Name'] != home_state) &
        (~majority_state_items['Material'].isin(already_bought))
    ].sort_values('StateValue', ascending=False).head(TOP_N_CROSS_STATE)
    
    for _, r in candidates.iterrows():
        cross_state_recs.append({
            'Contact Name': contact,
            'Home State': home_state,
            'Trending In State': r['Customer State Name'],
            'Part Group': r['Part Group'],
            'Material': r['Material'],
            'Material Desc': r['Material Desc'],
            'Score': round(r['StateShare'], 3),
        })

cross_state_df = pd.DataFrame(cross_state_recs)
print(f'{len(cross_state_df)} cross-state recommendations generated')
print('✅ Cross-state trending signals computed')

# %%
# ========== COMBINE INTO FINAL RECOMMENDATIONS ==========

# Personal recommendations
personal_recs = top_materials[['Contact Name', 'Part Group', 'Material', 'Material Desc', 'Score']].copy()
personal_recs['Reason'] = 'Personal Repeat Purchase'
personal_recs['Recommendation Type'] = 1  # Target variable

# Cross-state recommendations
cross_recs = cross_state_df[['Contact Name', 'Part Group', 'Material', 'Material Desc', 'Score']].copy()
cross_recs['Reason'] = 'Cross-State Trending'
cross_recs['Recommendation Type'] = 1

# Combine
final_recommendations = pd.concat([personal_recs, cross_recs], ignore_index=True)
final_recommendations = final_recommendations.sort_values(['Contact Name', 'Score'], ascending=[True, False])
final_recommendations['Rank'] = final_recommendations.groupby('Contact Name').cumcount() + 1

print(f'FINAL HYBRID RECOMMENDATIONS: {len(final_recommendations)} rows')
print(f'  - Personal repeats: {len(personal_recs)}')
print(f'  - Cross-state trending: {len(cross_recs)}')
print()
print(final_recommendations.head(15))

# %% [markdown]
# ## Create Validation Dataset (June Purchases) 

# %%
# Use June as test month (simulating August prediction)
TEST_MONTH = 6
test_df = df[df['MonthNum'] == TEST_MONTH].copy()

print(f'Test data (June): {len(test_df)} rows')

# Get actual purchases in June
test_actual_purchases = (test_df.groupby(['Contact Name', 'Material', 'Material Desc', 'Part Group'])
                         .agg(ActualQty=('Quantity', 'sum'), ActualValue=('Total VALUE', 'sum'))
                         .reset_index())

test_actual_purchases['actual_purchase'] = 1  # These were bought

print(f'Unique (Contact, Material) pairs purchased in June: {len(test_actual_purchases)}')

# %%
# ========== CREATE BINARY CLASSIFICATION TARGET ==========

# For each (Contact, Material) in our recommendations,
# check if it was actually bought in June

validation_df = final_recommendations[['Contact Name', 'Part Group', 'Material', 'Material Desc']].copy()
validation_df = validation_df.drop_duplicates()

# Merge with actual June purchases
validation_df = validation_df.merge(
    test_actual_purchases[['Contact Name', 'Material', 'actual_purchase']],
    on=['Contact Name', 'Material'],
    how='left'
)

# 1 = was recommended AND was purchased, 0 = recommended but not purchased
validation_df['target'] = validation_df['actual_purchase'].fillna(0).astype(int)

print(f'Validation dataset: {len(validation_df)} rows')
print(f'  Recommendations that converted (target=1): {(validation_df["target"]==1).sum()}')
print(f'  Recommendations that didn\'t convert (target=0): {(validation_df["target"]==0).sum()}')
print(f'  Conversion rate: {(validation_df["target"]==1).sum() / len(validation_df) * 100:.2f}%')

# %% [markdown]
# ## 5. Feature Engineering for Classification

# %%
# Build features from training data (Jan-May)

# Customer behavior features
customer_features = train_df.groupby('Contact Name').agg(
    Cust_Total_Qty=('Quantity', 'sum'),
    Cust_Total_Value=('Total VALUE', 'sum'),
    Cust_Avg_Order_Value=('Total VALUE', 'mean'),
    Cust_Num_Txn=('Month', 'count'),
    Cust_Num_PartGroups=('Part Group', 'nunique'),
    Cust_Num_Materials=('Material', 'nunique'),
).reset_index()

# Part group features
partgroup_features = train_df.groupby('Part Group').agg(
    PG_Total_Qty=('Quantity', 'sum'),
    PG_Total_Value=('Total VALUE', 'sum'),
    PG_Num_Materials=('Material', 'nunique'),
    PG_Avg_Qty_Per_Txn=('Quantity', 'mean'),
).reset_index()

# Material features
material_features = train_df.groupby(['Material', 'Material Desc']).agg(
    Mat_Total_Qty=('Quantity', 'sum'),
    Mat_Total_Value=('Total VALUE', 'sum'),
    Mat_Num_Customers=('Contact Name', 'nunique'),
    Mat_Avg_Qty_Per_Txn=('Quantity', 'mean'),
).reset_index()

# Customer x Part Group interaction
cust_pg = train_df.groupby(['Contact Name', 'Part Group']).agg(
    CustPG_Qty=('Quantity', 'sum'),
    CustPG_Freq=('Month', 'count'),
).reset_index()

# Customer x Material interaction
cust_mat = train_df.groupby(['Contact Name', 'Material']).agg(
    CustMat_Qty=('Quantity', 'sum'),
    CustMat_Freq=('Month', 'count'),
).reset_index()

print('✅ Features engineered from training data:')
print(f'   Customer features: {len(customer_features)}')
print(f'   Part group features: {len(partgroup_features)}')
print(f'   Material features: {len(material_features)}')
print(f'   Cust×PartGroup interactions: {len(cust_pg)}')
print(f'   Cust×Material interactions: {len(cust_mat)}')

# %%
# Merge features into validation dataset
X = validation_df.copy()

X = X.merge(customer_features, on='Contact Name', how='left')
X = X.merge(partgroup_features, on='Part Group', how='left')
X = X.merge(material_features, on=['Material', 'Material Desc'], how='left')
X = X.merge(cust_pg, on=['Contact Name', 'Part Group'], how='left')
X = X.merge(cust_mat, on=['Contact Name', 'Material'], how='left')

# Fill NaN with 0
feature_cols = [c for c in X.columns if c not in ['Contact Name', 'Material', 'Material Desc', 'Part Group', 'target']]
for col in feature_cols:
    X[col] = X[col].fillna(0).astype(float)

# Handle inf values
X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

y = X['target'].copy()
X = X[feature_cols].copy()

print(f'Feature matrix: {X.shape}')
print(f'Features: {len(feature_cols)}')
print(f'Target distribution: {y.value_counts().to_dict()}')

# %% [markdown]
# ## 6. Train-Test Split & Feature Scaling

# %%
# Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_cols)

print(f'Training set: {X_train.shape[0]} samples')
print(f'Test set: {X_test.shape[0]} samples')
print(f'Positive class in training: {(y_train==1).sum()} ({(y_train==1).sum()/len(y_train)*100:.1f}%)')
print(f'Positive class in test: {(y_test==1).sum()} ({(y_test==1).sum()/len(y_test)*100:.1f}%)')

# %% [markdown]
# Training classification Models

# %%
models = {}

print('Training Logistic Regression...')
lr = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
lr.fit(X_train_scaled, y_train)
models['Logistic Regression'] = lr

print('Training Random Forest...')
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=12)
rf.fit(X_train_scaled, y_train)
models['Random Forest'] = rf

print('Training Gradient Boosting...')
gb = GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=5)
gb.fit(X_train_scaled, y_train)
models['Gradient Boosting'] = gb

print('\n✅ All models trained successfully')

# %% [markdown]
# Evaluating Model Performance

# %%
overall_results = []

for model_name, model in models.items():
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0.5
    
    overall_results.append({
        'Model': model_name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'AUC-ROC': auc,
    })

overall_perf = pd.DataFrame(overall_results)
print('OVERALL MODEL PERFORMANCE')
print('='*80)
print(overall_perf.to_string(index=False))

best_model_name = overall_perf.loc[overall_perf['F1-Score'].idxmax(), 'Model']
best_model = models[best_model_name]
y_pred_best = best_model.predict(X_test_scaled)
y_proba_best = best_model.predict_proba(X_test_scaled)[:, 1]

print(f'\n✅ Best Model: {best_model_name}')

# %% [markdown]
# ### Performance Metrics Per Part Group

# %%
# Add predictions to test set with metadata
test_results = X_test.copy()
test_results['Contact Name'] = validation_df.loc[X_test.index, 'Contact Name'].values
test_results['Material'] = validation_df.loc[X_test.index, 'Material'].values
test_results['Material Desc'] = validation_df.loc[X_test.index, 'Material Desc'].values
test_results['Part Group'] = validation_df.loc[X_test.index, 'Part Group'].values
test_results['actual'] = y_test.values
test_results['predicted'] = y_pred_best
test_results['predicted_proba'] = y_proba_best

# Performance per Part Group
partgroup_results = []

for pg in sorted(test_results['Part Group'].unique()):
    mask = test_results['Part Group'] == pg
    y_a = test_results.loc[mask, 'actual']
    y_p = test_results.loc[mask, 'predicted']
    y_prob = test_results.loc[mask, 'predicted_proba']
    
    if len(y_a) > 0 and y_a.sum() > 0:
        acc = accuracy_score(y_a, y_p)
        prec = precision_score(y_a, y_p, zero_division=0)
        rec = recall_score(y_a, y_p, zero_division=0)
        f1 = f1_score(y_a, y_p, zero_division=0)
        auc_val = roc_auc_score(y_a, y_prob) if len(np.unique(y_a)) > 1 else 0.5
        tp = ((y_a == 1) & (y_p == 1)).sum()
        fp = ((y_a == 0) & (y_p == 1)).sum()
        fn = ((y_a == 1) & (y_p == 0)).sum()
        tn = ((y_a == 0) & (y_p == 0)).sum()
        
        partgroup_results.append({
            'Part Group': pg,
            'Samples': len(y_a),
            'Positives': y_a.sum(),
            'Conversion Rate': y_a.sum() / len(y_a),
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'AUC-ROC': auc_val,
            'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
        })

pg_perf = pd.DataFrame(partgroup_results).sort_values('F1-Score', ascending=False)
print('PERFORMANCE PER PART GROUP')
print('='*100)
print(pg_perf[['Part Group', 'Samples', 'Conversion Rate', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']].to_string(index=False))

# %% [markdown]
# ### Performance Metrics Per Material Desc

# %%
# Performance per Material
material_results = []

for mat_desc in test_results['Material Desc'].dropna().unique():
    mask = test_results['Material Desc'] == mat_desc
    y_a = test_results.loc[mask, 'actual']
    y_p = test_results.loc[mask, 'predicted']
    y_prob = test_results.loc[mask, 'predicted_proba']
    
    if len(y_a) >= 3 and y_a.sum() > 0:
        acc = accuracy_score(y_a, y_p)
        prec = precision_score(y_a, y_p, zero_division=0)
        rec = recall_score(y_a, y_p, zero_division=0)
        f1 = f1_score(y_a, y_p, zero_division=0)
        auc_val = roc_auc_score(y_a, y_prob) if len(np.unique(y_a)) > 1 else 0.5
        tp = ((y_a == 1) & (y_p == 1)).sum()
        
        pg = test_results.loc[mask, 'Part Group'].iloc[0]
        
        material_results.append({
            'Material Desc': mat_desc,
            'Part Group': pg,
            'Samples': len(y_a),
            'Positives': y_a.sum(),
            'Conversion Rate': y_a.sum() / len(y_a),
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'AUC-ROC': auc_val,
            'TP': tp,
        })

mat_perf = pd.DataFrame(material_results).sort_values('F1-Score', ascending=False)
print(f'PERFORMANCE PER MATERIAL (Top 30 by F1-Score)')
print('='*120)
print(mat_perf[['Material Desc', 'Part Group', 'Samples', 'Conversion Rate', 'Precision', 'Recall', 'F1-Score']].head(30).to_string(index=False))
print(f'\n... and {len(mat_perf) - 30} more materials')

# %% [markdown]
# ### Implementing a 5-fold Cross - Validation

# %%
from sklearn.model_selection import cross_validate

scoring = {'accuracy': 'accuracy', 'precision': 'precision', 'recall': 'recall', 'f1': 'f1', 'roc_auc': 'roc_auc'}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print('CROSS-VALIDATION RESULTS (5-Fold)')
print('='*80)

for model_name, model in models.items():
    cv_scores = cross_validate(model, X_train_scaled, y_train, cv=skf, scoring=scoring, n_jobs=-1)
    print(f'{model_name}:')
    print(f'  Accuracy:  {cv_scores["test_accuracy"].mean():.3f} ± {cv_scores["test_accuracy"].std():.3f}')
    print(f'  Precision: {cv_scores["test_precision"].mean():.3f} ± {cv_scores["test_precision"].std():.3f}')
    print(f'  Recall:    {cv_scores["test_recall"].mean():.3f} ± {cv_scores["test_recall"].std():.3f}')
    print(f'  F1-Score:  {cv_scores["test_f1"].mean():.3f} ± {cv_scores["test_f1"].std():.3f}')
    print(f'  AUC-ROC:   {cv_scores["test_roc_auc"].mean():.3f} ± {cv_scores["test_roc_auc"].std():.3f}')
    print()

# %% [markdown]
# ### Feature Importance Analysis

# %%
# Extract feature importance from best model (usually GB or RF)
if 'Random Forest' in [best_model_name]:
    importance_scores = best_model.feature_importances_
elif 'Gradient Boosting' in [best_model_name]:
    importance_scores = best_model.feature_importances_
else:
    # For Logistic Regression, use coefficient magnitudes
    importance_scores = np.abs(best_model.coef_[0])

feature_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': importance_scores
}).sort_values('Importance', ascending=False)

print('TOP 20 IMPORTANT FEATURES')
print('='*60)
print(feature_importance.head(20).to_string(index=False))

# %% [markdown]
# ### Lift And Gain Analysis

# %%
# Decile analysis
test_sorted = test_results.sort_values('predicted_proba', ascending=False).reset_index(drop=True)
n_total = len(test_sorted)
n_pos = (test_sorted['actual'] == 1).sum()

decile_data = []
for d in range(1, 11):
    start = int((d-1) * n_total / 10)
    end = int(d * n_total / 10)
    decile_df = test_sorted.iloc[start:end]
    pos_in_d = (decile_df['actual'] == 1).sum()
    pct_pos_in_d = pos_in_d / len(decile_df) if len(decile_df) > 0 else 0
    baseline = n_pos / n_total
    lift = pct_pos_in_d / baseline if baseline > 0 else 0
    cum_pos = (test_sorted.iloc[:end]['actual'] == 1).sum()
    gain = cum_pos / n_pos if n_pos > 0 else 0
    
    decile_data.append({'Decile': d, 'Positives': pos_in_d, 'Lift': lift, 'Gain': gain*100})

deciles = pd.DataFrame(decile_data)
print('LIFT & GAIN ANALYSIS')
print('='*70)
print(deciles.to_string(index=False))

# %% [markdown]
# ### Generating Visualisations

# %%
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Part Group performance
pg_sorted = pg_perf.sort_values('F1-Score')
axes[0,0].barh(pg_sorted['Part Group'], pg_sorted['F1-Score'], color='steelblue')
axes[0,0].set_xlabel('F1-Score')
axes[0,0].set_title('Performance by Part Group')

# Confusion matrix for best model
cm = confusion_matrix(y_test, y_pred_best)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(ax=axes[0,1])
axes[0,1].set_title(f'{best_model_name} - Confusion Matrix')

# Lift chart
axes[1,0].plot(deciles['Decile'], deciles['Lift'], marker='o', linewidth=2)
axes[1,0].axhline(y=1, color='r', linestyle='--')
axes[1,0].set_xlabel('Decile')
axes[1,0].set_ylabel('Lift')
axes[1,0].set_title('Lift Chart')
axes[1,0].grid(alpha=0.3)

# Feature importance
feature_importance.head(10).sort_values('Importance').plot(x='Feature', y='Importance', kind='barh', ax=axes[1,1], color='orange')
axes[1,1].set_xlabel('Importance')
axes[1,1].set_title('Top 10 Features')

plt.tight_layout()
plt.savefig('model_validation_results.png', dpi=100, bbox_inches='tight')
plt.show()

print('✅ Visualizations saved')

# %%
# Export all results
overall_perf.to_csv('01_overall_model_performance.csv', index=False)
pg_perf.to_csv('02_performance_per_partgroup.csv', index=False)
mat_perf.to_csv('03_performance_per_material.csv', index=False)
deciles.to_csv('04_lift_gain_analysis.csv', index=False)
feature_importance.to_csv('05_feature_importance.csv', index=False)
final_recommendations.to_csv('06_final_recommendations.csv', index=False)
test_results.to_csv('07_test_predictions.csv', index=False)

print('✅ Results exported to CSV files:')
print('   - 01_overall_model_performance.csv')
print('   - 02_performance_per_partgroup.csv')
print('   - 03_performance_per_material.csv')
print('   - 04_lift_gain_analysis.csv')
print('   - 05_feature_importance.csv')
print('   - 06_final_recommendations.csv')
print('   - 07_test_predictions.csv')

# %% [markdown]
# ### Generating Summaries and Important Findings

# %%
print('='*80)
print('HYBRID RECOMMENDATION SYSTEM - MODEL VALIDATION REPORT')
print('='*80)
print()
print('📊 OVERALL PERFORMANCE')
print('-'*80)
best_row = overall_perf.loc[overall_perf['F1-Score'].idxmax()]
print(f'Best Model: {best_row["Model"]}')
print(f'  Accuracy:  {best_row["Accuracy"]:.1%}')
print(f'  Precision: {best_row["Precision"]:.1%}')
print(f'  Recall:    {best_row["Recall"]:.1%}')
print(f'  F1-Score:  {best_row["F1-Score"]:.3f}')
print(f'  AUC-ROC:   {best_row["AUC-ROC"]:.3f}')
print()
print('🎯 PART GROUP RANKINGS')
print('-'*80)
top_pg = pg_perf[['Part Group', 'F1-Score', 'Precision', 'Recall', 'Conversion Rate']].head(7)
print(top_pg.to_string(index=False))
print()
print('⭐ TOP 5 MATERIALS')
print('-'*80)
top_mat = mat_perf[['Material Desc', 'Part Group', 'F1-Score', 'Conversion Rate']].head(5)
print(top_mat.to_string(index=False))
print()
print('📈 BUSINESS IMPACT')
print('-'*80)
print(f'Overall conversion rate: {(test_results["actual"]==1).sum()/len(test_results)*100:.1f}%')
print(f'Lift in top decile: {deciles.iloc[0]["Lift"]:.2f}x')
print(f'Cumulative gain (top 30%): {deciles.iloc[2]["Gain"]:.1f}%')
print()
print('💡 KEY RECOMMENDATIONS')
print('-'*80)
print(f'1. Deploy {best_row["Model"]} for production')
print(f'2. Focus on top {len(pg_perf[pg_perf["F1-Score"]>0.7])} high-performing part groups')
print(f'3. Investigate low-performing materials (F1 < 0.5)')
print(f'4. Use model scores for ranking recommendations')
print(f'5. Monitor model performance monthly on new data')


