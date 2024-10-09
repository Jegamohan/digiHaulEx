import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from GeoDistanceCalculator import GeoDistanceCalculator


from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
from FrequencyAbstraction import FourierTransformation
from sklearn.cluster import KMeans

# --------------------------------------------------------------
# Load data sets
# --------------------------------------------------------------

gps_df = pd.read_pickle("../../data/interim/gps_df_processed.pkl")
nb_df = pd.read_pickle("../../data/interim/nb_df_processed.pkl")
sb_df = pd.read_pickle("../../data/interim/sb_df_processed.pkl")


# --------------------------------------------------------------
# GPS
# --------------------------------------------------------------
gps_df.dtypes

# Group by SHIPMENT_NUMBER and calculate timeTaken
gps_time_taken = gps_df.groupby("SHIPMENT_NUMBER")["RECORD_TIMESTAMP"].agg(
    ["min", "max"]
)
# Calculate the time difference
gps_time_taken["gps_time_taken"] = (
    (gps_time_taken["max"] - gps_time_taken["min"]).dt.total_seconds() / (60)
).round()  # Convert to minutes

gps_time_taken.describe()

# gps_df.dtypes
# sb_df.dtypes
# nb_df.dtype


# --------------------------------------------------------------
# sb_df
# --------------------------------------------------------------


sb_df["deliveryDistance"] = (
    sb_df.apply(
        lambda row: GeoDistanceCalculator.from_row(row).calculate_delivery_distance(),
        axis=1,
    )
).round()

sb_df["FIRST_COLLECTION_Difference"] = (
    (
        sb_df["FIRST_COLLECTION_SCHEDULE_LATEST"]
        - sb_df["FIRST_COLLECTION_SCHEDULE_EARLIEST"]
    ).dt.total_seconds()
    / 60
).round()  # Convert to minutes

sb_df["FIRST_COLLECTION_LAST_DELIVERY"] = (
    (
        sb_df["LAST_DELIVERY_SCHEDULE_LATEST"]
        - sb_df["FIRST_COLLECTION_SCHEDULE_LATEST"]
    ).dt.total_seconds()
    / 60
).round()  # Convert to minutes

sb_df["LAST_DELIVERY_Difference"] = (
    (
        sb_df["LAST_DELIVERY_SCHEDULE_LATEST"]
        - sb_df["LAST_DELIVERY_SCHEDULE_EARLIEST"]
    ).dt.total_seconds()
    / 60
).round()  # Convert to minutes


# Merge sb_df &
merged_df = pd.merge(sb_df, gps_time_taken, on="SHIPMENT_NUMBER", how="left")


merged_df["Delivery_Status_time"] = (
    (merged_df["LAST_DELIVERY_SCHEDULE_LATEST"] - merged_df["max"]).dt.total_seconds()
    / 60
).round()  # Convert to minutes


# creating a target variable
merged_df["Delivery_Status"] = np.where(merged_df["Delivery_Status_time"] >= -30, 1, 0)


# Filter the DataFrame to include only rows from October 2023
filtered_df = merged_df[
    merged_df["LAST_DELIVERY_SCHEDULE_EARLIEST"]
    .dt.strftime("%Y-%m")
    .str.contains("2023-10")
]

#####################
# What percentage of shipments met the on-time delivery threshold in October
#

count_of_ones = filtered_df["Delivery_Status"].sum()  # Sum gives the count of 1's
total_count = filtered_df[
    "Delivery_Status"
].count()  # Count gives the total number of entries

# Calculate the percentage
percentage = (count_of_ones / total_count) * 100

# Print the result
print(f"Percentage of Delivery_Status = 1: {percentage:.2f}%")
###################

# further look at the data set
merged_df.describe()


# VEHICLE_SIZE=
merged_df["VEHICLE_SIZE"].unique()

# One-hot encode the VEHICLE_SIZE column
merged_df = pd.get_dummies(
    merged_df,
    columns=[
        "VEHICLE_SIZE",
    ],
    dtype=int,
)

# VEHICLE_BUILD_UP
merged_df["VEHICLE_BUILD_UP"].unique()

# One-hot encode the VEHICLE_SIZE column
merged_df = pd.get_dummies(
    merged_df,
    columns=[
        "VEHICLE_BUILD_UP",
    ],
    dtype=int,
)


# Replace spaces in column names with underscores
merged_df.columns = merged_df.columns.str.replace(" ", "_")

# Drop  columns from the merged DataFrame
merged_df.drop(
    columns=[
        "PROJECT_ID",
        "FIRST_COLLECTION_POST_CODE",
        "LAST_DELIVERY_POST_CODE",
        "CARRIER_DISPLAY_ID",
        "FIRST_COLLECTION_LATITUDE",
        "FIRST_COLLECTION_LONGITUDE",
        "LAST_DELIVERY_LATITUDE",
        "LAST_DELIVERY_LONGITUDE",
        "FIRST_COLLECTION_SCHEDULE_EARLIEST",
        "FIRST_COLLECTION_SCHEDULE_LATEST",
        "LAST_DELIVERY_SCHEDULE_EARLIEST",
        "LAST_DELIVERY_SCHEDULE_LATEST",
        "Delivery_Status_time",
        "min",
        "max",
    ],
    inplace=True,
)


# --------------------------------------------------------------
# nb_df
# --------------------------------------------------------------

nb_df["deliveryDistance"] = (
    nb_df.apply(
        lambda row: GeoDistanceCalculator.from_row(row).calculate_delivery_distance(),
        axis=1,
    )
).round()

nb_df["FIRST_COLLECTION_Difference"] = (
    (
        nb_df["FIRST_COLLECTION_SCHEDULE_LATEST"]
        - nb_df["FIRST_COLLECTION_SCHEDULE_EARLIEST"]
    ).dt.total_seconds()
    / 60
).round()  # Convert to minutes

nb_df["FIRST_COLLECTION_LAST_DELIVERY"] = (
    (
        nb_df["LAST_DELIVERY_SCHEDULE_LATEST"]
        - nb_df["FIRST_COLLECTION_SCHEDULE_LATEST"]
    ).dt.total_seconds()
    / 60
).round()  # Convert to minutes

nb_df["LAST_DELIVERY_Difference"] = (
    (
        nb_df["LAST_DELIVERY_SCHEDULE_LATEST"]
        - nb_df["LAST_DELIVERY_SCHEDULE_EARLIEST"]
    ).dt.total_seconds()
    / 60
).round()  # Convert to minutes


# creating a target variable
nb_df["Delivery_Status"] = "NA"

# further look at the data set
nb_df.describe()


# VEHICLE_SIZE=
nb_df["VEHICLE_SIZE"].unique()

# One-hot encode the VEHICLE_SIZE column
nb_df = pd.get_dummies(
    nb_df,
    columns=[
        "VEHICLE_SIZE",
    ],
    dtype=int,
)

# VEHICLE_BUILD_UP
nb_df["VEHICLE_BUILD_UP"].unique()

# One-hot encode the VEHICLE_SIZE column
nb_df = pd.get_dummies(
    nb_df,
    columns=[
        "VEHICLE_BUILD_UP",
    ],
    dtype=int,
)


# Replace spaces in column names with underscores
nb_df.columns = nb_df.columns.str.replace(" ", "_")

# Drop  columns from the merged DataFrame
nb_df.drop(
    columns=[
        "SHIPPER_ID",
        "FIRST_COLLECTION_POST_CODE",
        "LAST_DELIVERY_POST_CODE",
        "CARRIER_ID",
        "FIRST_COLLECTION_LATITUDE",
        "FIRST_COLLECTION_LONGITUDE",
        "LAST_DELIVERY_LATITUDE",
        "LAST_DELIVERY_LONGITUDE",
        "FIRST_COLLECTION_SCHEDULE_EARLIEST",
        "FIRST_COLLECTION_SCHEDULE_LATEST",
        "LAST_DELIVERY_SCHEDULE_EARLIEST",
        "LAST_DELIVERY_SCHEDULE_LATEST",
    ],
    inplace=True,
)


#####FinalDF

fina_df = pd.concat([nb_df, merged_df], ignore_index=True)

# Replace NaN values with 0
fina_df.fillna(0, inplace=True)

# Set SHIPMENT_NUMBER as the index
fina_df.set_index("SHIPMENT_NUMBER", inplace=True)

# Move Delivery_Status to the first column
# Create a list of columns with 'Delivery_Status' at the front
cols = ["Delivery_Status"] + [
    col for col in fina_df.columns if col != "Delivery_Status"
]

# Reorder the DataFrame based on the new column order
fina_df = fina_df[cols]


####################At this point run the remove_outliers.py#########
fina_df.to_pickle("../../data/interim/fina_df.pkl")
####################################################################


# Bring in the Data once you've dealt with outliers'

f_df = pd.read_pickle("../../data/interim/fina_df_outliers_removed_chauvenets.pkl")
f_df.info()

f_df = f_df.drop(["gps_time_taken"], axis=1)

# --------------------------------------------------------------
# Dealing with missing values (imputation)
# --------------------------------------------------------------

predictor_columns = list(f_df.columns[1:5])

for col in predictor_columns:
    f_df[col] = f_df[col].interpolate()

f_df.info()

# --------------------------------------------------------------
# Principal component analysis PCA
# --------------------------------------------------------------
df_pca = f_df.copy()
PCA = PrincipalComponentAnalysis()

pc_values = PCA.determine_pc_explained_variance(df_pca, predictor_columns)

plt.figure(figsize=(10, 10))
plt.plot(range(1, len(predictor_columns) + 1), pc_values)
plt.xlabel("principal component number")
plt.ylabel("explained variance ")
plt.show()

df_pca = PCA.apply_pca(df_pca, predictor_columns, 2)

subset = df_pca[df_pca["Delivery_Status"] == 0]

subset[["pca_1", "pca_2"]].plot()

# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------


df_cluster = df_pca.copy()
df_cluster = df_cluster.loc[:, (df_cluster != 0).any(axis=0)]

df_cluster.info()

cluster_columns = list(f_df.columns[1:27])

print(df_cluster[cluster_columns].isnull().sum())

# VEHICLE_BUILD_UP_Tractor_Only
k_values = range(2, 10)
inertias = []  # Corrected variable name

for k in k_values:
    subset = df_cluster[cluster_columns]
    kmeans = KMeans(n_clusters=k, n_init=20, random_state=0)
    cluster_labels = kmeans.fit_predict(subset)
    inertias.append(kmeans.inertia_)


plt.figure(figsize=(10, 20))
plt.plot(k_values, inertias)
plt.xlabel("k")
plt.ylabel("Sum of squared distance")
plt.show()

kmeans = KMeans(n_clusters=6, n_init=20, random_state=0)
subset = df_cluster[cluster_columns]
df_cluster["cluster"] = kmeans.fit_predict(subset)

# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

df_cluster.to_pickle("../../data/interim/f_data_features.pkl")
