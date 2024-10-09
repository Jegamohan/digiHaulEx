import pandas as pd
import numpy as np


# --------------------------------------------------------------
# List/Load all data in data/raw/MetaMotion
# --------------------------------------------------------------

# gps_df -> GPS_data.csv
# nb_df -> New_bookings.csv
# sb_df -> Shipment_bookings.csv

gps_df = pd.read_csv("../../data/raw/GPS_data.csv")
nb_df = pd.read_csv("../../data/raw/New_bookings.csv")
sb_df = pd.read_csv("../../data/raw/Shipment_bookings.csv")

# --------------------------------------------------------------
# #Cleaning GPS data & Checking
# --------------------------------------------------------------

# Cleaning GPS data & Checking
# Convert Timestamp to datetime format
gps_df.dtypes
gps_df.isnull().any()

gps_df["RECORD_TIMESTAMP"] = pd.to_datetime(gps_df["RECORD_TIMESTAMP"], utc=True)

gps_df.dtypes
gps_df.isnull().any()


gps_df.to_pickle("../../data/interim/gps_df_processed.pkl")

# Drop duplicates in shipment data
gps_df.drop_duplicates(inplace=True)

# --------------------------------------------------------------
# Shipment_bookings.csv data & Checking
# --------------------------------------------------------------

sb_df.dtypes
sb_df.isnull().any()

sb_df["FIRST_COLLECTION_SCHEDULE_EARLIEST"] = pd.to_datetime(
    sb_df["FIRST_COLLECTION_SCHEDULE_EARLIEST"], utc=True
)
sb_df["FIRST_COLLECTION_SCHEDULE_LATEST"] = pd.to_datetime(
    sb_df["FIRST_COLLECTION_SCHEDULE_LATEST"], utc=True
)
sb_df["LAST_DELIVERY_SCHEDULE_EARLIEST"] = pd.to_datetime(
    sb_df["LAST_DELIVERY_SCHEDULE_EARLIEST"], utc=True
)
sb_df["LAST_DELIVERY_SCHEDULE_LATEST"] = pd.to_datetime(
    sb_df["LAST_DELIVERY_SCHEDULE_LATEST"], utc=True
)

sb_df.dtypes
sb_df.isnull().any()

# Drop duplicates in shipment data
sb_df.drop_duplicates(inplace=True)


sb_df.to_pickle("../../data/interim/sb_df_processed.pkl")


# --------------------------------------------------------------
# Shipment_bookings.csv data & Checking
# --------------------------------------------------------------

nb_df.dtypes
nb_df.isnull().any()

nb_df["FIRST_COLLECTION_SCHEDULE_EARLIEST"] = pd.to_datetime(
    nb_df["FIRST_COLLECTION_SCHEDULE_EARLIEST"], utc=True
)
nb_df["FIRST_COLLECTION_SCHEDULE_LATEST"] = pd.to_datetime(
    nb_df["FIRST_COLLECTION_SCHEDULE_LATEST"], utc=True
)
nb_df["LAST_DELIVERY_SCHEDULE_EARLIEST"] = pd.to_datetime(
    nb_df["LAST_DELIVERY_SCHEDULE_EARLIEST"], utc=True
)
nb_df["LAST_DELIVERY_SCHEDULE_LATEST"] = pd.to_datetime(
    nb_df["LAST_DELIVERY_SCHEDULE_LATEST"], utc=True
)

nb_df.dtypes
nb_df.isnull().any()

# Drop duplicates in shipment data
nb_df.drop_duplicates(inplace=True)

nb_df.to_pickle("../../data/interim/nb_df_processed.pkl")
