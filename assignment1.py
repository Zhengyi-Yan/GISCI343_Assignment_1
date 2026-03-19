import pandas as pd

# Part A
ped = pd.read_csv('data/akl_ped-2024.csv')
ped.rename(columns={"59 High Stret": "59 High Street"}, inplace=True)

mask = ped['Date'].str.match(r'^\d{4}', na=False) # Remove non-date rows (e.g. "Daylight Savings")
ped = ped[mask].copy()

sensor_cols = ped.columns[2:]

ped["Date"] = pd.to_datetime(ped["Date"])
ped["month"] = ped["Date"].dt.month
ped["day_of_week"] = ped["Date"].dt.day_name()
ped["day_num"] = ped["Date"].dt.dayofweek
ped["hour"] = ped["Time"].str.split(":").str[0].astype(int)

ped["is_weekend"] = ped["day_num"] >= 5

ped_long = pd.melt(
    ped,
    id_vars=["Date", "Time", "month", "day_of_week", "day_num", "hour", "is_weekend"],
    value_vars=sensor_cols.tolist(),
    var_name="location",
    value_name="count"
)
print(ped_long.head())
ped_long.info()
# After reshaping the 2024 dataset to long format, 
# I inspected the dataframe with info() and found no null values, so no row deletion or imputation was required.

# Part B
