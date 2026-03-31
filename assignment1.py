import pandas as pd
import openmeteo_requests
import requests_cache
from retry_requests import retry
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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

# Part B/C

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

# Make sure all required weather variables are listed here
# The order of variables in hourly or daily is important to assign them correctly below
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
	"latitude": -36.8485,
	"longitude": 174.7635,
	"start_date": "2024-01-01",
	"end_date": "2024-12-31",
	"daily": ["rain_sum", "weather_code", "temperature_2m_mean"],
	"timezone": "Pacific/Auckland",
}
responses = openmeteo.weather_api(url, params = params)

# Process first location. Add a for-loop for multiple locations or weather models
response = responses[0]
print(f"Coordinates: {response.Latitude()}°N {response.Longitude()}°E")
print(f"Elevation: {response.Elevation()} m asl")
print(f"Timezone: {response.Timezone()}{response.TimezoneAbbreviation()}")
print(f"Timezone difference to GMT+0: {response.UtcOffsetSeconds()}s")

# Process daily data. The order of variables needs to be the same as requested.
daily = response.Daily()
daily_rain_sum = daily.Variables(0).ValuesAsNumpy()
daily_weather_code = daily.Variables(1).ValuesAsNumpy()
daily_temperature_2m_mean = daily.Variables(2).ValuesAsNumpy()

daily_data = {"Date": pd.date_range(
	start = pd.to_datetime(daily.Time() + response.UtcOffsetSeconds(), unit = "s", utc = True),
	end =  pd.to_datetime(daily.TimeEnd() + response.UtcOffsetSeconds(), unit = "s", utc = True),
	freq = pd.Timedelta(seconds = daily.Interval()),
	inclusive = "left"
)}

daily_data["rain_sum"] = daily_rain_sum
daily_data["weather_code"] = daily_weather_code
daily_data["temperature_2m_mean"] = daily_temperature_2m_mean

weather_df = pd.DataFrame(data = daily_data)
weather_df["Date"] = pd.to_datetime(weather_df["Date"]).dt.tz_localize(None).dt.normalize()

df = ped_long.merge(weather_df, on="Date", how="left")

weekdays = df[df["is_weekend"] == False]
weekday_mean = weekdays["count"].mean()

weekends = df[df["is_weekend"]]
weekend_mean = weekends["count"].mean()

weekday_hourly = weekdays.groupby("hour")["count"].mean()

weekend_hourly = weekends.groupby("hour")["count"].mean()

sensors = [
    "107 Quay Street",
    "Te Ara Tahuhu Walkway",
    "Commerce Street West",
    "7 Custom Street East",
    "45 Queen Street",
    "30 Queen Street",
    "19 Shortland Street",
    "2 High Street",
    "1 Courthouse Lane",
    "61 Federal Street",
    "59 High Street",
    "210 Queen Street",
    "205 Queen Street",
    "8 Darby Street EW",
    "8 Darby Street NS",
    "261 Queen Street",
    "297 Queen Street",
    "150 K Road",
    "183 K Road",
    "188 Quay Street Lower Albert (EW)",
    "188 Quay Street Lower Albert (NS)"
]

def get_mean_weekday_weekend_count(df, sensor):
    weekdays = df[df["is_weekend"] == False]
    weekdays_mean = weekdays[weekdays["location"] == sensor]["count"].mean()

    weekends = df[df["is_weekend"]]
    weekends_mean = weekends[weekends["location"] == sensor]["count"].mean()

    return weekdays_mean, weekends_mean

sensors_count_dict = {}
for sensor in sensors:
    weekdays_mean, weekends_mean = get_mean_weekday_weekend_count(df, sensor)
    sensors_count_dict[sensor] = {"weekdays_mean": weekdays_mean, "weekends_mean": weekends_mean}

sensors_df = pd.DataFrame.from_dict(sensors_count_dict, orient="index")
sensors_df = sensors_df.reset_index()
sensors_df = sensors_df.rename(columns={"index": "location"})

#difference col: weekends mean - weekdays mean (positive = busier on weekends)

sensors_df["difference"] = sensors_df["weekends_mean"] - sensors_df["weekdays_mean"]

weekend_heavy = sensors_df.sort_values("difference", ascending=False)
weekday_heavy = sensors_df.sort_values("difference", ascending=True)

df["condition"] = df["rain_sum"].apply(lambda x: "wet" if x > 0 else "dry")

#grouped by weekday/weekend and rain/no rain
count_by_weekend_and_condition = df.groupby(["is_weekend", "condition"])["count"].mean()
print(count_by_weekend_and_condition)
#Part D

#weekday and weekend hourly line chart
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xticks(range(0, 24, 2))
ax.grid(True, alpha=0.3)
ax.plot(weekday_hourly.index, weekday_hourly.values, color=mcolors.CSS4_COLORS["seagreen"], marker="o", label="Weekday")
ax.plot(weekend_hourly.index, weekend_hourly.values, color=mcolors.CSS4_COLORS["crimson"], marker="o", label = "Weekend")
ax.set_xlabel('Hours', fontsize=16)
ax.set_ylabel('Mean pedestrian count', fontsize=16)
ax.set_title('Mean Pedestrian Count by Hour (Auckland CBD)', fontsize=18)
ax.tick_params(axis="both", labelsize=14)
ax.legend(fontsize=16)

#dry/wet weekday/weekend bar chart
categories = ["Dry Weekday", "Wet Weekday", "Dry Weekend", "Wet Weekend"]

values = [
    count_by_weekend_and_condition[(False, "dry")],
    count_by_weekend_and_condition[(False, "wet")],
    count_by_weekend_and_condition[(True, "dry")],
    count_by_weekend_and_condition[(True, "wet")]
]
colors = ["mediumseagreen", "seagreen", "lightcoral", "crimson"]
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
for bar in bars:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height + 2,
        f"{height:.0f}",
        ha="center",
        va="bottom",
        fontsize=14
    )
ax.set_xlabel('Day type', fontsize=16)
ax.set_ylabel('Pedestrian count', fontsize=16)
ax.set_title('Average Pedestrian Count by Day Type and Rain Condition', fontsize=18)
ax.grid(axis='y', alpha=0.3)
ax.tick_params(axis="both", labelsize=14)
ax.set_ylim(0, 350)








plt.tight_layout()
plt.show()
