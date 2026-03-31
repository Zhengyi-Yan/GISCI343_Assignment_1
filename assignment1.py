import pandas as pd
import openmeteo_requests
import requests_cache
from retry_requests import retry

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

print(sensors_df)
