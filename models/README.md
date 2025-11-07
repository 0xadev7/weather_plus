Place your trained artifacts here, named by Open-Meteo hourly variable:

- temperature_2m.joblib (EMOS/NGR or GBDT/QRF in °C)
- dew_point_2m.joblib (°C)
- surface_pressure.joblib (hPa)
- precipitation.joblib (mm)
- wind_speed_100m.joblib (km/h)
- wind_direction_100m.joblib (degrees from north, clockwise)

Each artifact should accept X = [lat, lon, hour_of_day, lead, baseline] and output a calibrated value or delta.
