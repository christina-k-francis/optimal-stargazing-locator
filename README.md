# Optimal Stargazing Locator

A web application that allows users to see a 7-day forecast of stargazing conditions over any region in the continental United States, using light pollution data from David Lorenz, and meteorological data from the U.S. National Weather Service.

---

## 🌌 Overview

The Optimal Stargazing Locator is a geospatial web tool designed to help users find the best times and places for stargazing in the U.S. (and eventually the globe). By integrating satellite-derived light pollution data, meteorological cloud cover forecasts, precipitation forecasts, moon phase, and moon altitude information, the application evaluates stargazing conditions and provides location-based quality ratings across the U.S.

This project combines backend geospatial processing, automated data pipelines, and a React-based frontend with interactive map layers.

---

## 📌 Features

* High-resolution **light pollution map** representing the latest version of the World Atlas of the Artificial Night Sky Brightness, developed by David J. Lorenz, based on Pierantonio Cinzano's original atlas, using more recent VIIRS satellite data from 2024
* 7-day **cloud cover**, **precipitation**, **temperature**, and **relative humidity** forecasts from the NOAA National Weather Service, processed and tiled for efficient map rendering
* Dynamic **Stargazing Grade** calculation incorporating light pollution, cloud cover, precipitation, and moon illumination data
* Location search with instant visibility grade summary
* Supabase-hosted GeoTIFF->map tiles and Zarr datasets served to the frontend
* Automated data pipeline running every 6 hours to get the latest weather data via Render cron jobs
* Responsive, clean frontend built with modern React and Tailwind CSS

---

## 📁 Project Structure

```
Optimal-Stargazing-Locator/
├── cron_job/
│   ├── scripts/                 # Python backend data processing scripts
│   │   ├── main_nws_download.py # Master script for automated NWS data download & tile generation
│   │   ├── main_stargazing_calc.py # Master script for evaluating stagazing conditions across 7-day forecast
│   │   ├── nws_sky_coverage_download.py
│   │   ├── nws_precipitation_probability_download.py
│   │   ├── nws_average_temperature_download.py
│   │   ├── nws_relative_humidity_download.py
│   │   ├── nws_wind_speed_and_direction_download.py
│   ├── render.yaml # Render.com deployment configuration for cron job
│   ├── requirements.txt
├── tile_server/
│   ├── tile_server.py # Backend script for serving map tiles to Mapbox Studio for visualization
│   ├── render.yaml # Render.com deployment configuration for tile web server
│   ├── requirements.txt
└── README.md
```

## 📒 Usage Example

1. Open the Optimal Stargazing Locator web app
2. Enter your location or click on any location on the map
3. View your Stargazing Index grade (A-F) for tonight and upcoming days
4. Explore interactive light pollution, precipitation, and cloud cover layers
5. Plan your stargazing trip accordingly!

---

## ⚠ Known Issues & Limitations

* Coverage limited to the continental U.S.
* Light pollution dataset currently static using 2024 data; updates require manual processing
* Some delay (minutes) between automated data pipeline execution and updated map display
* Heavy cloud cover forecasts may occasionally vary from actual conditions

---

## 🤝 Contributing

Contributions welcome! Please:

1. Open an issue for bugs or feature suggestions
2. Fork the repo and submit a pull request for code contributions

---

## 🌟 Acknowledgments

* NOAA/NWS for meteorological forecast data
* David J. Lorenz for the 2024 Nighttime Radiance dataset
* Supabase for cloud storage
* Mapbox for map visualization
* Open-source contributors: Xarray, GDAL, Skyfield, and related geospatial tools

---

## ✨ Future Plans

* Expand to global coverage
* Incorporate warning for proximity to urban light dome 
* Add real-time wind direction and wind speed animations
* Improve mobile experience
* Incorporate user-submitted stargazing reports
