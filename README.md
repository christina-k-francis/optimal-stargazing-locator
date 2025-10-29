# Optimal Stargazing Locator

A web application that allows users to see a 6-day forecast of stargazing conditions over any region in the continental United States using 2024 static light pollution data from [David J. Lorenz](https://djlorenz.github.io/astronomy/lp/), meteorological data from the [U.S. National Weather Service](https://digital.weather.gov/), and moon phase and moon altitude data from [NASA's Jet Propulsion Laboratory (JPL)](https://ssd.jpl.nasa.gov/planets/eph_export.html).  

---

## Overview

The Optimal Stargazing Locator is a geospatial web tool designed to help users find the best times and places for stargazing in the Continental U.S. I combine satellite-derived light pollution data, meteorological cloud cover forecasts, precipitation forecasts, moon phase, and moon altitude information to evaluate stargazing conditions and consequently provide location-based letter grade ratings across the United States.

---

## Features

* A high-resolution **light pollution map** representing the latest version of the World Atlas of the Artificial Night Sky Brightness developed by [David J. Lorenz](https://djlorenz.github.io/astronomy/lp/) using more recent VIIRS satellite data from 2024. His methodology is based on the one described in [Pierantonio Cinzano's original atlas](http://www.lightpollution.it/cinzano/download/0108052.pdf).
* 7-day **cloud cover**, **precipitation**, **temperature**, and **relative humidity** forecasts from the NOAA National Weather Service, processed and tiled for efficient map rendering
* Dynamic **Stargazing Grade** calculation incorporating aforementioned astronomical and meteorological data.
* Location search with instant visibility grade summary
* Supabase-hosted GeoTIFF->map tiles and Zarr datasets served to the frontend
* Automated data pipeline running every 6 hours to get the latest weather data via Render cron jobs
* Responsive, clean frontend built with modern React and Tailwind CSS

---

## Project Structure

```
Optimal-Stargazing-Locator/
├── cron_job/
│   ├── scripts/                 # Python backend data processing scripts
│   │   ├── main_stargazing_calc.py # Master script for evaluating stagazing conditions across 7-day forecast
│   │   ├── nws_sky_coverage_download.py
│   │   ├── process_sky_coverage.py
│   │   ├── nws_precipitation_probability_download.py
│   │   ├── process_precipitation_probability.py
│   │   ├── nws_average_temperature_download.py
│   │   ├── process_average_temperature.py
│   │   ├── nws_relative_humidity_download.py
│   │   ├── nws_wind_speed_and_direction_download.py
│   │   ├── process_auxillary.py
│   │   ├── utils/               # Collection of helper functions
│   │   │   ├── gif_tools.py
│   │   │   ├── memory_logger.py
│   │   │   ├── tile_tools.py
│   │   │   ├── upload_download_tools.py
│   ├── render_stargazing.yaml 
│   ├── render_skycover.yaml
│   ├── render_precipitation.yaml 
│   ├── render_temperature.yaml
│   ├── render_auxillary.yaml
│   ├── requirements.txt
├── tile_server/
│   ├── tile_server.py # Backend script for serving map tiles to Mapbox Studio for visualization
│   ├── render.yaml # Render.com deployment configuration
│   ├── requirements.txt
└── README.md
```

## Usage Example

1. Open the Optimal Stargazing Locator web app
2. Enter your location or click on any location on the map
3. View your Stargazing Index grade (A-F) for tonight and upcoming days
4. Explore interactive light pollution, precipitation, and cloud cover layers
5. Plan your stargazing trip accordingly!

---

## Known Issues & Limitations

* Coverage limited to the continental U.S.
* Light pollution dataset currently static using 2024 data; updates require manual processing
* Some delay (minutes) between automated data pipeline execution and updated map display
* Heavy cloud cover forecasts may occasionally vary from actual conditions

---

## Contributions

Contributions welcome! Please:

1. Open an issue for bugs or feature suggestions
2. Fork the repo and submit a pull request for code contributions

---

## Acknowledgments

* NOAA/NWS for meteorological forecast data
* David J. Lorenz for the 2024 Nighttime Radiance dataset
* Supabase for cloud storage
* Mapbox for map visualization
* Open-source contributors: Xarray, GDAL, Skyfield, and related geospatial tools

---

## Future Plans ✨

* Implementation for documenting user feedback
* Provide viewable GIF dataset forecasts
* Feautre that extracts lat,lon coordinates of selected map location
* Expand to global coverage
* Incorporate warning for proximity to urban light dome 
* Add real-time wind direction and wind speed animations
* Improve mobile experience
* Incorporate user-submitted stargazing reports
