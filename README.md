# Optimal Stargazing Locator 🌌

A web application that allows users to see a 6-day forecast of stargazing conditions over any region in the continental United States using 2024 static light pollution data from [David J. Lorenz](https://djlorenz.github.io/astronomy/lp/), meteorological data from the [U.S. National Weather Service](https://digital.weather.gov/), and moon phase and moon altitude data from [NASA's Jet Propulsion Laboratory (JPL)](https://ssd.jpl.nasa.gov/planets/eph_export.html) via the Skyfield python package.  

---

## Usage Example

1. Open the Optimal Stargazing Locator web application
2. Enter your location in the search bar or use the Selector Hand Tool to click on any location on the map
3. View the Stargazing Conditions grade (A-F) for tonight and upcoming days in that region
4. Plan your stargazing trip accordingly!

## Feedback

If you have any questions, recommendations for improvements, or bugs to report, please let me know using [this form](https://docs.google.com/forms/d/e/1FAIpQLSfdIMB5K-sNsudNJI-uT7wc9Fw7BOt6q37A-dDYr-T4q6boQQ/viewform?usp=header). Thank you!

## Acknowledgments

* NOAA NWS for meteorological forecast data
* David J. Lorenz for the 2024 Artificial Nighttime Radiance dataset
* NASA Jet Propulsion Laboratory for Lunar data
* Cloudflare/R2 for cloud storage
* Mapbox for map visualization
* Open-source contributors: Xarray, GDAL, Skyfield, and Rioxarray

## Future Plans

* Add map layers for light pollution, precipitation, and cloud cover data
* Provide viewable GIFs of forecasts
* Incorporate a direction-aware warning for proximity to an urban light dome 
* Improve the mobile experience
* Expand the stargazing grade forecast coverage area
* Incorporate user-submitted stargazing reports
