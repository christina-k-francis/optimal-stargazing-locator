# Optimal Stargazing Locator 🌌

A web application that allows users to see a 6-day forecast of stargazing conditions over any region in the continental United States using 2024 static light pollution data from [David J. Lorenz](https://djlorenz.github.io/astronomy/lp/), meteorological data from the [U.S. National Weather Service](https://digital.weather.gov/), and moon phase and moon altitude data from [NASA's Jet Propulsion Laboratory (JPL)](https://ssd.jpl.nasa.gov/planets/eph_export.html) via the Skyfield python package.  

---

## Usage Example

1. Open the Optimal Stargazing Locator web app
2. Enter your location in the search bar or use the Selector Tool click on any location on the map
3. View the Stargazing Conditions grade (A-F) for tonight and upcoming days
4. Explore interactive light pollution, precipitation, cloud cover, and helpful temperature and relative humidity data layers
5. Plan your stargazing trip accordingly!

## Feedback

If you have any questions, recommendations for improvements, or bugs to report, please let me know using [this form](https://docs.google.com/forms/d/e/1FAIpQLSfdIMB5K-sNsudNJI-uT7wc9Fw7BOt6q37A-dDYr-T4q6boQQ/viewform?usp=header). Thank you!

## Acknowledgments

* NOAA NWS for meteorological forecast data
* David J. Lorenz for the 2024 Artificial Nighttime Radiance dataset
* Cloudflare/R2 for cloud storage
* Mapbox for map visualization
* Open-source contributors: Xarray, GDAL, Skyfield, and Rioxarray

## Future Plans

* Provide viewable GIF dataset forecasts
* Expand to global coverage
* Incorporate warning for proximity to urban light dome 
* Add real-time wind direction and wind speed animations
* Improve mobile experience
* Incorporate user-submitted stargazing reports
