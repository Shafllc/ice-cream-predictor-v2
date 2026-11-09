# 🍦 Ice Cream Sales Predictor (version 2: temperature + weekend + rain)

**Live app:** https://ice-cream-predictor-v2.streamlit.app

Pick a city and a date. The app asks a free weather API for the forecast high and the chance of rain, checks whether the date is a weekend, gives all three to a trained model, and shows the predicted number of cones sold with a short business message.

## What changed from version 1

Version 1 (the class example) used **temperature only**. Version 2 adds two more predictors:

| Predictor | Where the app gets it | Effect in the model |
|---|---|---|
| `temperature` | Open-Meteo forecast high (°F) | about +5 cones per degree |
| `is_weekend` | the date (1 = Saturday or Sunday) | about +59 cones |
| `rain_chance` | Open-Meteo chance of rain (%) | about -12 cones per 10% |

On the training data, R² rose from 0.70 (temperature only) to 0.95 (all three).

## Files

- `app.py`: the Streamlit app
- `model.pkl`: the trained model (linear regression), saved from the notebook
- `ice_cream_v2.ipynb`: the notebook that makes the data, compares version 1 and 2, and saves the model
- `requirements.txt`: the packages Streamlit Cloud installs; scikit-learn is pinned to the version that saved the model

The data is made up to show the steps, as in the class example. Replace it with real sales data to use this for a real shop.

## If something goes wrong

- An error when loading `model.pkl`: check that it is next to `app.py` and that the scikit-learn version in `requirements.txt` matches the one used to save it.
- An error about the forecast: the weather service may be busy. Wait a minute and try again.
