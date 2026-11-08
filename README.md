# Miami rainfall and runoff

Linear regression of annual rainfall at Miami International Airport, and of surface runoff estimated from that rainfall. Rainfall is NOAA’s Global Summary of the Year for station `USW00012839` (1948 through the last finished year). Runoff for 1996–2010 is the Miami-Dade surface-runoff series in [USGS SIR 2014-5162](https://pubs.usgs.gov/sir/2014/5162/), figure 7. A future year is predicted by chaining the two lines: year → rainfall → runoff.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

Training downloads the NOAA file, fits both lines, and writes the model plus a forecast for the next 10 years.

```bash
python rainfall.py
```

That command also prints the fitted equations, R², RMSE, and the slope p-values.

Train again from the CSV already in `data/` without calling NOAA:

```bash
python rainfall.py --offline
```

## Predict

Pass the years you want on the command line. Each of these retrains on the NOAA record (or on the local file with `--offline`) and prints the forecast.

One year:

```bash
python rainfall.py 2040
```

An inclusive range:

```bash
python rainfall.py 2030-2040
```

Several years:

```bash
python rainfall.py 2030 2035 2045
```

Same requests offline, after the NOAA file has been downloaded once:

```bash
python rainfall.py --offline 2040
python rainfall.py --offline 2030-2040
```

With no years, the forecast is the next 10 complete years after the last year in the training data.

Each row is annual inches. `rainfall_low_95` / `rainfall_high_95` and `runoff_low_95` / `runoff_high_95` are 95% prediction intervals for a new year.

## Output files

| Path | Contents |
| --- | --- |
| `data/noaa_miami_intl_gsoy_prcp.csv` | Raw NOAA annual precipitation |
| `data/miami_annual.csv` | Training table: rainfall, USGS runoff where it exists, runoff estimated from rainfall |
| `models/miami_linear_regression.joblib` | Both fitted scikit-learn models and fit statistics |
| `outputs/forecast.csv` | The years you just requested |
| `outputs/model_metrics.json` | Equations, R², RMSE, and p-values |
| `outputs/miami_rainfall_runoff.png` | History, fitted lines, and the forecast bands |

## What the fit means

Rainfall rises by about 0.18 inches per year over 1948–2025. That slope is detectable, and it still leaves most of the year-to-year swing unexplained (R² about 0.11, RMSE about 12 inches). A single-year rainfall forecast is close to the long-term level; the prediction interval is the number to use.

Runoff stays near the 1996–2010 average of about 3.4 inches. Airport rainfall and the USGS runoff series barely move together (R² about 0.05), so treat the runoff interval, not the point value, as the result.






<img width="753" height="357" alt="image" src="https://github.com/user-attachments/assets/ad24b6b7-6fee-4b8e-90f6-366877d0f6c1" />




<img width="1107" height="837" alt="image" src="https://github.com/user-attachments/assets/65b79771-0cd0-4902-8cc4-594ff07ec0d1" />


