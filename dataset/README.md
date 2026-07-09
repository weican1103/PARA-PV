# Dataset directory

Place one of the supported PV Excel files here:

- `Solarstation1-50MW-2019.xlsx`
- `Solarstation6-35MW-20192020.xlsx`

The files are intentionally excluded from Git. The loader expects a timestamp
column named `date` or `Time(year-month-day h:m:s)` and a target column named
`power` or `Power (MW)`. Other numeric columns are treated as covariates.
