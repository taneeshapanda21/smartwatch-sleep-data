#Load required libraries
library(astsa)
library(forecast)
library(tseries)
library(lubridate)
library(dplyr)
library(readr)

# (S)ARIMA on Smartwatch Sleep Tracking data

sleep <- read_csv("C:/Users/prasa/Desktop/data-science/RStudio/pstat174/smartwatch_sleep_dataset.csv")

# avg sleep duration per month
sleep_monthly <- sleep %>%
  mutate(month = floor_date(date_recorded, "month")) %>%
  group_by(month) %>%
  summarise(
    avg_sleep = mean(duration_minutes, na.rm = TRUE)
  ) %>%
  arrange(month)

start_year  <- year(min(sleep_monthly$month))
start_month <- month(min(sleep_monthly$month))

sleep_ts <- ts(
  sleep_monthly$avg_sleep,
  frequency = 12,
  start = c(start_year, start_month)
)

# plot raw time series
plot(sleep_ts,
     main = "Monthly Average Sleep Duration",
     xlab = "Year",
     ylab = "Avg Sleep Duration (min)")

# This shows the monthly average sleep duration from 2018 to 2025.
# The series oscillates irregularly around a stable level with no systematic upward or downward trend,
# and the amplitude of fluctuation appears roughly constant over time, suggesting constant variance.

adf.test(sleep_ts)

# The plot shows the monthly average sleep duration from 2018 to 2025.
# The series is stationary (d = 0); no differencing required.

acf(sleep_ts, main = "ACF of Monthly Average Sleep Duration")
pacf(sleep_ts, main = "PACF of Monthly Average Sleep Duration")

acf(sleep_ts, lag.max = 40, main = "ACF up to Lag 40 (Seasonal Check)")

# ACF plot shows that all lags fall within 95% confidence bands with no significant spikes.
# PACF shows that all lags fall within 95% confidence bands with no significant spikes.
# Neither the ACF nor PACF shows any statistically significant autocorrelation beyond lag 0.
# This means there is no detectable AR or MA structure in the series.
# No significant spikes visible at lags 12, 24, 36 which indicates the series does not exhibit monthly seasonality.
# A SARIMA model was considered but not supported.

# Model Selection

fit_000 <- Arima(sleep_ts, order = c(0,0,0))  # white noise baseline
fit_100 <- Arima(sleep_ts, order = c(1,0,0))  # AR(1)
fit_001 <- Arima(sleep_ts, order = c(0,0,1))  # MA(1)

# Compare information criteria
AIC(fit_000)
AIC(fit_100)
AIC(fit_001)
BIC(fit_000)
BIC(fit_100)
BIC(fit_001)

# Try SARIMA(0,0,0)(1,0,0)[12] — seasonal AR(1)
fit_sarima <- Arima(sleep_ts, order = c(0,0,0), seasonal = list(order = c(1,0,0), period = 12))
summary(fit_sarima)

# Also try auto.arima with seasonality allowed
auto.arima(sleep_ts, seasonal = TRUE, stepwise = FALSE, approximation = FALSE)

# The results present a nuanced picture.
# AR(1) achieves the lowest AIC, suggesting marginally better fit when penalized lightly for the extra parameter.
# However, ARIMA(0,0,0) achieves the lowest BIC, which applies a stricter penalty.
# Given n = 96 observations, BIC strongly favors the simpler model.
# ARIMA(0,0,0) achieves the lowest BIC and is preferred.
# Alternative models (AR(1), MA(1), SARIMA) provide no meaningful improvement.

fit <- Arima(sleep_ts, order = c(0,0,0))
summary(fit)
checkresiduals(fit)

forecast_12 <- forecast(fit, h = 12)
plot(forecast_12,
     main = "12-Month Forecast: Monthly Avg Sleep Duration",
     xlab = "Year",
     ylab = "Avg Sleep Duration (minutes)",
     col  = "steelblue", fcol = "firebrick", shadecols = c("pink", "lightblue"))
legend("topright", legend = c("Historical", "Forecast", "80% CI", "95% CI"),
       col = c("steelblue","firebrick","lightblue","pink"), lty = c(1,1,NA,NA),
       pch = c(NA,NA,15,15))

# The ARIMA(0,0,0) model estimates mean monthly sleep duration at 449.15 minutes (7 hrs 29 min),
# with RMSE = 4.60 min and MAE = 3.78 min (MAPE = 0.84%).
# All three residual diagnostics confirm white noise behavior, and the Ljung–Box test fails to reject the no-autocorrelation null.
# The 12-month forecast is a flat line at 449.15 min with a constant 95% prediction interval of [440.08, 458.22].
# Intervals do not widen because the d = 0 white noise structure carries the same variance at every horizon.
# Overall, monthly sleep duration shows no exploitable temporal pattern; variation around the long-run mean appears to be random noise.
# ARIMA(0,0,0) stands as the most defensible model for this series.

# Spectral Analysis on Sleep Data

# Raw periodogram diagnostics
raw_spec <- spec.pgram(sleep_ts, log = 'no', taper = 0, fast = FALSE,
                       main = "Raw Periodogram — Monthly Sleep Duration")
abline(v = 1/12, col = 'firebrick', lty = 2, lwd = 1.5)
abline(v = 1/6,  col = 'steelblue', lty = 3, lwd = 1.2)

# Smoothed spectrum
smooth_spec <- spec.pgram(sleep_ts, spans = c(3,3), taper = 0.1, log = 'no',
                          main = "Smoothed Spectrum — Monthly Sleep Duration")

# Points outside KS bands → reject white noise at that frequency
cpgram(sleep_ts, main = "Cumulative Periodogram — White Noise Test")

# Overall, no dominant peaks are observed, and there is no evidence of annual or semi-annual cycles.
# The smoothed spectrum is flat across frequencies.
# The detected peak occurs at a high-frequency noise region, indicating this is not a meaningful cycle.
# The cumulative periodogram curve remains within Kolmogorov–Smirnov bounds, thus we fail to reject white noise.

# Seasonal power extraction
freqs  <- raw_spec$freq
powers <- raw_spec$spec
idx_annual     <- which.min(abs(freqs - 1/12))
idx_semiannual <- which.min(abs(freqs - 1/6))
cat('Annual (f=1/12):', round(powers[idx_annual], 2),
    '| Semi-annual:', round(powers[idx_semiannual], 2),
    '| Mean:', round(mean(powers), 2),
    '| Max:', round(max(powers), 2),
    '| Ratio (max/annual):', round(max(powers)/powers[idx_annual], 2), '\n')

# Power at annual (1/12) and semi-annual (1/6) frequencies is below the mean spectral level.
# There is no evidence of seasonal periodicity.
# Overall spectral diagnostics indicate a flat spectrum with no dominant frequencies, consistent with a white noise process.
