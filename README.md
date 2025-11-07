# Weather+ (Open-Meteo compatible, locally calibrated)

This service fetches a strong baseline (default: Open-Meteo) and applies **per-variable calibration** (EMOS/NGR / GBDT / QRF) to **out-perform Open-Meteo** on your box/time via hindcast-trained artifacts.

## Run

```bash
pip install -e .
uvicorn weather_plus.api.server:app --host 0.0.0.0 --port 8000
```
