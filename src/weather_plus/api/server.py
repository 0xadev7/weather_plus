from fastapi import FastAPI

from weather_plus.routers.forecast import router as forecast_router

app = FastAPI(title="OpenMeteo-Compatible ML Forecast")
app.include_router(forecast_router)
