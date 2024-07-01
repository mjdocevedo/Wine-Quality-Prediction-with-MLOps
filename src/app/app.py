from fastapi import FastAPI, Form, HTTPException, Depends, status
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from starlette.requests import Request
from hashlib import sha256
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import jwt
from cryptography.fernet import Fernet
from src.pipeline_steps.prediction import PredictionPipeline

app = FastAPI()  # Initializing a FastAPI app

# Constants
JSON_FILE_PATH = os.path.expanduser("./users/users.json")
SECRET_KEY = Fernet.generate_key()
ALGORITHM = "H256"
oauth_scheme = OAuth2PasswordBearer(tokenUrl="/token")

templates = Jinja2Templates(directory="templates")  # Directory for HTML templates

@app.get("/", response_class=HTMLResponse)
async def home_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/train")
async def training():
    os.system("python main.py")
    return {"message": "Training successful!"}

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    fixed_acidity: float = Form(...),
    volatile_acidity: float = Form(...),
    citric_acid: float = Form(...),
    residual_sugar: float = Form(...),
    chlorides: float = Form(...),
    free_sulfur_dioxide: float = Form(...),
    total_sulfur_dioxide: float = Form(...),
    density: float = Form(...),
    pH: float = Form(...),
    sulphates: float = Form(...),
    alcohol: float = Form(...)
):
    try:
        data = np.array([
            fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
            free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol
        ]).reshape(1, 11)

        obj = PredictionPipeline()
        predict = obj.predict(data)

        return templates.TemplateResponse("results.html", {"request": request, "prediction": str(predict)})
    
    except Exception as e:
        print('The Exception message is:', e)
        return templates.TemplateResponse("results.html", {"request": request, "prediction": "something is wrong"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
