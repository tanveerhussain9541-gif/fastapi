from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def read_root():
    return {"message": "API is running"}


@app.post("/predict")
async def predict(attendance: float, previous_marks: float, study_hours: float):
    import joblib
    import numpy as np

    model = joblib.load("Random_Forest_model.pkl")

    input_data = np.array([[attendance, previous_marks, study_hours]])
    prediction = model.predict(input_data)

    return {"needs_support": bool(prediction[0])}
