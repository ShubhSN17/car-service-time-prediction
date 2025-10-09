from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

app = Flask(__name__)

# ==========================
# 🔹 Load Model + Encoders
# ==========================
model = joblib.load("model/service_model.pkl")
encoder_car = joblib.load("model/car_name_encoder.pkl")
encoder_service = joblib.load("model/service_type_encoder.pkl")
encoder_engine = joblib.load("model/engine_type_encoder.pkl")

# File to store prediction logs
record_file = "records.csv"
if not os.path.exists(record_file):
    pd.DataFrame(columns=[
        "Car Name", "Service Type", "Engine Type", "Condition",
        "Spare Parts", "Turbocharged", "Check Engine Light",
        "Mileage (km)", "Car Age (years)", "Engine Capacity (cc)",
        "Engine Wear", "Oil Quality", "Temperature (°C)",
        "Months Since Last Service", "Predicted Time (hours)"
    ]).to_csv(record_file, index=False)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # --- read categorical fields (strings)
    car_name = request.form['car_name']
    service_type = request.form['service_type']
    engine_type = request.form['engine_type']

    # --- helper to parse numbers and handle checkboxes/strings
    def to_float(val, default=0.0):
        try:
            return float(val)
        except:
            return default

    def to_int(val, default=0):
        # accepts "on", "true", "1", etc.
        if isinstance(val, str) and val.lower() in ('on','true','yes'):
            return 1
        try:
            return int(float(val))
        except:
            return default

    # --- parse numeric fields IN THE SAME ORDER as X_df.columns used during training
    car_age = to_float(request.form.get('car_age', 0))
    mileage = to_float(request.form.get('mileage', 0))
    parts_to_replace = to_float(request.form.get('parts_to_replace', 0))
    workload_queue = to_float(request.form.get('workload_queue', 0))
    mechanic_experience_years = to_float(request.form.get('mechanic_experience_years', 0))

    # condition & parts availability were mapped during training: map them the same way
    condition_raw = request.form.get('condition_of_vehicle', 'Average')  # e.g. "Good"/"Average"/"Poor"
    cond_map = {'Poor': 0, 'Average': 1, 'Good': 2}
    condition = cond_map.get(condition_raw, to_int(condition_raw, 1))

    parts_raw = request.form.get('availability_of_spare_parts', 'Available')  # e.g. "Available"/"Limited"/"Not Available"
    parts_map = {'Not Available': 0, 'Limited': 1, 'Available': 2}
    parts_avail = parts_map.get(parts_raw, to_int(parts_raw, 2))

    engine_capacity = to_float(request.form.get('engine_capacity', 0))
    num_cylinders = to_int(request.form.get('num_cylinders', 0))
    turbo = to_int(request.form.get('turbo', 0))                 # 1 or 0
    battery_kwh = to_float(request.form.get('battery_kwh', 0))
    oil_quality = to_float(request.form.get('oil_quality', 0))
    check_light = to_int(request.form.get('check_engine_light', 0))  # 1 or 0
    temp = to_float(request.form.get('avg_engine_temp_c', 0))
    service_months = to_float(request.form.get('last_service_months_ago', 0))
    engine_wear = to_float(request.form.get('engine_wear_score', 0))

    # --- Build numeric array in EXACT same order as training X_df.columns
    numeric_data = np.array([[
        car_age,
        mileage,
        parts_to_replace,
        workload_queue,
        mechanic_experience_years,
        condition,
        parts_avail,
        engine_capacity,
        num_cylinders,
        turbo,
        battery_kwh,
        oil_quality,
        check_light,
        temp,
        service_months,
        engine_wear
    ]], dtype=float)  # shape (1, 16)

    # --- encode categorical using saved encoders
    car_encoded = encoder_car.transform([[car_name]])        # shape (1, n_car)
    service_encoded = encoder_service.transform([[service_type]])  # shape (1, n_service)
    engine_encoded = encoder_engine.transform([[engine_type]])     # shape (1, n_engine)

    # --- concatenate in same order as training
    X_input = np.concatenate([numeric_data, car_encoded, service_encoded, engine_encoded], axis=1)

    # --- DEBUG: check shapes (remove or comment out in production)
    print("DEBUG shapes -> numeric:", numeric_data.shape, 
          "car:", car_encoded.shape, 
          "service:", service_encoded.shape, 
          "engine:", engine_encoded.shape,
          "TOTAL:", X_input.shape, 
          "model expects:", model.n_features_in_)

    # sanity check (optional): ensure feature count matches model expectation
    if X_input.shape[1] != model.n_features_in_:
        # helpful debug message printed to console — do not use exception handling yet
        return render_template('index.html', prediction_text=f"Feature mismatch: input {X_input.shape[1]} vs model {model.n_features_in_}")

    # --- predict
    prediction = model.predict(X_input)[0]

    # --- log the record (as you did before)
    new_row = pd.DataFrame([[
        car_name, service_type, engine_type, condition, parts_avail,
        turbo, check_light, mileage, car_age, engine_capacity, engine_wear,
        oil_quality, temp, service_months, round(prediction, 2)
    ]], columns=[
        "Car Name", "Service Type", "Engine Type", "Condition",
        "Spare Parts", "Turbocharged", "Check Engine Light",
        "Mileage (km)", "Car Age (years)", "Engine Capacity (cc)",
        "Engine Wear", "Oil Quality", "Temperature (°C)",
        "Months Since Last Service", "Predicted Time (hours)"
    ])
    new_row.to_csv(record_file, mode='a', header=False, index=False)

    return render_template('index.html', prediction_text=f"Estimated Service Time: {prediction:.2f} hours")

@app.route('/dashboard')
def dashboard():
    df = pd.read_csv(record_file)

    avg_time = df['Predicted Time (hours)'].mean()
    total_preds = len(df)

    plt.figure(figsize=(6, 4))
    plt.hist(df['Predicted Time (hours)'], bins=10, color='skyblue', edgecolor='black')
    plt.title("Distribution of Predicted Service Times")
    plt.xlabel("Service Time (hours)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("static/service_time_chart.png")

    return render_template('dashboard.html',
                           avg_time=round(avg_time, 2),
                           total_preds=total_preds,
                           image_path="static/service_time_chart.png")


if __name__ == '__main__':
    app.run(debug=True)