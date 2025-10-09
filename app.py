from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')  # ✅ Prevent GUI crash on macOS
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
record_file = "predicted_records.csv"
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

    condition_raw = request.form.get('condition_of_vehicle', 'Average')
    cond_map = {'Poor': 0, 'Average': 1, 'Good': 2}
    condition = cond_map.get(condition_raw, to_int(condition_raw, 1))

    parts_raw = request.form.get('availability_of_spare_parts', 'Available')
    parts_map = {'Not Available': 0, 'Limited': 1, 'Available': 2}
    parts_avail = parts_map.get(parts_raw, to_int(parts_raw, 2))

    engine_capacity = to_float(request.form.get('engine_capacity', 0))
    num_cylinders = to_int(request.form.get('num_cylinders', 0))
    turbo = to_int(request.form.get('turbo', 0))
    battery_kwh = to_float(request.form.get('battery_kwh', 0))
    oil_quality = to_float(request.form.get('oil_quality', 0))
    check_light = to_int(request.form.get('check_engine_light', 0))
    temp = to_float(request.form.get('avg_engine_temp_c', 0))
    service_months = to_float(request.form.get('last_service_months_ago', 0))
    engine_wear = to_float(request.form.get('engine_wear_score', 0))

    # --- Build numeric array
    numeric_data = np.array([[
        car_age, mileage, parts_to_replace, workload_queue,
        mechanic_experience_years, condition, parts_avail,
        engine_capacity, num_cylinders, turbo, battery_kwh,
        oil_quality, check_light, temp, service_months, engine_wear
    ]], dtype=float)

    # --- Encode categoricals
    car_encoded = encoder_car.transform([[car_name]])
    service_encoded = encoder_service.transform([[service_type]])
    engine_encoded = encoder_engine.transform([[engine_type]])

    X_input = np.concatenate([numeric_data, car_encoded, service_encoded, engine_encoded], axis=1)

        # --- Predict (improved with realistic minimums)
    raw_pred = model.predict(X_input)[0]

    # Context-based minimum time by service type
    service_min = {
        "Basic": 0.5,
        "Full": 2.0,
        "Engine Repair": 5.0,
        "Transmission": 4.0,
        "Electrical": 1.5,
        "Painting": 3.0
    }

    MIN_SERVICE_TIME = service_min.get(service_type, 1.0)
    prediction = max(raw_pred, MIN_SERVICE_TIME)

    # ✅ --- Save prediction record (only once) ---
    header = [
        "Car Name", "Service Type", "Engine Type", "Condition",
        "Spare Parts", "Turbocharged", "Check Engine Light",
        "Mileage (km)", "Car Age (years)", "Engine Capacity (cc)",
        "Engine Wear", "Oil Quality", "Temperature (°C)",
        "Months Since Last Service", "Predicted Time (hours)"
    ]

    file_exists = os.path.isfile(record_file)
    with open(record_file, 'a', newline='') as f:
        writer = pd.DataFrame([[
            car_name, service_type, engine_type, condition, parts_avail,
            turbo, check_light, mileage, car_age, engine_capacity, engine_wear,
            oil_quality, temp, service_months, round(prediction, 2)
        ]], columns=header)
        writer.to_csv(f, header=not file_exists, index=False)

    print("✅ Prediction logged successfully!")

    return render_template('index.html', prediction_text=f"Estimated Service Time: {prediction:.2f} hours")


@app.route('/dashboard')
def dashboard():
    if not os.path.exists(record_file):
        return render_template('dashboard.html', tables=None, msg="No records found yet. Predict some results first!")

    df = pd.read_csv(record_file)

    # Convert numeric just in case
    df["Predicted Time (hours)"] = pd.to_numeric(df["Predicted Time (hours)"], errors='coerce')

    avg_time = round(df["Predicted Time (hours)"].mean(), 2)
    common_car = df["Car Name"].mode()[0]
    total_records = len(df)

    plt.figure(figsize=(8, 4))
    car_avg = df.groupby("Car Name")["Predicted Time (hours)"].mean().sort_values()
    car_avg.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title("Average Service Time by Car")
    plt.ylabel("Hours")
    plt.tight_layout()
    chart_path = "static/dashboard_chart.png"
    plt.savefig(chart_path)
    plt.close()

    return render_template("dashboard.html",
                           tables=df.tail(10).values.tolist(),
                           avg_time=avg_time,
                           common_car=common_car,
                           total_records=total_records,
                           chart=chart_path)


if __name__ == '__main__':
    app.run(debug=True)