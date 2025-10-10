# 🚗 Car Service Time Prediction

An AI-powered Flask web application that predicts **car service completion time** based on multiple real-world factors — car model, service type, engine details, mechanic experience, and more.  
It also provides a **visual analytics dashboard** for insights.

---

## 📘 Overview

This project combines **Machine Learning** and **Web Development** to help workshops or service centers estimate how long a car service will take.  
The model uses trained encoders and regression techniques to deliver realistic service time predictions.

---

## ⚙️ Tech Stack

| Component | Technology |
|------------|-------------|
| **Frontend** | HTML5, CSS3, Bootstrap 5 |
| **Backend** | Flask (Python) |
| **Machine Learning** | Scikit-learn, Pandas, NumPy |
| **Visualization** | Matplotlib |
| **Storage** | CSV-based record logging |

---

## 🧠 Machine Learning Model

The trained **Linear Regression** model is used to predict service time based on:
- Car model and engine type  
- Type of service (Basic, Full, Engine Repair, etc.)  
- Vehicle condition and spare part availability  
- Mechanic’s experience, workload, temperature, and engine wear

A **minimum threshold** ensures predictions are always practical (e.g., basic service ≥ 0.5 hours).

---

## 📂 Folder Structure

```
carServiceTimePrediction/
│
├── app.py                        → Flask application
├── model/                        → Encoders & ML model
│   ├── car_name_encoder.pkl
│   ├── service_type_encoder.pkl
│   ├── engine_type_encoder.pkl
│   └── service_model.pkl
│
├── static/                       → CSS & generated charts
│   ├── style.css
│   └── dashboard_chart.png
│
├── templates/                    → Frontend pages
│   ├── index.html
│   └── dashboard.html
│
├── train_service_model.ipynb     → Model training notebook
├── predicted_records.csv         → User predictions
├── requirements.txt              → Dependencies
├── Procfile                      → Render deployment file
├── .gitignore                    → Ignored files list
└── README.md                     → Project documentation
```

---

## 🚀 How to Run Locally

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/carServiceTimePrediction.git
cd carServiceTimePrediction
```

### 2️⃣ Create a Virtual Environment
```bash
python -m venv venv
```
Activate it:  
- Windows → `venv\Scripts\activate`  
- macOS/Linux → `source venv/bin/activate`

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Flask App
```bash
python app.py
```

Then open your browser at:  
👉 **http://127.0.0.1:5000**

---

## 🧾 Example Input

| Field | Example |
|--------|----------|
| Car Name | Hyundai Creta |
| Service Type | Full |
| Engine Type | Diesel |
| Condition | Good |
| Spare Parts | Available |
| Mechanic Experience | 5 |
| Workload Queue | 3 |
| Engine Capacity | 1493 |
| Mileage | 42000 |
| Temperature | 95°C |

---

## 📈 Dashboard Features

- ✅ Displays **average service time**
- ✅ Highlights **most common car**
- ✅ Shows **recent 10 predictions**
- ✅ Auto-generated **bar chart**
- ✅ Interactive UI with smooth animations

---

## 🎨 User Interface

- ✨ Clean, professional **form-style layout**
- 🌈 Soft gradient background & glow effects
- 📱 Fully responsive for all devices
- 🧊 Glassmorphism-inspired cards
- ⚡ Smooth animations and hover interactions

---

## 📦 Dependencies

Core Python libraries used:

```
Flask
pandas
numpy
scikit-learn
matplotlib
joblib
gunicorn
```

Recreate them using:
```bash
pip install -r requirements.txt
```

---

## 🌍 Deployment (Render)

1. Push your project to GitHub  
2. Add a **Procfile**:
   ```
   web: gunicorn app:app
   ```
3. Add **gunicorn** to `requirements.txt`
4. Deploy to **https://render.com/**
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app`

Your app will be live on a public URL like:
```
https://carservicetime.onrender.com/
```

---

## 👨‍💻 Author

**Shubham Sanap**  
🎓 Final Year IT Engineering Student  
💡 Passionate about AI, ML, and Web Development  
📍 India  
🔗 GitHub: [https://github.com/YOUR_USERNAME](https://github.com/YOUR_USERNAME)

---

## 🛡 License

This project is licensed under the **MIT License** — free to use, modify, and distribute.

---

## ⭐ Support

If you found this helpful, please **star ⭐ this repo** on GitHub and share it with others learning Flask + ML.
