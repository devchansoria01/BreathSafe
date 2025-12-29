# VayuSense: The Virtual Air Quality Sensor Network 🌍💨

![Status](https://img.shields.io/badge/Status-Prototype-blue)
![Python](https://img.shields.io/badge/Made%20with-Python-3776AB?logo=python&logoColor=white)
![ML](https://img.shields.io/badge/AI-XGBoost-orange)

> **"Seeing the Invisible. Breathing the Future."**

---

## 📖 Overview
**VayuSense** is a hyper-local air quality monitoring system that fills the data gaps left by traditional hardware sensors. Instead of relying solely on sparse physical stations (which are expensive and limited to specific locations), VayuSense uses a **Virtual Sensor Architecture**.

By fusing **Sentinel-5P Satellite data**, **Real-time Traffic congestion levels**, and **Meteorological patterns**, our AI model estimates ground-level PM2.5 and AQI for any coordinate in a city—effectively turning every street corner into a monitoring station.

## 💡 The Problem
* **Limited Coverage:** In a city like New Delhi, reliable CPCB sensors are few and far between. A sensor 5km away does not reflect the pollution on your specific street.
* **High Cost:** Reference-grade monitoring stations cost ~$150,000, making it impossible to cover an entire city densely.
* **Blind Spots:** Rural areas and suburbs often have zero data, leaving citizens unaware of the air they breathe.

## 🚀 Our Solution: The Virtual Sensor
VayuSense treats Air Quality prediction not just as a *time-series* problem, but as a **Spatial Data Fusion** problem.

### Key Features
1.  **Hyper-Local Heatmaps:** Generates AQI predictions for areas without sensors using satellite overlays.
2.  **Stubble Burning Integration:** Specifically models the impact of seasonal farm fires (using NASA FIRMS data) on urban air quality.
3.  **Route Optimization (Concept):** Suggests "Cleaner Commutes" rather than just the fastest route.
4.  **Cost-Effective:** Zero new hardware required. scalable to any city with satellite coverage.

## ⚙️ How It Works (Architecture)
The system follows a **Multi-Modal Data Fusion** approach:

1.  **Data Ingestion Layer:**
    * **Satellite:** Sentinel-5P TROPOMI (NO2 & Aerosol Optical Depth) via Google Earth Engine.
    * **Ground:** Historical PM2.5 data from CPCB/OpenAQ (Ground Truth).
    * **Context:** Weather (OpenWeatherMap) + Fire Spots (NASA FIRMS) + Traffic Density.

2.  **Processing Layer:**
    * Spatio-temporal alignment (matching daily satellite flyovers with hourly ground data).
    * Feature Engineering: Lag variables, Wind Direction vectors, Boundary Layer Height approximations.

3.  **AI Engine:**
    * **Model:** Gradient Boosting Regressor (XGBoost/LightGBM).
    * **Logic:** Learns the correlation between "Column Density" (Satellite) and "Ground Concentration" (Sensors), corrected by weather.

4.  **Output Layer:**
    * Interactive Dashboard showing Real vs. Predicted AQI.
    * Interpolated Heatmap for the target city.

## 🛠️ Tech Stack
* **Language:** Python 3.9+
* **Data Handling:** Pandas, NumPy, GeoPandas
* **Machine Learning:** Scikit-Learn, XGBoost
* **APIs & SDKs:** `sentinelsat`, OpenWeatherMap API, NASA FIRMS
* **Visualization:** Matplotlib, Seaborn, Folium (for maps)

## 📊 Feasibility Analysis
* **Scientific Validity:** Leverages established correlations between AOD (Satellite) and PM2.5, corrected for humidity and boundary layer height.
* **Reliability:** Implements a "Fallback Mode"—if satellite data is blocked by clouds, the model reverts to a robust meteorological forecast.
* **Scalability:** The model trained on Delhi can be fine-tuned for Mumbai or Bangalore with minimal retraining.

## 💻 Installation & Usage

1.  **Clone the Repo**
    ```bash
    git clone [https://github.com/yourusername/vayusense.git](https://github.com/yourusername/vayusense.git)
    cd vayusense
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set Up API Keys**
    * Create a `.env` file and add your keys:
    ```env
    OPENWEATHER_API_KEY=your_key_here
    NASA_FIRMS_KEY=your_key_here
    ```

4.  **Run the Data Fetcher**
    ```bash
    python src/data_ingestion.py
    ```

5.  **Train the Model**
    ```bash
    python src/train_model.py
    ```

6.  **Launch Dashboard**
    ```bash
    streamlit run app.py
    ```

## 🔮 Future Scope
* **Computer Vision Module:** Allow users to upload skyline photos to estimate visibility/AQI using CNNs.
* **IoT Calibration:** Use the model to auto-calibrate low-cost sensors in the network.
* **Public API:** Release an API for ride-sharing apps to integrate "Clean Route" options.

## 👥 Team
* [Your Name] - Data Scientist
* [Teammate Name] - Backend Developer
* [Teammate Name] - Frontend/Visualization

---

*Built with ❤️ for cleaner air.*
