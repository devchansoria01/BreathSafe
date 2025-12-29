# Breath-Safe: The Virtual Air Quality Sensor Network 🌍💨

> **"Seeing the Invisible. Breathing the Future."**

---

## 🛠️ Tech Stack
* **Language:** Python
* **Data Handling:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn, random FOrest , lstm
* **APIs & SDKs:** `sentinelsat`, OpenWeatherMap API, NASA FIRMS
* **Visualization:** Matplotlib, Seaborn, Folium (for maps)

## 💻 Installation & Usage

1.  **Clone the Repo**
    ```bash
    git clone [https://github.com/devchansoria01/BreathSafe.git)
    cd breathsafe
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

4.  **Run the script**
    ```bash
    python lstm.py
    ```

    ## 📊 Random Forest Model Performance

| Metric | Value |
| :--- | :--- |
| **Overall Accuracy** | 92.27% |
| **Overall RMSE** | 68.51 AQI |

   ## 📊 LSTM Model Performance

| Metric | Value |
| :--- | :--- |
| **Overall Accuracy** | 84.34% |
| **Overall RMSE** | 82.21 AQI |

## 👥 Team
* [Your Name] - Data Scientist
* [Teammate Name] - Backend Developer
* [Teammate Name] - Frontend/Visualization

---
