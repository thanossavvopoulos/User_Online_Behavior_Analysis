# 📊 User Behavior Analysis & Churn Prediction

This Python project performs **data cleaning, user profiling, segmentation, visualization, and churn prediction** using a user behavior dataset.

It combines **data analysis** and **machine learning** techniques to extract insights and identify users at risk of leaving.

---

## 🚀 Features

* ✅ Data loading & cleaning (handling missing values and duplicates)
* ✅ User scoring system based on behavior
* ✅ User profiling (Active / Moderate / Inactive)
* ✅ User segmentation (VIP / Active / Standard)
* ✅ Aggregated statistics with visualizations
* ✅ Churn prediction using Logistic Regression
* ✅ Visualization of churn distribution

---

## 📂 Project Structure

```
User_Online_Behavior_Analysis/
│── main.py
│── ex4_user_behavior_dataset.csv
│── README.md
```

---

## 📦 Requirements

Install the required dependencies:

```bash
pip install pandas matplotlib seaborn scikit-learn
```

---

## ⚙️ How It Works

### 1. Data Cleaning

* Removes duplicates
* Drops missing values

### 2. User Scoring System

Each user gets a score based on:

```
Score =
(2 × sessions)
+ (avg_session_time × sessions)
+ (5 × purchases)
- (1.5 × last_active_days)
```

### 3. User Profiling

| Score Range | Profile                |
| ----------- | ---------------------- |
| ≥ 500       | Highly Active User     |
| ≥ 200       | Moderately Active User |
| < 200       | Inactive User          |

---

### 4. User Segmentation

Users are grouped into:

* **VIP** → purchases > 10
* **Active** → sessions > 50
* **Standard** → all others

---

### 5. Aggregation & Visualization

* GroupBy statistics per user category
* Bar chart showing user distribution

---

### 6. Churn Prediction

* Model: **Logistic Regression**
* Features:

  * sessions
  * avg_session_time
  * purchases
  * last_active_days

Outputs:

* Model accuracy
* Churn probability per user
* High-risk users (> 0.7 probability)
* Pie chart visualization

---

## ▶️ How to Run

```bash
python main.py
```

---

## 📊 Example Outputs

* User profile distribution
* Aggregated statistics table
* Bar chart (user groups)
* Pie chart (churn vs active)
* Model accuracy score
* Count of high-risk users

---

## 🧠 Technologies Used

* Python
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn

---

## 👨‍💻 Authors

- **Load and Clean Data:** Metallinos Konstantinos (@thebigice), Karsanidou Maria Antiopi (@antiopem)
- **User Profiling and Scoring:** Desyllas Konstantinos (@KostasDes06), Tsea Anastasia (@Gemmiee)
- **User Grouping, Aggregation and Visualization:** Metallinos Konstantinos (@thebigice), Karsanidou Maria Antiopi (@antiopem)
- **Churn Prediction:** Arapisonoglou Anastasios (@AnastasisARP), Savvopoulos Athanasios (@thanossavvopoulos)

