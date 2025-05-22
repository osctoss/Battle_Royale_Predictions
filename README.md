# 🛡️ Battle Royale Survival Predictor 🧠🎮

**Project Type:** Machine Learning, Data Analysis, Game Analytics
**Tech Stack:** Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Jupyter Notebook

---

## 🎯 TL;DR

As both a gamer and a coder, I always wondered: *"Can I predict how long a player will survive in a Battle Royale match just from their stats?"*
So, I trained a **Machine Learning model** to do exactly that, and added rich visualizations to analyze how experience, achievements, and region affect survival time.

> This project has two sides:

* 📊 `BRPrediction.ipynb` – for exploratory data analysis and interactive graphs
* 🧠 `survival_model_training.py` – for model training and evaluation

---

## 🧩 Problem We’re Solving

Survival time in Battle Royale games is not just luck — it's tied to experience, skill (KD ratio), region, and player behavior.
However, in-game data often has **inconsistencies**, **false entries**, and **unstructured formats**. This project solves that by:

* Cleaning and structuring raw BR player data
* Predicting survival time using machine learning
* Visualizing player behavior across servers, tiers, and activity levels

---

## 🎮 Dataset Overview

We used a `.csv.xlsx` file containing player stats with features like:

* `KD`, `Level`, `Experience`, `Achievements`
* `Tier`, `Mode`, `Server`, `Role`, `ActivityStatus`

Plus, we implemented logic to flag fake entries (e.g., low experience with high achievements or inconsistent tier vs activity status).

---

## 🚧 Project Structure

```
📁 BattleRoyaleSurvivalPredictor/
│
├── BRPrediction.ipynb            # Main notebook with visuals and insights
├── src/survival_model_training.py    # Script to clean, process and train the ML model
├── Data/test_Battle_Royale.csv.xlsx   # Raw dataset
├── models/
│   ├── survival_model.pkl        # Trained Random Forest model
│   ├── label_encoder.pkl         # Label encoder for class labels
│   └── feature_columns.pkl       # List of feature columns used for predictions
└── README.md                     # You're reading it right now!
```

---

## ⚙️ Features & Visuals

### 📌 Model-Based Predictions

* Uses a **Random Forest Classifier** to classify players into:

  * `<10min`, `<20min`, `<30min`, or `Uncertain` categories
* Rule-based logic to label training samples
* Encoded, cleaned, and split before training

### 📈 Visualizations in `BRPrediction.ipynb`

* **Average KD & Survival Time per Server**
* **Player distribution per region using interactive dots**
* **Achievements vs Level per Region (Bubble Chart)**
* **Tier-wise KD Ratio heatmaps**
* Graphs designed to look **premium and gamer-friendly** with rich color palettes

---

## 🔮 How to Run

### Option 1: Jupyter Notebook (Recommended for Analysis)

```bash
jupyter notebook BRPrediction.ipynb
```

You’ll be able to:

* See step-by-step visuals
* Understand how stats differ per server, tier, and more
* Run predictions on your custom player stats

### Option 2: Python Script (For Model Training)

```bash
python survival_model_training.py
```

This will:

* Load and clean data
* Apply rule-based survival time labels
* Train the model
* Save the model and related files under `models/`

---

## 🎯 Why This Project is Cool

* Combines gaming instincts with data science
* Helps understand player psychology based on stats
* Can be extended into a **real-time prediction app** for streamers or coaches

---

## 💡 Future Plans

* Build a web-based predictor using Streamlit or Flask
* Let players upload their stats and get survival predictions + gameplay tips
* Add clustering to group similar players and strategies

---

## 👨‍💻 Author

**Manish Patel (aka Osctoss)**
📫 [osctoss.net@gmail.com](mailto:osctoss.net@gmail.com)
🌐 [LinkedIn](https://linkedin.com/in/manish-patel-osctoss) | [GitHub](https://github.com/osctoss)

---

## 🧠 Pro Tip for Coders

> When data speaks, strategy wins. Whether it's zones or zeroes, let your model guide your shots.

---
