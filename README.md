# Battle Royale Survival Time Prediction

This project analyzes player data from a Battle Royale game to predict the **survival time** category of players based on their in-game statistics using machine learning.

---

## Project Overview

- **Data Source:** Player statistics from a Battle Royale game in an Excel file.
- **Goal:** Predict how long a player is likely to survive in the game (`<10min`, `<20min`, `<30min`).
- **Approach:** 
  - Clean and preprocess data by handling missing values and filtering out false entries.
  - Apply heuristic rules to categorize survival time.
  - Train a Random Forest classifier on relevant features.
  - Evaluate the model using accuracy, classification report, and confusion matrix.
- **Visualization:** Data insights and model results are visualized using seaborn and matplotlib.

---

## Features and Columns Used

- **Categorical Features:** `Role`, `Tier`, `ActivityStatus`, `Server`, `Mode`
- **Numerical Features:** `Experience`, `Level`, `Achievements`, `KD` (Kill-Death ratio)
- **Derived Features:**
  - `FalseEntry`: Identifies suspicious player data based on custom conditions.
  - `LegendPlayer` & `UltimateLegend`: Flags for experienced and returning top-tier players.
  - `SurvivalTime`: Target variable categorized into `<10min`, `<20min`, `<30min` based on heuristics.

---

## Dependencies

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- joblib (for saving/loading models)
- openpyxl (for reading Excel files)

Install dependencies with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib openpyxl
