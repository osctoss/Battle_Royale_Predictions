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
````

---

## How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/your-repo.git
   cd your-repo
   ```

2. Place the dataset Excel file as `Data/test_Battle_Royale.csv.xlsx`.

3. Run the Python script or Jupyter notebook:

   ```bash
   python battle_royale_survival_prediction.py
   ```

---

## Project Breakdown

### Data Cleaning

* Fill missing categorical values with `'Unknown'`.
* Fill missing numerical values with median values.

### False Entry Filtering

* Remove suspicious player data based on specific conditions.

### Target Labeling

* Predict survival time using rule-based heuristics.

### Exploratory Data Analysis

* Visualize distributions of survival time, false entries, and player types.

### Feature Encoding

* One-hot encode categorical variables.
* Label encode target variable.

### Model Training & Evaluation

* Train a Random Forest classifier.
* Evaluate using accuracy, classification report, and confusion matrix.

---

## Sample Visualizations

* Survival Time distribution across players
* False Entries by Activity Status and Tier
* Counts of Legend and Ultimate Legend players
* KD ratio distribution by Survival Time category

---

## Results

* The model achieves an accuracy score of approximately **X%** (replace with actual accuracy after running).
* Classification report and confusion matrix provide insights into prediction performance across survival time classes.

---

## Future Improvements

* Experiment with other classification algorithms like XGBoost or LightGBM.
* Hyperparameter tuning for better accuracy.
* Incorporate more detailed player activity data.
* Deploy the model as a web app or API for real-time prediction.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

Created by **Manish Patel**
Feel free to open issues or contribute!
Find me on [GitHub](https://github.com/osctoss)
