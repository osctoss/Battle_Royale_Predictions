import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

def predict_survival(row):
    if row['KD'] <= 5 and row['ActivityStatus'] in ['Returner', 'Casual'] and row['Tier'] in ['Gold', 'Platinum', 'Diamond']:
        return '<10min'
    elif 3 < row['KD'] <= 10 and row['Tier'] in ['Diamond', 'Crown']:
        return '<20min'
    elif 8 <= row['KD'] <= 15 and row['ActivityStatus'] in ['Regular', 'Casual'] and row['Tier'] in ['Ace', 'Dominator', 'Legend', 'Conqueror']:
        return '<30min'
    else:
        return 'Uncertain'

def clean_data(df):
    cat_cols = ['Server', 'Mode', 'Role', 'Tier', 'ActivityStatus']
    num_cols = ['Experience', 'Level', 'Achievements', 'KD']
    
    df[cat_cols] = df[cat_cols].fillna('Unknown')
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    invalid1 = (df['ActivityStatus'] == 'Returner') & (df['Tier'].isin(['Legend', 'Dominator', 'Conqueror']))
    invalid2 = (df['Experience'] <= 3) & (df['Achievements'] > 5000)
    df['FalseEntry'] = invalid1 | invalid2
    valid_df = df[~df['FalseEntry']].copy()
    valid_df['SurvivalTime'] = valid_df.apply(predict_survival, axis=1)
    
    return valid_df

def train_survival_model(df):
    train_df = df[df['SurvivalTime'] != 'Uncertain'].copy()

    le = LabelEncoder()
    train_df['SurvivalTimeLabel'] = le.fit_transform(train_df['SurvivalTime'])
    
    encoded = pd.get_dummies(train_df[['Server', 'Mode', 'Role', 'Tier', 'ActivityStatus']], drop_first=True)
    X = pd.concat([train_df[['Experience', 'Level', 'Achievements', 'KD']], encoded], axis=1)
    y = train_df['SurvivalTimeLabel']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("\nModel Evaluation:\n")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    return clf, le, X.columns

def predict_survival_p(model, encoder, feature_cols, player_dict):
    player_df = pd.DataFrame([player_dict])
    player_encoded = pd.get_dummies(player_df)

    # Handling missing Values
    for col in feature_cols:
        if col not in player_encoded.columns:
            player_encoded[col] = 0
    player_encoded = player_encoded[feature_cols]

    pred_label = model.predict(player_encoded)[0]
    pred_time = encoder.inverse_transform([pred_label])[0]
    return pred_time

if __name__ == "__main__":
    # Load data
    df = pd.read_excel('Data/test_Battle_Royale.csv.xlsx')
    clean_df = clean_data(df)
    model, encoder, features = train_survival_model(clean_df)

    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/survival_model.pkl')
    joblib.dump(encoder, 'models/label_encoder.pkl')
    joblib.dump(features, 'models/feature_columns.pkl')

    print("Model training completed and saved successfully.")

    print("\nWould you like to test a prediction with custom player input? (y/n)")
    choice = input().strip().lower()

    if choice == 'y':
        player_input = {}
        try:
            player_input['Experience'] = float(input("Experience (How many years?: 1-7 ): "))
            player_input['Level'] = int(input("Level (Min: 30): "))
            player_input['Achievements'] = int(input("Achievement points: "))
            player_input['KD'] = float(input("KD (Kill/Death Ratio): "))
            player_input['Server'] = input("Server (Asia/Middle_East ...etc): ").strip()
            player_input['Mode'] = input("Mode (Solo/Duo/Squad): ").strip()
            player_input['Role'] = input("Role (Sniper/Camper/Assaulter/IGL): ").strip()
            player_input['Tier'] = input("Tier (Diamond, Crown ...etc): ").strip()
            player_input['ActivityStatus'] = input("ActivityStatus (Regular/Casual/Returner): ").strip()

            pred = predict_survival_p(model, encoder, features, player_input)
            print(f"\nPredicted Survival Time: {pred}\n")

        except Exception as e:
            print("Error in input:", e)
