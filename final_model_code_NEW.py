import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore")

# LOAD DATASETS
def load_datasets():
    print(" Searching for dataset files...")

    base_dir = os.path.dirname(os.path.abspath(__file__))

    possible_paths = [
        base_dir,
        os.path.join(base_dir, "dataset"),
        os.path.join(base_dir, "datasets"),
        os.path.join(base_dir, "data"),
    ]

    diabetes_path = heart_path = bp_path = None

    for path in possible_paths:
        if os.path.exists(os.path.join(path, "diabetes.csv")):
            diabetes_path = os.path.join(path, "diabetes.csv")
        if os.path.exists(os.path.join(path, "heart.csv")):
            heart_path = os.path.join(path, "heart.csv")
        if os.path.exists(os.path.join(path, "hypertension_dataset.csv")):
            bp_path = os.path.join(path, "hypertension_dataset.csv")

    if not all([diabetes_path, heart_path, bp_path]):
        print(" Missing dataset files. Place all 3 CSVs in folder.")
        exit(1)

    print(" Datasets loaded successfully!\n")

    return pd.read_csv(diabetes_path), pd.read_csv(heart_path), pd.read_csv(bp_path)

# TRAIN SINGLE MODEL
def train_model(df, target_col):
    df = df.copy()

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    df = df.fillna(df.median(numeric_only=True))

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(
        random_state=42,
        n_estimators=10,
        max_depth=4,
        max_features=4
    )

    model.fit(X_train_scaled, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test_scaled))

    return model, scaler, accuracy, X_test, y_test

# TRAIN ALL MODELS
def train_all_models(diabetes, heart, bp):
    print(" Training models...\n")

    dia_model, dia_scaler, acc_d, X_dia_test, y_dia_test = train_model(diabetes, "Outcome")
    heart_model, heart_scaler, acc_h, X_heart_test, y_heart_test = train_model(heart, "target")
    bp_model, bp_scaler, acc_bp, X_bp_test, y_bp_test = train_model(bp, "Hypertension")

    print(f" Diabetes Model Accuracy: {acc_d*100:.2f}%")
    print(f" Heart Disease Model Accuracy: {acc_h*100:.2f}%")
    print(f" Hypertension Model Accuracy: {acc_bp*100:.2f}%\n")

    return {
        "diabetes": (dia_model, dia_scaler, acc_d, X_dia_test, y_dia_test),
        "heart": (heart_model, heart_scaler, acc_h, X_heart_test, y_heart_test),
        "bp": (bp_model, bp_scaler, acc_bp, X_bp_test, y_bp_test)
    }


# USER INPUT
def get_user_input():
    print("\nEnter your health details:\n")

    print("\n--- Diabetes Input ---")
    diabetes_data = [
        float(input("Number of Pregnancies: ")),
        float(input("Glucose Level (mg/dL): ")),
        float(input("Blood Pressure (mm Hg): ")),
        float(input("Skin Thickness (mm): ")),
        float(input("Insulin Level: ")),
        float(input("BMI (IU/ml): ")),
        float(input("Diabetes Pedigree Function: ")),
        float(input("Age: "))
    ]

    print("\n--- Heart Disease Input ---")
    heart_data = [
        float(input("Age: ")),
        float(input("Sex (1=Male, 0=Female): ")),
        float(input("Chest Pain Type (0–3): ")),
        float(input("Resting BP: ")),
        float(input("Cholesterol: ")),
        float(input("Fasting Blood Sugar >120 (1=Yes,0=No): ")),
        float(input("Resting ECG (0–2): ")),
        float(input("Max Heart Rate: ")),
        float(input("Exercise Induced Angina (1=Yes, 0=No): ")),
        float(input("ST Depression(Oldpeak): ")),
        float(input("Slope (0–2): ")),
        float(input("Major Vessels Colored (0–3): ")),
        float(input("Thalassemia (0–3): "))
    ]

    print("\n--- Hypertension Input ---")
    bp_data = [
        float(input("Age: ")),
        float(input("BMI: ")),
        float(input("Average Sleep Hours: ")),
        float(input("Stress Level (1–10): ")),
        float(input("Salt Intake (1–10): "))
    ]

    return diabetes_data, heart_data, bp_data

# PREDICT RISKS
def predict_risks(models, dia_in, heart_in, bp_in):

    results = {}

    # Diabetes
    m, s, _, _, _ = models["diabetes"]
    results["Diabetes"] = round(m.predict_proba(s.transform([dia_in]))[0][1] * 100, 1)

    # Heart
    m, s, _, _, _ = models["heart"]
    results["Heart Disease"] = round(m.predict_proba(s.transform([heart_in]))[0][1] * 100, 1)

    # Hypertension
    m, s, _, _, _ = models["bp"]
    bp_in = bp_in + [0] * (s.n_features_in_ - len(bp_in))
    results["Hypertension"] = round(m.predict_proba(s.transform([bp_in]))[0][1] * 100, 1)

    return results

# RECOMMENDATIONS
def generate_recommendations(results):
    rec = {}

    for disease, risk in results.items():
        if risk > 60:
            rec[disease] = [
                "⚠️ HIGH RISK — Immediate attention needed:",
                "• Consult a doctor as soon as possible.",
                "• Avoid junk, oily, sugary foods.",
                "• Exercise daily for 30 minutes.",
                "• Reduce stress and monitor regularly."
            ]
        elif risk > 30:
            rec[disease] = [
                "🟡 MODERATE RISK — Be cautious:",
                "• Improve your diet and lifestyle.",
                "• Increase physical activity.",
                "• Monitor symptoms regularly."
            ]
        else:
            rec[disease] = [
                "🟢 LOW RISK — Maintain healthy habits:",
                "• Continue regular exercise.",
                "• Eat a balanced diet.",
                "• Yearly health check-ups recommended."
            ]
    return rec

# BAR CHART
def generate_report(results, save_dir="results"):
    import datetime
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    diseases = list(results.keys())
    probs = list(results.values())
    colors = ['#e74c3c' if p > 60 else '#f1c40f' if p > 30 else '#2ecc71' for p in probs]

    plt.figure(figsize=(8, 5))
    plt.bar(diseases, probs, color=colors, edgecolor='black')
    plt.ylim(0, 100)
    plt.xlabel("Disease")
    plt.ylabel("Risk Probability (%)")
    plt.title("Predicted Disease Risk")
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"bar_chart_{ts}.png"))
    plt.show()

# COMBINED SCATTER PLOT
def plot_combined_scatter(models, save_dir="results"):
    import datetime
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    dia_model, dia_scaler, _, X_dia_test, y_dia_test = models["diabetes"]
    heart_model, heart_scaler, _, X_heart_test, y_heart_test = models["heart"]
    bp_model, bp_scaler, _, X_bp_test, y_bp_test = models["bp"]

    dia_pred = dia_model.predict_proba(dia_scaler.transform(X_dia_test))[:, 1]
    heart_pred = heart_model.predict_proba(heart_scaler.transform(X_heart_test))[:, 1]
    bp_pred = bp_model.predict_proba(bp_scaler.transform(X_bp_test))[:, 1]

    plt.figure(figsize=(9, 7))
    plt.scatter(y_dia_test, dia_pred, color="blue", alpha=0.6, label="Diabetes")
    plt.scatter(y_heart_test, heart_pred, color="red", alpha=0.6, label="Heart Disease")
    plt.scatter(y_bp_test, bp_pred, color="green", alpha=0.6, label="Hypertension")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title("Actual vs Predicted Probabilities")
    plt.xlabel("Actual (0 or 1)")
    plt.ylabel("Predicted Probability")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.savefig(os.path.join(save_dir, f"scatter_plot_{ts}.png"))
    plt.show()

# MAIN
def main():
    print("\n=== Multi-Disease Prediction System ===\n")

    diabetes, heart, bp = load_datasets()
    models = train_all_models(diabetes, heart, bp)

    dia_in, heart_in, bp_in = get_user_input()
    results = predict_risks(models, dia_in, heart_in, bp_in)
    recs = generate_recommendations(results)

    print("\n📊 Prediction Results:")
    for d, v in results.items():
        print(f"{d:15} : {v:.1f}%")

    print("\n🩺 Recommendations:")
    for d, adv in recs.items():
        print(f"\n{d}:")
        for a in adv:
            print("  " + a)

    generate_report(results)
    plot_combined_scatter(models)

    print("\n All charts saved successfully!\n")


if __name__ == "__main__":
    main()



