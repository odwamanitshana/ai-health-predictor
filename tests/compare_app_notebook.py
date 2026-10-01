import os

import joblib
import numpy as np


def load_random_forest():
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    rf_path = os.path.join(models_dir, "random_forest.pkl")
    return joblib.load(rf_path)


def run_examples(model):
    examples = [
        ("Low risk (typical)", [0, 85, 66, 29, 0, 26.6, 0.351, 31]),
        ("Medium risk", [2, 120, 70, 20, 79, 28.0, 0.5, 45]),
        ("High glucose/age", [4, 180, 85, 25, 200, 35.0, 1.2, 55]),
    ]

    results = []
    for name, values in examples:
        features = np.array(values).reshape(1, -1)
        prob = model.predict_proba(features)[0, 1]
        pred = int(model.predict(features)[0])
        results.append((name, prob, pred))

    return results


def main():
    model = load_random_forest()
    results = run_examples(model)

    print("Example predictions from Random Forest:")
    for name, prob, pred in results:
        print(f"{name}: probability={prob:.4f}, class={pred}")


if __name__ == "__main__":
    main()
