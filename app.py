from flask import Flask, request, render_template, url_for
import joblib
import numpy as np

app = Flask(__name__)

# --- LOAD YOUR SAVED MODEL ---
model = joblib.load(r'c:\Users\LAB-USER-01\Downloads\random_forest_model.pkl')  # Make sure this file exists in your app folder

@app.route('/')
def home():
    """Renders the main page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles the prediction request for predictive models.
    """
    try:
        # --- 1. Extract features from the HTML form ---
        # Replace feature1, feature2, feature3 with your actual input names from index.html
        feature_names = ['feature1', 'feature2', 'feature3']  # <-- Change these to match your form fields
        feature_values = []
        for feat in feature_names:
            value = request.form.get(feat)
            if value is None:
                raise ValueError(f"Missing input for {feat}")
            feature_values.append(float(value))

        # --- 2. Preprocess the features to match your model's input format ---
        final_features = np.array([feature_values])  # Single sample, 2D array for sklearn

        # --- 3. Make a prediction using the loaded model ---
        prediction = model.predict(final_features)

        # --- 4. Format the prediction result for display on the webpage ---
        output = f"Predicted Value: {prediction[0]:.2f}"

    except Exception as e:
        output = f"An error occurred: {e}"

    return render_template('index.html', prediction_text=output)

if __name__ == "__main__":
    app.run(debug=True)