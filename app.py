# Import necessary modules from Flask
from flask import Flask, render_template, request

# Import numpy and pandas for data handling
import numpy as np
import pandas as pd

# Import the pickle library for model loading
import pickle

# --- START: MODEL INTEGRATION ---
# This block MUST be at the top level (global scope) of your app.py file.

# Define the path to your saved model file
# IMPORTANT: Ensure 'PCA_model.pkl' is in the same directory as app.py.
MODEL_PATH = 'PCA_model.pkl'

# Initialize the model variable globally.
model = None

try:
    # Attempt to load your actual trained model using pickle
    with open(MODEL_PATH, 'rb') as file: # 'rb' for read binary mode
        model = pickle.load(file)
    print(f"SERVER STARTUP: Model loaded successfully from {MODEL_PATH}")
except FileNotFoundError:
    print(f"SERVER STARTUP ERROR: Model file not found at {MODEL_PATH}. Please ensure 'PCA_model.pkl' exists in the same directory as app.py.")
    class FallbackDummyModel:
        def predict(self, df):
            print("WARNING: Using FallbackDummyModel because actual model file was not found.")
            return "Model File Not Found - Cannot Predict."
    model = FallbackDummyModel()
except Exception as e:
    print(f"SERVER STARTUP ERROR: Failed to load model from {MODEL_PATH} due to an unexpected error: {e}")
    class FallbackDummyModel:
        def predict(self, df):
            print(f"WARNING: Using FallbackDummyModel due to model loading error: {e}")
            return f"Model Loading Failed: {e}"
    model = FallbackDummyModel()

# --- END: MODEL INTEGRATION ---


# Initialize the Flask application
# This line MUST come after model loading and before any @app.route decorators.
app = Flask(__name__)

# Define the home route
@app.route('/')
def home():
    """
    Renders the pca.html template when the root URL is accessed.
    """
    return render_template("pca.html")

# Define the predict route
@app.route('/predict', methods=["POST", "GET"])
def predict():
    """
    Handles predictions based on input features submitted from the form.
    """
    if request.method == "POST":
        try:
            # --- START DETAILED INPUT DEBUGGING ---
            print("\n--- PREDICTION REQUEST DEBUGGING ---")
            print(f"Raw request.form content: {request.form}")

            # Define the expected feature names from the HTML form.
            expected_form_fields = [
                'Global_reactive_power',
                'Global_intensity',
                'Sub_metering_1',
                'Sub_metering_2',
                'Sub_metering_3',
                'Voltage'
            ]
            print(f"Expected form fields (from HTML form 'name' attributes): {expected_form_fields}")

            raw_input_values = {}
            for field_name in expected_form_fields:
                value = request.form.get(field_name)
                if value is None:
                    print(f"DEBUG: Form field '{field_name}' was NOT found in the submitted form data. Critical mismatch.")
                    raise ValueError(f"Missing input for field: '{field_name}'. Please ensure all form fields have correct 'name' attributes.")
                raw_input_values[field_name] = value.strip()
                print(f"DEBUG: Collected field '{field_name}' with stripped value: '{raw_input_values[field_name]}'")

            numeric_inputs = {}
            for field_name, stripped_value in raw_input_values.items():
                try:
                    numeric_inputs[field_name] = float(stripped_value)
                except ValueError:
                    print(f"DEBUG: Conversion failed for field '{field_name}' with stripped value '{stripped_value}'.")
                    raise ValueError(f"Non-numeric input for field: '{field_name}' (value: '{stripped_value}').")

            print(f"Successfully converted numeric inputs: {numeric_inputs}")

            # --- Calculate 'sub_metering_4' ---
            # This calculation needs to be based on the numeric inputs.
            # A common way sub_metering_4 is calculated is:
            # (Global_active_power * 1000 / 60) - (Sub_metering_1 + Sub_metering_2 + Sub_metering_3)
            # Since Global_active_power is usually the target, we'll derive it from Global_intensity and Voltage.
            # Global_active_power (kW) = (Global_intensity (Amperes) * Voltage (Volts)) / 1000
            # Then convert to Wh per minute for consistency with sub_metering_X.
            global_active_power_derived_kW = (numeric_inputs['Global_intensity'] * numeric_inputs['Voltage']) / 1000.0
            global_active_power_Wh_per_minute = global_active_power_derived_kW * 1000 / 60

            sub_metering_4_calculated = global_active_power_Wh_per_minute - (numeric_inputs['Sub_metering_1'] + numeric_inputs['Sub_metering_2'] + numeric_inputs['Sub_metering_3'])

            numeric_inputs['sub_metering_4'] = sub_metering_4_calculated
            print(f"Calculated sub_metering_4: {sub_metering_4_calculated}")
            print(f"All features (including calculated): {numeric_inputs}")
            # --- END DETAILED INPUT DEBUGGING ---

            # Check if the model was successfully loaded at startup
            if model is None:
                raise RuntimeError("The machine learning model is not available. Please check server logs for loading errors during startup.")

            # Define the final feature names and their order for the DataFrame.
            # CRITICAL FIX: This order MUST EXACTLY match the input features (X)
            # that your model was trained on, based on the list you provided,
            # and including 'sub_metering_4' which was previously missing.
            # Assuming 'Global_active_power' is the target and not an input feature for prediction.
            features_name_for_df = [
                'Global_reactive_power',
                'Voltage',
                'Global_intensity',
                'Sub_metering_1',
                'Sub_metering_2',
                'Sub_metering_3',
                'sub_metering_4'  # This was explicitly mentioned as missing in a previous error
            ]


            # Create a list of values in the correct order for the DataFrame
            ordered_input_values = [numeric_inputs[name] for name in features_name_for_df]

            # Create a pandas DataFrame
            df = pd.DataFrame([np.array(ordered_input_values)], columns=features_name_for_df)
            print(f"DataFrame prepared for model prediction: \n{df}")
            print(f"DataFrame dtypes: \n{df.dtypes}") # Print data types of DataFrame columns

            # --- START DEBUGGING MODEL PREDICTION ---
            output = "Prediction Error" # Default in case prediction fails
            try:
                print(f"Attempting prediction with model of type: {type(model)}")
                output = model.predict(df)
                print(f"Model prediction raw output: {output}")
            except Exception as model_err:
                print(f"ERROR: An error occurred during model.predict(df): {model_err}")
                raise ValueError(f"Model prediction failed: {model_err}")
            # --- END DEBUGGING MODEL PREDICTION ---

            prediction_display_text = str(output)
            return render_template('result1.html', prediction_text=prediction_display_text)

        except ValueError as ve:
            print(f"ERROR: ValueError caught in predict function: {ve}")
            return render_template('result1.html', prediction_text=f"Invalid input: {ve}. Please ensure all fields are numeric and filled correctly.")
        except RuntimeError as re:
            print(f"ERROR: RuntimeError caught in predict function: {re}")
            return render_template('result1.html', prediction_text=f"Server Error: {re}")
        except Exception as e:
            print(f"ERROR: An unexpected exception occurred during prediction: {e}")
            return render_template('result1.html', prediction_text=f"An internal server error occurred: {e}")
    else:
        return "Please submit the form on the home page to get a prediction."

if __name__ == '__main__':
    app.run(debug=True)
