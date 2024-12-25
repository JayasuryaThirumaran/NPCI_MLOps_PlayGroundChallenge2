import gradio as gr
import pandas as pd
import joblib  # or pickle
import numpy as np

# Load the trained model (replace with your model file)
trained_model = joblib.load('random_forest_model.pkl')  # or pickle.load(open('bike_rental_model.pkl', 'rb'))

# UI input components
season_input = gr.Number(label="Season", value=2)
hr_input = gr.Number(label="Hour", value=14)
holiday_input = gr.Radio([0, 1], label="Holiday", value=0)
workingday_input = gr.Radio([0, 1], label="Working day", value=1)
weathersit_input = gr.Number(label="Weather situation", value=1)
temp_input = gr.Number(label="Temperature (Celsius)", value=1)
atemp_input = gr.Number(label="Apparent temperature (Celsius)", value=8.92)
hum_input = gr.Number(label="Humidity (%)", value=93.0)
windspeed_input = gr.Number(label="Windspeed (km/h)", value=27.9993)
yr_input = gr.Radio([0, 1], label="Year (0=2011, 1=2012)", value=1)
mnth_input = gr.Number(label="Month", value=5)
weekday_fri_input = gr.Radio([0, 1], label="Is Friday", value=0)
weekday_mon_input = gr.Radio([0, 1], label="Is Monday", value=0)
weekday_sat_input = gr.Radio([0, 1], label="Is Saturday", value=0)
weekday_sun_input = gr.Radio([0, 1], label="Is Sunday", value=1)
weekday_thu_input = gr.Radio([0, 1], label="Is Thursday", value=0)
weekday_tue_input = gr.Radio([0, 1], label="Is Tuesday", value=0)
weekday_wed_input = gr.Radio([0, 1], label="Is Wednesday", value=0)

# Output component for prediction
output_label = gr.Textbox(label="Predicted Bike Rentals")

# Create a function to transform the inputs into a DataFrame and predict
def predict_bike_rentals(season, hr, holiday, workingday, weathersit, temp, atemp, hum, windspeed, yr, mnth, 
                         weekday_fri, weekday_mon, weekday_sat, weekday_sun, weekday_thu, weekday_tue, weekday_wed):
    # Convert inputs into a dataframe to match the model's expected format
    input_data = pd.DataFrame({
        'season': [season],
        'hr': [hr],
        'holiday': [holiday],
        'workingday': [workingday],
        'weathersit': [weathersit],
        'temp': [temp],
        'atemp': [atemp],
        'hum': [hum],
        'windspeed': [windspeed],
        'yr': [yr],
        'mnth': [mnth],
        'weekday_Fri': [weekday_fri],
        'weekday_Mon': [weekday_mon],
        'weekday_Sat': [weekday_sat],
        'weekday_Sun': [weekday_sun],
        'weekday_Thu': [weekday_thu],
        'weekday_Tue': [weekday_tue],
        'weekday_Wed': [weekday_wed]
    })
    
    # Make the prediction using the trained model
    prediction = trained_model.predict(input_data)
    
    # Return the prediction result
    return f"Predicted Bike Rentals: {prediction[0]}"

# Define the Gradio interface
iface = gr.Interface(
    fn=predict_bike_rentals,
    inputs=[
        season_input, hr_input, holiday_input, workingday_input, weathersit_input, temp_input, atemp_input, 
        hum_input, windspeed_input, yr_input, mnth_input, weekday_fri_input, weekday_mon_input, weekday_sat_input, 
        weekday_sun_input, weekday_thu_input, weekday_tue_input, weekday_wed_input
    ],
    outputs=[output_label],
    title="Bike Rental Prediction",
    description="Input features to predict the number of bike rentals for the given conditions.",
)

# Launch the Gradio app
iface.launch(server_name="0.0.0.0", server_port=7860)
