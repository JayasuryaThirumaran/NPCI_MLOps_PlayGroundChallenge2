# Bike Rent Prediction Model - Dockerized Gradio UI

Welcome to the **Bike Rent Prediction Model** repository! This project is designed as a hands-on challenge for participants to dockerize a machine learning model and create an interactive user interface using Gradio.

---

## Objective
Participants are required to:
1. Create an `app.py` file that leverages [Gradio](https://gradio.app/) to create a user-friendly UI for a pre-trained bike rent prediction model.
2. Dockerize the application for easy deployment and portability.

---

## Repository Structure

```
├── model.pkl         # Pre-trained bike rent prediction model (provided)
├── README.md         # Instructions and details (this file)
└── requirements.txt  # Dependencies for the project
```

---

## Instructions

### Step 1: Clone the Repository
```bash
git clone <repo-url>
cd <repo-name>
```

### Step 2: Install Dependencies
To ensure the app runs locally (before dockerizing), install the required Python libraries listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

### Step 3: Implement the Gradio UI (`app.py`)
1. Create an `app.py` file in the root directory.
2. In `app.py`, load the `model.pkl` file and implement the prediction logic.
3. Use Gradio to build a UI for users to input the necessary features for bike rent prediction.
4. Example workflow for `app.py`:
   - Load the `model.pkl`.
   - Define a prediction function.
   - Use Gradio to create the UI.

### Example Code Skeleton for `app.py`:
```python
import gradio as gr
import pickle

# Load the pre-trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

def predict(features):
    """
    Make predictions using the loaded model.
    Args:
        features: List of feature inputs from the UI.
    Returns:
        Predicted bike rental count.
    """
    prediction = model.predict([features])
    return prediction[0]

# Create a Gradio interface
inputs = [gr.inputs.Number(label="Feature 1"),
          gr.inputs.Number(label="Feature 2"),
          gr.inputs.Number(label="Feature 3")]  # Add all required features

outputs = gr.outputs.Textbox(label="Predicted Bike Rentals")

gr.Interface(fn=predict, inputs=inputs, outputs=outputs, title="Bike Rent Prediction").launch()
```

### Step 4: Dockerize the Application
1. Create a `Dockerfile` in the root directory.
2. Use the following template for your `Dockerfile`:

```dockerfile
# Use a lightweight Python image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy application files to the container
COPY . /app

# Install dependencies
RUN pip install -r requirements.txt

# Expose the port used by Gradio (default is 7860)
EXPOSE 7860

# Run the Gradio app
CMD ["python", "app.py"]
```

### Step 5: Build and Run the Docker Container
1. Build the Docker image:
```bash
docker build -t bike-rent-prediction-app .
```

2. Run the Docker container:
```bash
docker run -p 7860:7860 bike-rent-prediction-app
```

3. Access the application in your browser at: [http://localhost:7860](http://localhost:7860)

---

## Deliverables
Participants are expected to submit:
1. `app.py` file with the Gradio UI implementation.
2. A `Dockerfile` to dockerize the application.
3. Optionally, screenshots or a short video showing the working UI and Docker container.

---

## Tips
- Test your application locally before dockerizing.
- Make sure the Gradio app runs on `0.0.0.0` inside the container to allow external access.
- Use a `.dockerignore` file to exclude unnecessary files when building the Docker image.

---

## Resources
- [Gradio Documentation](https://gradio.app/docs/)
- [Docker Documentation](https://docs.docker.com/)
- [Python Pickle Module](https://docs.python.org/3/library/pickle.html)

---

## Contact
For any issues or questions, please reach out to the repository maintainers.
