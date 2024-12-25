# NPCI MLOps Playground Challenge - 2

Dockerizing Bike Rental Count Prediction model.

---

## Objective
Participants are required to:
1. Create an `app.py` file that leverages [Gradio](https://gradio.app/) to create a user-friendly UI for a pre-trained bike rent count prediction model.
2. Dockerize the application for easy deployment and portability.

---

## Repository Structure

```
├── random_forest_model.pkl         # Pre-trained bike rent prediction model (provided)
├── README.md         # Instructions and details (this file)
└── requirements.txt  # Dependencies for the project
```

---

## Instructions

### Step 1: Clone the Repository


### Step 2: Update Dependencies in requirements.txt


### Step 3: Implement the Gradio UI (`app.py`)
1. Create an `app.py` file in the root directory.
2. In `app.py`, load the `model.pkl` file and implement the prediction logic.
3. Use Gradio to build a UI for users to input the necessary features for bike rent prediction.
4. Example workflow for `app.py`:
   - Load the `model.pkl`.
   - Define a prediction function.
   - Use Gradio to create the UI.



### Step 4: Dockerize the Application
 Create a `Dockerfile` in the root directory.


### Step 5: Build and Run the Docker Container
1. Build the Docker image

2. Run the Docker container

