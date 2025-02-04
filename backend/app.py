import os
import io
import base64
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from flask import Flask, render_template, request

# Configure Matplotlib backend
plt.switch_backend('Agg')

# Flask app configuration
app = Flask(
    __name__,
    template_folder="../frontend/templates",
    static_folder="../frontend/static"
)

# Constants
MODEL_PATH = 'model/xgb_model.pkl'
SCALER_PATH = 'model/scaler_xgb.pkl'
QUESTIONS_PATH = 'questions.txt'
RESULTS_FILE = 'personality_results.xlsx'

# Load the XGBoost models and scaler
xgboost_models = joblib.load(MODEL_PATH)  # Load the dictionary of models
scaler = joblib.load(SCALER_PATH)

# Ensure results file exists
if not os.path.exists(RESULTS_FILE):
    pd.DataFrame(columns=[
        'Name', 'Age', 'Gender', 'Hand', 'Openness', 'Conscientiousness',
        'Extraversion', 'Agreeableness', 'Neuroticism', 'Overall Personality'
    ]).to_excel(RESULTS_FILE, index=False)

# Utility Functions
def fetch_questions():
    """Fetches questions from the text file."""
    with open(QUESTIONS_PATH, 'r') as file:
        return [line.strip() for line in file.readlines()]

def create_pie_chart(result):
    """
    Generates a donut chart with the exact percentages for each trait.
    Returns its Base64 URL.
    """
    traits = list(result.keys())  # Trait names
    values = list(result.values())  # Scaled percentages (0–100)

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['#8348f9', '#c2a3ff', '#b348ff', '#a61eff', '#9148ff']

    # Generate the pie chart
    wedges, texts, autotexts = ax.pie(
        values,
        labels=traits,
        colors=colors,
        autopct=lambda p: f'{p:.1f}%',  # Show percentage on the chart
        startangle=90,
        pctdistance=0.80,
        radius=0.9,
        textprops={'fontsize': 10}
    )

    # Add a white circle to create a donut chart
    centre_circle = plt.Circle((0, 0), 0.55, fc='white')
    fig.gca().add_artist(centre_circle)

    # Finalize chart appearance
    ax.axis('equal')  # Ensure pie chart is circular
    ax.set_title("Big Five Personality Traits", fontsize=12, color="purple", pad=25)

    # Save chart as Base64 image
    img = io.BytesIO()
    plt.savefig(img, format='png', transparent=True, bbox_inches='tight')
    img.seek(0)
    chart_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close(fig)

    return chart_url

def calculate_personality(result):
    """Assigns a personality type name and description based on the average score."""
    avg_score = np.mean(list(result.values()))

    if avg_score >= 90:
        name = "Bold & Visionary Thinker"
        description = "🌟 Trailblazer 🚀 - You inspire others with your confidence and ability to take on challenges fearlessly."
    elif avg_score >= 75:
        name = "Driven & Strong Personality"
        description = "💪 Go-Getter - Highly motivated and determined, you push past obstacles to achieve success."
    elif avg_score >= 60:
        name = "Confident & Inspiring"
        description = "🌟 Charismatic Leader - A natural leader who uplifts those around you with ease."
    elif avg_score >= 45:
        name = "Well-Rounded & Adaptable"
        description = "🎯 Balanced & Grounded - Able to adapt to different situations and bring stability wherever you go."
    elif avg_score >= 30:
        name = "Reflective & Thoughtful"
        description = "🤔 Deep Thinker - Introspective and thoughtful, you often analyze things deeply."
    else:
        name = "Calm & Peaceful Personality"
        description = "🌿 Gentle Soul - Empathetic and kind, you are a comforting presence to those around you."
    
    return name, description

def append_results_to_file(new_row):
    """Appends user results to the Excel file."""
    try:
        results_df = pd.read_excel(RESULTS_FILE)
        results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
        results_df.to_excel(RESULTS_FILE, index=False)
    except Exception as e:
        raise RuntimeError(f"Error saving results: {e}")

# Routes
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/quiz', methods=['POST'])
def quiz():
    user_details = {
        'name': request.form['name'],
        'age': request.form['age'],
        'gender': request.form['gender'],
        'hand': request.form['hand']
    }
    questions = fetch_questions()
    return render_template('quiz.html', **user_details, questions=questions)

@app.route('/result', methods=['POST'])
def result():
    try:
        # Fetch user details
        user_details = {
            'name': request.form['name'],
            'age': request.form['age'],
            'gender': request.form['gender'],
            'hand': request.form['hand']
        }

        # Collect responses
        responses = [int(request.form[f'q{i+1}']) for i in range(50)]
        extra_features = [
            int(user_details['age']),
            1 if user_details['gender'] == 'Male' else 0,
            1 if user_details['hand'] == 'Right' else 0 if user_details['hand'] == 'Left' else 2
        ]

        # Preprocess and predict
        responses_with_extra = responses + extra_features
        responses_scaled = scaler.transform([responses_with_extra])
        predictions = {}
        for trait, model in xgboost_models.items():
            # Predict raw values and scale them from 0-5 to 0-100
            raw_score = model.predict(responses_scaled)[0]
            scaled_score = round(raw_score * 20, 2)  # Scale and round to 2 decimal places
            predictions[trait] = scaled_score

        # Analyze results
        overall_name, overall_description = calculate_personality(predictions)

        # Generate chart and save results
        chart_url = create_pie_chart(predictions)
        
        new_row = {
            'Name': user_details['name'],
            'Age': user_details['age'],
            'Gender': user_details['gender'],
            'Hand': user_details['hand'],
            **predictions,
            'Overall Personality': overall_name
        }
        append_results_to_file(new_row)

        return render_template(
            'result.html',
            **user_details,
            result=predictions,
            overall_name=overall_name,
            overall_description=overall_description,
            chart_url=chart_url
        )

    except Exception as e:
        return f"An error occurred: {e}"


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, port=5000)
