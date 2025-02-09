import os
import io
import base64
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from flask import Flask, render_template, request

# Configure Matplotlib
plt.switch_backend('Agg')


app = Flask(
    __name__,
    template_folder="../frontend/templates",
    static_folder="../frontend/static"
)


MODEL_PATH = './model/xgb_model.pkl'
SCALER_PATH = 'model/scaler_xgb.pkl'
QUESTIONS_PATH = 'questions.txt'
RESULTS_FILE = 'personality_results.xlsx'

xgboost_models = joblib.load(MODEL_PATH)  
scaler = joblib.load(SCALER_PATH)


if not os.path.exists(RESULTS_FILE):
    pd.DataFrame(columns=[
        'Name', 'Age', 'Gender', 'Hand', 'Openness', 'Conscientiousness',
        'Extraversion', 'Agreeableness', 'Neuroticism', 'Overall Personality'
    ]).to_excel(RESULTS_FILE, index=False)

# Fetches questions from the text file
def fetch_questions():
    with open(QUESTIONS_PATH, 'r') as file:
        return [line.strip() for line in file.readlines()]

#Donut chart
def create_pie_chart(result):
    
    traits = list(result.keys())  
    values = list(result.values()) 


    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['#8348f9', '#c2a3ff', '#b348ff', '#a61eff', '#9148ff']

    wedges, texts, autotexts = ax.pie(
        values,
        labels=traits,
        colors=colors,
        autopct=lambda p: f'{p:.1f}%',  
        startangle=90,
        pctdistance=0.80,
        radius=0.9,
        textprops={'fontsize': 10}
    )
    
    centre_circle = plt.Circle((0, 0), 0.55, fc='white')
    fig.gca().add_artist(centre_circle)

    
    ax.axis('equal')  
    ax.set_title("Big Five Personality Traits", fontsize=12, color="purple", pad=25)

    # Save chart as Base64 image
    img = io.BytesIO()
    plt.savefig(img, format='png', transparent=True, bbox_inches='tight')
    img.seek(0)
    chart_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close(fig)

    return chart_url

#Overall personality types
def calculate_personality(result):
    avg_score = np.mean(list(result.values()))
    
    if avg_score >= 90:
        name = "Trailblazing Visionary"
        description = "🚀 High Openness - You are an innovative thinker who embraces new ideas fearlessly."
    elif avg_score >= 80:
        name = "Determined Achiever"
        description = "🔥 High Conscientiousness - You are driven, disciplined, and unstoppable in your pursuit of success."
    elif avg_score >= 70:
        name = "Energetic Leader"
        description = "🌟 High Extraversion - A natural leader who thrives in social settings and inspires those around you."
    elif avg_score >= 60:
        name = "Balanced Strategist"
        description = "🎯 A blend of traits - You are adaptable and thoughtful, able to approach life with stability and logic."
    elif avg_score >= 50:
        name = "Compassionate Diplomat"
        description = "💖 High Agreeableness - You foster harmony and bring people together with empathy and kindness."
    elif avg_score >= 40:
        name = "Curious Explorer"
        description = "🔍 Moderate Openness - You enjoy learning and discovering new perspectives, always questioning the status quo."
    elif avg_score >= 30:
        name = "Introspective Analyst"
        description = "🤔 High Neuroticism - You analyze situations deeply, often reflecting on emotions and motivations."
    elif avg_score >= 20:
        name = "Gentle Soul"
        description = "🌿 High Agreeableness & Low Extraversion - You are a peaceful presence, offering support and understanding to those in need."
    else:
        name = "Laid-Back Observer"
        description = "😌 Low Conscientiousness & Extraversion - You go with the flow, preferring to watch and reflect rather than take charge."
    
    return name, description

#Store in Excel file
def append_results_to_file(new_row):
    try:
        results_df = pd.read_excel(RESULTS_FILE)
        results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
        results_df.to_excel(RESULTS_FILE, index=False)
    except Exception as e:
        raise RuntimeError(f"Error saving results: {e}")


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
            scaled_score = round(raw_score * 20, 2)  
            predictions[trait] = scaled_score

        overall_name, overall_description = calculate_personality(predictions)


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



if __name__ == '__main__':
    app.run(debug=True, port=5000)
