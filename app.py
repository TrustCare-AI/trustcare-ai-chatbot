from flask import Flask, render_template, request, jsonify
from chat_engine import ChatEngine

app = Flask(__name__)
# Initialize the ML Engine once
engine = ChatEngine()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    action_type = data.get('type')

    if action_type == 'initial':
        # User typed their first symptoms
        user_message = data.get('message', '')
        response = engine.parse_initial_message(user_message)
        return jsonify(response)
        
    elif action_type == 'predict':
        # User finished selecting co-occurring symptoms, make final prediction
        final_symptoms = data.get('symptoms', [])
        if not final_symptoms:
            return jsonify({"status": "error", "message": "No symptoms provided."})
            
        predictions = engine.predict_diseases(final_symptoms)
        
        # Build markdown-like conversational response
        top_disease = predictions[0]
        
        # 1. Symptoms said
        resp = f"<b>Symptoms Analyzed:</b> {', '.join(final_symptoms).title()}<br><br>"
        
        # 2. Predicted disease & Confidence
        resp += f"<b>Primary Diagnosis:</b> The most likely condition is **{top_disease['name']}** (Confidence: {round(top_disease['score']*100)}%).<br><br>"
        
        # 3. What that disease means
        resp += f"<b>What this means:</b> <i>{top_disease['wiki']}</i><br><br>"
        
        # 4. Other likely diseases
        others = [p['name'] for p in predictions[1:4]]
        resp += f"<b>Other likely diseases:</b> {', '.join(others)}.<br><hr style='opacity:0.2;margin:10px 0;'><small><i>Note: I am an AI, not a doctor. Please consult a medical professional for a real diagnosis.</i></small>"
        
        return jsonify({
            "status": "success",
            "message": resp
        })

if __name__ == '__main__':
    print("Starting Flask Server...")
    app.run(debug=True, port=5000)
