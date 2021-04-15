"""
Flask API of the SMS Spam detection model model.
"""
import traceback
import joblib
from flask import Flask, jsonify, request
from flasgger import Swagger
import pandas as pd

from text_preprocessing import prepare, _extract_message_len, _text_process

app = Flask(__name__)
swagger = Swagger(app)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict whether an SMS is Spam.
    ---
    consumes:
      - application/json
    parameters:
        - name: input_data
          in: body
          description: message to be classified.
          required: True
          schema:
            type: object
            required: sms
            properties:
                sms:
                    type: string
                    example: This is an example of an SMS.
    responses:
      200:
        description: "The result of the classification: 'spam' or 'ham'."
    """
    input_data = request.get_json()
    sms = input_data.get('sms')
    processed_sms = prepare(sms)
    model = joblib.load('output/model.joblib')
    prediction = model.predict(processed_sms)[0]
    
    return jsonify({
        "result": prediction,
        "classifier": "decision tree",
        "sms": sms
    })

@app.route('/dumbpredict', methods=['POST'])
def dumb_predict():
    """
    Dumb prediction of whether a given SMS is Spam or Ham (always predicts 'ham').
    ---
    consumes:
      - application/json
    parameters:
        - name: input_data
          in: body
          description: message to be classified.
          required: True
          schema:
            type: object
            required: sms
            properties:
                sms:
                    type: string
                    example: This is an example of an SMS.
    responses:
      200:
        description: "The result of the classification: 'spam' or 'ham'."
    """
    input_data = request.get_json()
    sms = input_data.get('sms')
    
    return jsonify({
        "result": "Spam",
        "classifier": "decision tree",
        "sms": sms
    })

if __name__ == '__main__':
    clf = joblib.load('output/model.joblib')
    app.run(port=8080, debug=True)
