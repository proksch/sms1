"""
Flask API of the SMS Spam detection model model.
"""
import traceback
import joblib
from flask import Flask, jsonify, request
from flasgger import Swagger
import pandas as pd

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
          default: "{'sms':'this is my request'}"
          schema:
            type: object
            properties:
                sms:
                    type: string
    responses:
      200:
        description: "The result of the classification: Spam or Ham."
    """
    input_data = request.get_json()
    sms = input_data.get('sms')
    return jsonify({
        "result": "Spam",
        "classifier": "decision tree",
        "sms": sms
    })
    # try:
    #     json_ = request.json
    #     query_df = pd.DataFrame(json_)
    #     query = pd.get_dummies(query_df)
    #     for col in model_columns:
    #         if col not in query.columns:
    #             query[col] = 0
    #     prediction = clf.predict(query)
    #     return jsonify({'prediction': list(prediction)})
    # except Exception as e:
    #     return jsonify({'error': str(e), 'trace': traceback.format_exc()})

if __name__ == '__main__':
    clf = joblib.load('output/model.joblib')
    app.run(port=8080, debug=True)
