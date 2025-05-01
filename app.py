 

# In[ ]:


from flask import Flask, request, jsonify
import pandas as pd
from inference.inference import predict

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_route():
    try:
        data = request.get_json()

        # Handle both single and batch input
        if isinstance(data, list):
            input_df = pd.DataFrame(data)
        else:
            input_df = pd.DataFrame([data])

        predictions = predict(input_df)
        return jsonify(predictions)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
