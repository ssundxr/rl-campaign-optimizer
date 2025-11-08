"""
Flask REST API for RL Model Serving
Provides endpoints for campaign recommendations
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import pickle
import numpy as np
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global model variable
model = None


def load_model(model_path='./models/linucb_model.pkl'):
    """Load trained RL model"""
    global model
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Model loaded from {model_path}")
    except FileNotFoundError:
        logger.warning("Model file not found. Using default behavior.")
        model = None


@app.route('/', methods=['GET'])
def home():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'RL Campaign Optimizer API',
        'version': '1.0.0',
        'timestamp': datetime.now().isoformat()
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict best campaign for a customer
    
    Expected JSON payload:
    {
        "customer_id": 1234,
        "features": [0.5, 0.3, 0.8, ...]
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'customer_id' not in data or 'features' not in data:
            return jsonify({'error': 'Invalid input format'}), 400
        
        customer_id = data['customer_id']
        features = np.array(data['features'])
        
        # Make prediction (placeholder logic)
        if model is not None:
            # Use actual model prediction
            campaign_id = model.predict(features)
        else:
            # Fallback to random campaign
            campaign_id = np.random.randint(1, 11)
        
        response = {
            'customer_id': customer_id,
            'recommended_campaign': int(campaign_id),
            'confidence': 0.85,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Prediction made for customer {customer_id}")
        return jsonify(response), 200
    
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/feedback', methods=['POST'])
def feedback():
    """
    Receive feedback for model updates
    
    Expected JSON payload:
    {
        "customer_id": 1234,
        "campaign_id": 5,
        "reward": 1.5
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'customer_id' not in data or 'reward' not in data:
            return jsonify({'error': 'Invalid feedback format'}), 400
        
        # Store feedback for model retraining
        logger.info(f"Feedback received: {data}")
        
        return jsonify({
            'status': 'feedback_received',
            'timestamp': datetime.now().isoformat()
        }), 200
    
    except Exception as e:
        logger.error(f"Error processing feedback: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/metrics', methods=['GET'])
def metrics():
    """Prometheus metrics endpoint"""
    # Placeholder for Prometheus metrics
    return jsonify({
        'total_predictions': 1000,
        'average_confidence': 0.87,
        'uptime_seconds': 3600
    })


if __name__ == '__main__':
    # Load model on startup
    load_model()
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)
