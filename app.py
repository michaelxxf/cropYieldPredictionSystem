from flask import Flask, request, jsonify, render_template, session, flash, redirect, url_for
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.wrappers import Response as WerkzeugResponse
from functools import wraps
import numpy as np
import pandas as pd
import os
import pickle
import logging
import mysql.connector
from flask import session, redirect, url_for, flash
from werkzeug.security import generate_password_hash, check_password_hash
from typing import Dict, Any, Union, Tuple, List, Optional, cast
import json
from flask.wrappers import Response
import requests
import plotly

# Import prediction pipeline classes
from src.pipeline.prediction_pipeline import PredictPipeline, CustomData

# Initialize Flask app
app = Flask(__name__)

# Set secret key for session management
app.secret_key = os.urandom(24).hex()  # Generate a secure random key

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the ML model
try:
    model_path = os.path.join('dataset', 'model.pkl')
    logger.info(f"Loading model from {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    model = None

@app.after_request
def add_security_headers(response: WerkzeugResponse) -> WerkzeugResponse:
    response.headers['Content-Security-Policy'] = "default-src 'self'; img-src 'self' https:; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline';"
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

# Database configuration
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '',  # Set your MySQL password here
    'database': 'crop_prediction'
}

# Weather API configuration
WEATHER_API_KEY = os.getenv('OPENWEATHERMAP_API_KEY', '53d81f3426cc1ee8748db205dc530348')

def get_db_connection():
    return mysql.connector.connect(**db_config)

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Check if user is not logged in
        if 'user_id' not in session:
            # Store the requested URL in session for redirect after login
            session['next'] = request.url
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/login', methods=['GET', 'POST'])
def login():
    # Redirect if user is already logged in
    if 'user_id' in session:
        return redirect(url_for('predict'))
        
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute('SELECT * FROM users WHERE email = %s', (email,))
        user = cursor.fetchone()
        cursor.close()
        conn.close()

        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            flash('Logged in successfully!', 'success')
            
            # Redirect to stored next URL or default to predict page
            next_page = session.pop('next', None)
            return redirect(next_page or url_for('predict'))
        
        flash('Invalid email or password.', 'danger')
    return render_template('auth/login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            flash('Passwords do not match.', 'danger')
            return render_template('auth/signup.html')

        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Check if email already exists
        cursor.execute('SELECT id FROM users WHERE email = %s', (email,))
        if cursor.fetchone():
            flash('Email already registered.', 'danger')
            cursor.close()
            conn.close()
            return render_template('auth/signup.html')

        # Create new user
        hashed_password = generate_password_hash(password)
        cursor.execute('INSERT INTO users (username, email, password) VALUES (%s, %s, %s)',
                      (username, email, hashed_password))
        conn.commit()
        cursor.close()
        conn.close()

        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))

    return render_template('auth/signup.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out successfully.', 'success')
    return redirect(url_for('login'))

@app.route('/')
def home():
    return render_template('home.html')

def get_historical_predictions(user_id, days=30):
    """Get historical predictions for a user"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        query = '''
            SELECT nitrogen, phosphorus, potassium, temperature, humidity, 
                   ph, rainfall, yield_prediction, created_at
            FROM predictions 
            WHERE user_id = %s 
            AND created_at >= DATE_SUB(NOW(), INTERVAL %s DAY)
            ORDER BY created_at DESC
        '''
        logger.info(f"Executing historical query for user_id: {user_id}")
        cursor.execute(query, (user_id, days))
        predictions = cursor.fetchall()
        logger.info(f"Found {len(predictions)} historical predictions")
        
        if predictions:
            # Convert all numeric fields to float
            for pred in predictions:
                for key in pred:
                    if key != 'created_at':
                        pred[key] = float(pred[key])
            
        return predictions
    except Exception as e:
        logger.error(f"Error in get_historical_predictions: {str(e)}")
        return []
    finally:
        cursor.close()
        conn.close()

def create_yield_trend_plot(predictions):
    """Create a plotly graph for yield trends"""
    try:
        if not predictions:
            logger.info("No predictions available for plotting")
            return None
        
        df = pd.DataFrame(predictions)
        logger.info(f"Creating plot with columns: {df.columns.tolist()}")
        
        # Ensure proper data types
        df['created_at'] = pd.to_datetime(df['created_at'])
        df['yield_prediction'] = pd.to_numeric(df['yield_prediction'], errors='coerce')
        
        # Check for valid data
        if df['yield_prediction'].isnull().all():
            logger.error("No valid prediction values for plotting")
            return None
            
        # Create the plot
        fig = {
            'data': [{
                'x': df['created_at'].tolist(),
                'y': df['yield_prediction'].tolist(),
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': 'Crop Predictions',
                'line': {'color': '#2193b0'},
                'marker': {'color': '#6dd5ed'}
            }],
            'layout': {
                'title': 'Crop Predictions Over Time',
                'xaxis': {'title': 'Date'},
                'yaxis': {'title': 'Predicted Yield'},
                'template': 'plotly_white',
                'height': 400,
                'margin': {'t': 50, 'b': 50, 'l': 50, 'r': 50}
            }
        }
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    except Exception as e:
        logger.error(f"Error creating yield trend plot: {str(e)}")
        return None

def create_parameter_correlation_plot(predictions):
    """Create a plotly graph showing correlation between parameters and yield"""
    if not predictions:
        return None
    
    df = pd.DataFrame(predictions)
    df['yield_prediction'] = df['yield_prediction'].astype(float)
    
    parameters = ['nitrogen', 'phosphorus', 'potassium', 'temperature', 'humidity', 'ph', 'rainfall']
    correlations = []
    
    for param in parameters:
        corr = np.corrcoef(df[param], df['yield_prediction'])[0, 1]
        correlations.append({
            'parameter': param.capitalize(),
            'correlation': corr
        })
    
    # Sort by absolute correlation value
    correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
    
    fig = {
        'data': [{
            'x': [c['parameter'] for c in correlations],
            'y': [c['correlation'] for c in correlations],
            'type': 'bar',
            'marker': {
                'color': ['#2193b0' if c > 0 else '#ff6b6b' for c in [c['correlation'] for c in correlations]]
            }
        }],
        'layout': {
            'title': 'Parameter Impact on Yield',
            'xaxis': {'title': 'Parameters'},
            'yaxis': {'title': 'Correlation with Yield'},
            'template': 'plotly_white',
            'height': 400,
            'margin': {'t': 50, 'b': 50, 'l': 50, 'r': 50}
        }
    }
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict() -> Union[str, Tuple[Response, int]]:
    try:
        if request.method == 'GET':
            # Get crop facts from database
            conn = get_db_connection()
            cursor = conn.cursor(dictionary=True)
            cursor.execute('SELECT title, description FROM crop_facts ORDER BY RAND() LIMIT 5')
            crop_facts = cursor.fetchall()
            cursor.close()
            conn.close()
            return render_template('predict.html', crop_facts=crop_facts)
        
        # Get data from either JSON or form data
        data = request.get_json() if request.is_json else request.form
        
        # Log the incoming request data
        logger.info(f"Received prediction request with method: {request.method}")
        logger.info(f"Request data: {data}")
        
        try:
            # Create CustomData instance for prediction
            custom_data = CustomData(
                N=float(data['nitrogen']),
                P=float(data['phosphorus']),
                K=float(data['potassium']),
                temperature=float(data['temperature']),
                humidity=float(data['humidity']),
                ph=float(data['ph']),
                rainfall=float(data['rainfall'])
            )
            
            # Log the processed data
            df = custom_data.get_data_as_dataframe()
            logger.info(f"Input DataFrame shape: {df.shape}")
            logger.info(f"Input DataFrame columns: {df.columns.tolist()}")
            logger.info(f"Input data: {df.to_dict('records')}")
            
            # Verify DataFrame is not empty
            if df.empty:
                logger.error("Empty DataFrame created")
                return jsonify({
                    'error': 'Invalid input data',
                    'status': 'validation_error',
                    'details': 'DataFrame creation failed'
                }), 400
            
            # Initialize prediction pipeline and get prediction
            predict_pipeline = PredictPipeline()
            logger.info("Initialized prediction pipeline")
            
            try:
                yield_prediction = predict_pipeline.predict(df)
                logger.info(f"Raw prediction result: {yield_prediction}")
                
                if yield_prediction is None:
                    logger.error("Prediction returned None")
                    return jsonify({
                        'error': 'Prediction failed',
                        'status': 'prediction_error',
                        'details': 'Model returned None'
                    }), 500
                
                if not isinstance(yield_prediction, (list, np.ndarray)) or len(yield_prediction) == 0:
                    logger.error(f"Invalid prediction type: {type(yield_prediction)}")
                    return jsonify({
                        'error': 'Invalid prediction format',
                        'status': 'prediction_error',
                        'details': f'Expected array-like, got {type(yield_prediction)}'
                    }), 500
                
                prediction_value = float(yield_prediction[0])
                logger.info(f"Processed prediction value: {prediction_value}")
                
                # Validate prediction value
                if not np.isfinite(prediction_value):
                    logger.error(f"Invalid prediction value: {prediction_value}")
                    return jsonify({
                        'error': 'Invalid prediction value',
                        'status': 'prediction_error',
                        'details': 'Model returned an invalid number'
                    }), 500
                
            except Exception as pred_error:
                logger.error(f"Prediction error: {str(pred_error)}", exc_info=True)
                return jsonify({
                    'error': 'Prediction computation failed',
                    'status': 'prediction_error',
                    'details': str(pred_error)
                }), 500
            
            # Store prediction in database
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO predictions (
                    user_id, nitrogen, phosphorus, potassium, 
                    temperature, humidity, ph, rainfall, 
                    yield_prediction, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
            """, (
                session['user_id'], float(data['nitrogen']), float(data['phosphorus']),
                float(data['potassium']), float(data['temperature']), float(data['humidity']),
                float(data['ph']), float(data['rainfall']), prediction_value
            ))
            conn.commit()
            cursor.close()
            conn.close()

            # Get recommendations
            recommendations = get_recommendations(data)
            
            # Create response
            response = {
                'yield_prediction': round(prediction_value, 2),
                'recommendations': recommendations,
            }
            
            return jsonify(response)
            
        except Exception as e:
            logger.error(f"Prediction pipeline error: {str(e)}", exc_info=True)
            return jsonify({
                'error': 'Prediction failed',
                'status': 'system_error',
                'details': str(e)
            }), 500

    except Exception as e:
        logger.error(f"Unexpected error in predict endpoint: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'Internal server error',
            'status': 'system_error',
            'details': str(e)
        }), 500

@app.route('/get_historical_data')
@login_required
def get_historical_data():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get historical data for current user only
        cursor.execute(""" 
            SELECT nitrogen, phosphorus, potassium, temperature, humidity, 
            ph, rainfall, yield_prediction, created_at
            FROM predictions 
            WHERE user_id = %s
            ORDER BY created_at DESC
        """, (session['user_id'],))
        
        # Convert to pandas DataFrame
        columns = ['nitrogen', 'phosphorus', 'potassium', 'temperature', 
                  'humidity', 'ph', 'rainfall', 'yield_prediction', 'created_at']
        df = pd.DataFrame(cursor.fetchall(), columns=columns)
        
        if df.empty:
            return jsonify({'error': 'No historical data available'})

        # Convert yield_prediction to float
        df['yield_prediction'] = df['yield_prediction'].astype(float)
        
        # Create time series plot data
        time_series_data = {
            'data': [{
                'x': df['created_at'].tolist(),
                'y': df['yield_prediction'].tolist(),
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': 'Yield Predictions'
            }],
            'layout': {
                'title': 'Historical Yield Predictions',
                'xaxis': {'title': 'Date'},
                'yaxis': {'title': 'Predicted Yield (tons)'}
            }
        }
        
        # Calculate correlations
        numeric_columns = ['nitrogen', 'phosphorus', 'potassium', 'temperature', 
                         'humidity', 'ph', 'rainfall']
        correlations = {}
        
        for param in numeric_columns:
            df[param] = df[param].astype(float)
            corr = np.corrcoef(df[param], df['yield_prediction'])[0, 1]
            correlations[param] = round(corr, 3)
        
        # Create correlation plot data
        correlation_data = {
            'data': [{
                'x': list(correlations.keys()),
                'y': list(correlations.values()),
                'type': 'bar',
                'name': 'Correlation with Yield'
            }],
            'layout': {
                'title': 'Parameter Correlations with Yield',
                'xaxis': {'title': 'Parameters'},
                'yaxis': {'title': 'Correlation Coefficient'}
            }
        }
        
        return jsonify({
            'time_series': time_series_data,
            'correlations': correlation_data
        })
    except Exception as e:
        app.logger.error(f"Error getting historical data: {str(e)}")
        return jsonify({'error': str(e)}), 500
    finally:
        cursor.close()
        conn.close()

def get_crop_facts():
    """Get random crop facts"""
    facts = [
        "Crops need different amounts of sunlight, water, and nutrients to grow well.",
        "Soil pH affects how well plants can absorb nutrients.",
        "Nitrogen helps plants grow strong leaves and stems.",
        "Phosphorus helps plants develop strong roots and produce flowers and fruits.",
        "Potassium helps plants resist diseases and produce better quality fruits.",
        "The right amount of rainfall is crucial for crop growth and development.",
        "Temperature affects how quickly plants grow and develop.",
        "Humidity can impact plant diseases and water requirements.",
        "Some crops are more resistant to environmental stress than others.",
        "Crop rotation helps maintain soil health and prevent diseases."
    ]
    return facts

@app.route('/get_facts')
@login_required
def facts_endpoint():
    try:
        facts = get_crop_facts()
        return jsonify({'facts': facts})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def get_fertilizer_recommendation(N: float, P: float, K: float, ph: float) -> str:
    recommendations: List[str] = []
    
    if N < 30:
        recommendations.append("Low Nitrogen: Consider adding nitrogen-rich fertilizers like urea or ammonium sulfate.")
    elif N > 100:
        recommendations.append("High Nitrogen: Reduce nitrogen fertilizer application.")
        
    if P < 30:
        recommendations.append("Low Phosphorus: Add phosphate fertilizers like DAP or rock phosphate.")
    elif P > 100:
        recommendations.append("High Phosphorus: Reduce phosphorus fertilizer application.")
        
    if K < 30:
        recommendations.append("Low Potassium: Add potash fertilizers like muriate of potash.")
    elif K > 100:
        recommendations.append("High Potassium: Reduce potassium fertilizer application.")
        
    if ph < 6.5:
        recommendations.append("Acidic Soil: Consider adding lime to raise pH.")
    elif ph > 7.5:
        recommendations.append("Alkaline Soil: Consider adding sulfur to lower pH.")
        
    return "\n".join(recommendations) if recommendations else "NPK levels and pH are within optimal range."

def get_weather(latitude: str, longitude: str) -> Dict[str, Any]:
    if not latitude or not longitude:
        raise ValueError('Missing coordinates')
        
    try:
        api_key = os.getenv('OPENWEATHER_API_KEY')
        if not api_key:
            raise ValueError('OpenWeather API key not found')
            
        url = f'https://api.openweathermap.org/data/2.5/weather?lat={latitude}&lon={longitude}&appid={api_key}&units=metric'
        response = requests.get(url)
        response.raise_for_status()
        
        return cast(Dict[str, Any], response.json())
    except requests.RequestException as e:
        logger.error(f"Weather API error: {str(e)}")
        raise

def get_recommendations(data: Dict[str, Any]) -> str:
    recommendations: List[str] = []
    
    # Get NPK and pH recommendations
    recommendations.append(get_fertilizer_recommendation(
        float(data['nitrogen']),
        float(data['phosphorus']),
        float(data['potassium']),
        float(data['ph'])
    ))
    
    # Temperature recommendations
    temp = float(data['temperature'])
    if temp > 30:
        recommendations.append("High temperature detected. Consider providing shade or increasing irrigation.")
    elif temp < 15:
        recommendations.append("Low temperature detected. Consider using protective measures against frost.")
        
    # Humidity recommendations
    humidity = float(data['humidity'])
    if humidity > 80:
        recommendations.append("High humidity levels may increase disease risk. Ensure good air circulation.")
    elif humidity < 40:
        recommendations.append("Low humidity may cause water stress. Consider increasing irrigation.")
        
    # Rainfall recommendations
    rainfall = float(data['rainfall'])
    if rainfall < 100:
        recommendations.append("Low rainfall detected. Consider supplemental irrigation.")
    elif rainfall > 200:
        recommendations.append("High rainfall detected. Ensure proper drainage to prevent waterlogging.")
        
    return "\n".join(recommendations)

@app.route('/get_historical_predictions')
@login_required
def get_historical_predictions_endpoint():
    """Endpoint to get historical predictions data"""
    try:
        user_id = session['user_id']
        days = request.args.get('days', default=30, type=int)
        logger.info(f"Fetching predictions for user {user_id} for the last {days} days")
        
        predictions = get_historical_predictions(user_id, days)
        logger.info(f"Retrieved {len(predictions) if predictions else 0} predictions")
        
        if not predictions:
            return jsonify({
                'error': 'No predictions found',
                'predictions': [],
                'yield_trend': None,
                'parameter_impact': None
            })
            
        # Create plots
        yield_trend = create_yield_trend_plot(predictions)
        parameter_impact = create_parameter_correlation_plot(predictions)
        
        if not yield_trend or not parameter_impact:
            logger.error("Failed to create plots")
            return jsonify({
                'error': 'Failed to create plots',
                'predictions': predictions,
                'yield_trend': None,
                'parameter_impact': None
            })
            
        response_data = {
            'predictions': predictions,
            'yield_trend': yield_trend,
            'parameter_impact': parameter_impact
        }
        
        logger.info("Successfully created response with plots")
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error in get_historical_predictions_endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

def get_prediction_statistics(user_id: int, current_prediction: float) -> dict:
    """Get statistical context for the current prediction"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get historical predictions
        cursor.execute("""
            SELECT yield_prediction 
            FROM predictions 
            WHERE user_id = %s 
            ORDER BY created_at DESC 
            LIMIT 10
        """, (user_id,))
        
        predictions = [float(row[0]) for row in cursor.fetchall()]
        
        if not predictions:
            return None
            
        stats = {
            'average': round(sum(predictions) / len(predictions), 2),
            'maximum': round(max(predictions), 2),
            'minimum': round(min(predictions), 2),
            'trend': 'up' if current_prediction > (sum(predictions) / len(predictions)) else 'down'
        }
        
        cursor.close()
        conn.close()
        return stats
        
    except Exception as e:
        logger.error(f"Error calculating prediction statistics: {str(e)}")
        return None

def calculate_prediction_confidence(features_df: pd.DataFrame) -> float:
    """Calculate a confidence score for the prediction"""
    try:
        # Simple confidence calculation based on input ranges
        confidence_scores = []
        
        # NPK levels (0-200 is normal range)
        for col in ['N', 'P', 'K']:
            value = features_df[col].iloc[0]
            if 0 <= value <= 200:
                confidence_scores.append(100)
            else:
                confidence_scores.append(max(0, 100 - abs(value - 100) / 2))
        
        # Temperature (15-35°C is optimal)
        temp = features_df['temperature'].iloc[0]
        if 15 <= temp <= 35:
            confidence_scores.append(100)
        else:
            confidence_scores.append(max(0, 100 - abs(temp - 25) * 4))
        
        # Humidity (40-80% is optimal)
        humidity = features_df['humidity'].iloc[0]
        if 40 <= humidity <= 80:
            confidence_scores.append(100)
        else:
            confidence_scores.append(max(0, 100 - abs(humidity - 60) * 2))
        
        # pH (5.5-7.5 is optimal)
        ph = features_df['pH'].iloc[0]
        if 5.5 <= ph <= 7.5:
            confidence_scores.append(100)
        else:
            confidence_scores.append(max(0, 100 - abs(ph - 6.5) * 20))
        
        # Calculate average confidence
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        return round(avg_confidence, 1)
        
    except Exception as e:
        logger.error(f"Error calculating prediction confidence: {str(e)}")
        return 0.0

def get_recommendations(data):
    """Get recommendations based on input data"""
    recommendations = []
    
    # Get fertilizer recommendations
    recommendations.append(get_fertilizer_recommendation(
        float(data['nitrogen']), 
        float(data['phosphorus']), 
        float(data['potassium']), 
        float(data['ph'])
    ))
    
    # Add weather-based recommendations
    if float(data['temperature']) > 30:
        recommendations.append("High temperature detected. Consider providing shade or increasing irrigation.")
    elif float(data['temperature']) < 15:
        recommendations.append("Low temperature detected. Consider using protective measures against frost.")
    
    if float(data['humidity']) > 80:
        recommendations.append("High humidity levels may increase disease risk. Ensure good air circulation.")
    elif float(data['humidity']) < 40:
        recommendations.append("Low humidity may cause water stress. Consider increasing irrigation.")
    
    if float(data['rainfall']) < 100:
        recommendations.append("Low rainfall detected. Consider supplemental irrigation.")
    elif float(data['rainfall']) > 200:
        recommendations.append("High rainfall detected. Ensure proper drainage to prevent waterlogging.")
    
    return "\n".join(recommendations)

def validate_prediction_input(data: Dict[str, Any]) -> Optional[str]:
    """Validate the input data for prediction"""
    
    # Check for missing fields
    required_fields = ['nitrogen', 'phosphorus', 'potassium', 'temperature', 
                      'humidity', 'ph', 'rainfall']
    
    for field in required_fields:
        if field not in data:
            return f'Missing required field: {field}'

    # Validate field types and ranges
    try:
        # NPK values should be positive
        for nutrient in ['nitrogen', 'phosphorus', 'potassium']:
            value = float(data[nutrient])
            if value < 0 or value > 200:
                return f'{nutrient.capitalize()} value must be between 0 and 200'
        
        # Temperature validation (-20 to 50 Celsius seems reasonable)
        temp = float(data['temperature'])
        if temp < -20 or temp > 50:
            return 'Temperature must be between -20 and 50 °C'
            
        # Humidity validation (0-100%)
        humidity = float(data['humidity'])
        if humidity < 0 or humidity > 100:
            return 'Humidity must be between 0 and 100%'
            
        # pH validation (0-14)
        ph = float(data['ph'])
        if ph < 0 or ph > 14:
            return 'pH must be between 0 and 14'
            
        # Rainfall validation (in mm, 0-5000mm seems reasonable for annual rainfall)
        rainfall = float(data['rainfall'])
        if rainfall < 0 or rainfall > 5000:
            return 'Rainfall must be between 0 and 5000mm'
            
    except ValueError:
        return 'All values must be valid numbers'
        
    return None

@app.route('/history')
@login_required
def history():
    """Render the history page"""
    return render_template('history.html')

@app.route('/delete_prediction/<int:prediction_id>', methods=['DELETE'])
@login_required
def delete_prediction(prediction_id):
    """Delete a prediction"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Verify the prediction belongs to the current user
        cursor.execute("""
            SELECT user_id FROM predictions 
            WHERE id = %s
        """, (prediction_id,))
        
        result = cursor.fetchone()
        if not result or result[0] != session['user_id']:
            return jsonify({'error': 'Prediction not found or unauthorized'}), 404
        
        # Delete the prediction
        cursor.execute("""
            DELETE FROM predictions 
            WHERE id = %s AND user_id = %s
        """, (prediction_id, session['user_id']))
        
        conn.commit()
        return jsonify({'message': 'Prediction deleted successfully'})
        
    except Exception as e:
        logger.error(f"Error deleting prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    app.run(debug=True)


