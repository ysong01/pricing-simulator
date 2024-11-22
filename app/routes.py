from flask import jsonify, request
from app import app
from app.models import PricingCalculator

@app.route('/api/simulate', methods=['POST'])
def simulate_pricing():
    try:
        data = request.get_json()
        calculator = PricingCalculator(data)
        results = calculator.calculate_optimal_price()
        return jsonify({"success": True, "data": results}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

@app.route('/api/sensitivity', methods=['POST'])
def analyze_sensitivity():
    try:
        data = request.get_json()
        calculator = PricingCalculator(data['inputs'])
        sensitivity_results = calculator.sensitivity_analysis(
            variable=data['variable'],
            range_percent=data.get('range_percent', 0.2)
        )
        return jsonify({"success": True, "data": sensitivity_results}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route('/api/forecast', methods=['POST'])
def forecast_prices():
    try:
        data = request.get_json()
        calculator = PricingCalculator(data)
        forecast = calculator.forecast_optimal_prices(periods=data.get('periods', 12))
        return jsonify({"success": True, "data": forecast}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route('/api/risk-analysis', methods=['POST'])
def analyze_risk():
    try:
        data = request.get_json()
        calculator = PricingCalculator(data)
        risk_analysis = calculator.calculate_risk_adjusted_price()
        return jsonify({"success": True, "data": risk_analysis}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400 