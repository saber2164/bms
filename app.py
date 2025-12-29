import os
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from scripts.simple_ukf import SimplifiedDualUKF

app = Flask(__name__)

# --- Global State ---
dukf_instance = None

print("Using simplified UKF with linear OCV model (no neural network)")
print("OCV model: V = 1.2*SoC + 3.0 (3.0V at 0%, 4.2V at 100%)")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/init_filter', methods=['POST'])
def init_filter():
    """Initialize the Dual UKF filter"""
    global dukf_instance
    try:
        data = request.json
        initial_soc = float(data.get('initial_soc', 0.9))
        q_max = float(data.get('q_max', 1.5))
        r0 = float(data.get('r0', 0.05))
        
        # Create simplified UKF instance (no model/scaler needed!)
        dukf_instance = SimplifiedDualUKF(
            dt=1.0,
            C_nom=q_max,
            R0_nom=r0,
            ocv_a=1.2,  # Linear OCV slope
            ocv_b=3.0   # Linear OCV offset
        )
        
        # Set initial SoC
        dukf_instance.x[0] = initial_soc
        
        return jsonify({
            'status': 'success',
            'message': f'Filter initialized with SoC={initial_soc:.2f}, Q_max={q_max}Ah, R0={r0}Ω',
            'initial_state': {
                'soc': float(dukf_instance.x[0]),
                'q_max': float(dukf_instance.theta[0]),
                'r0': float(dukf_instance.theta[1])
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/api/predict_soc', methods=['POST'])
def predict_soc():
    """Run a single UKF step"""
    global dukf_instance
    if dukf_instance is None:
        return jsonify({'status': 'error', 'message': 'Filter not initialized'}), 400
    
    try:
        data = request.json
        print(f"Received prediction request: {data}")
        v_meas = float(data['voltage'])
        i_meas = float(data['current'])
        temp = float(data.get('temperature', 25.0))
        
        # Run UKF step
        soc_est, q_max_est, r0_est = dukf_instance.step(v_meas, i_meas, temp)
        print(f"UKF step result: SoC={soc_est}, Q_max={q_max_est}, R0={r0_est}")
        
        return jsonify({
            'status': 'success',
            'soc': float(soc_est),
            'q_max': float(q_max_est),
            'r0': float(r0_est)
        })
    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/api/reset', methods=['POST'])
def reset_filter():
    """Reset the filter state"""
    global dukf_instance
    dukf_instance = None
    return jsonify({'status': 'success', 'message': 'Filter reset'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
