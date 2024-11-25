from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from neural_networks import visualize

app = Flask(__name__)

# Define the main route
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle experiment parameters and trigger the experiment
@app.route('/run_experiment', methods=['POST'])
def run_experiment():
    data = request.json
    activation = data.get('activation', '').lower()
    lr = data.get('lr', None)
    step_num = data.get('step_num', None)

    # Input validation
    if activation not in ['relu', 'tanh', 'sigmoid']:
        return jsonify({"error": "Invalid activation function. Choose relu, tanh, or sigmoid."}), 400

    try:
        lr = float(lr)
        step_num = int(step_num)
    except (ValueError, TypeError):
        return jsonify({"error": "Learning rate must be a number and training steps must be an integer."}), 400

    if lr <= 0 or step_num <= 0:
        return jsonify({"error": "Learning rate and training steps must be positive values."}), 400

    # Run the experiment with the provided parameters
    visualize(activation, lr, step_num)

    # Check if result gif is generated and return their paths
    result_gif = "results/visualize.gif"
    
    return jsonify({
        "result_gif": result_gif if os.path.exists(result_gif) else None,
    })

# Route to serve result images
@app.route('/results/<filename>')
def results(filename):
    return send_from_directory('results', filename)

if __name__ == '__main__':
    app.run(debug=True)