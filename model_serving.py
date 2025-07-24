"""
Model Serving API for EV Charging LLM
Flask-based API for serving the fine-tuned model in production
"""

import os
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional

import torch
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForCausalLM
import threading
from queue import Queue
import psutil

from config import config

# Setup logging
logging.basicConfig(
    level=getattr(logging, config.Monitoring.LOG_LEVEL),
    format=config.Monitoring.LOG_FORMAT,
    handlers=[
        logging.FileHandler(config.Monitoring.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelServer:
    """Model serving class with caching and performance monitoring"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_loaded = False
        self.request_queue = Queue()
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time": 0.0,
            "model_load_time": 0.0,
            "last_request_time": None
        }
        
        logger.info(f"ModelServer initialized on device: {self.device}")
    
    def load_model(self):
        """Load the fine-tuned model"""
        start_time = time.time()
        
        try:
            model_path = config.Model.FINAL_MODEL_PATH
            
            # Check if fine-tuned model exists
            if os.path.exists(model_path):
                logger.info(f"Loading fine-tuned model from: {model_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None
                )
            else:
                # Fallback to base model
                logger.warning(f"Fine-tuned model not found at {model_path}, using base model")
                self.tokenizer = AutoTokenizer.from_pretrained(config.Model.BASE_MODEL_NAME)
                self.model = AutoModelForCausalLM.from_pretrained(
                    config.Model.BASE_MODEL_NAME,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None
                )
            
            # Setup tokenizer
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Move model to device if not using device_map
            if not torch.cuda.is_available():
                self.model = self.model.to(self.device)
            
            self.model.eval()
            self.model_loaded = True
            
            load_time = time.time() - start_time
            self.metrics["model_load_time"] = load_time
            
            logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
            logger.info(f"Model parameters: {self.model.num_parameters():,}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise e
    
    def format_prompt(self, instruction: str, input_text: str = "") -> str:
        """Format the prompt for the model"""
        if input_text:
            return f"### Instruction:\\n{instruction}\\n\\n### Input:\\n{input_text}\\n\\n### Response:\\n"
        else:
            return f"### Instruction:\\n{instruction}\\n\\n### Response:\\n"
    
    def generate_response(self, prompt: str, max_length: int = None, temperature: float = None) -> str:
        """Generate response using the loaded model"""
        if not self.model_loaded:
            raise RuntimeError("Model not loaded")
        
        # Use config defaults if not specified
        max_length = max_length or config.Model.GENERATION_MAX_LENGTH
        temperature = temperature or config.Model.GENERATION_TEMPERATURE
        
        try:
            # Tokenize input
            inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=min(max_length, inputs.shape[1] + 150),
                    num_return_sequences=1,
                    temperature=temperature,
                    top_p=config.Model.GENERATION_TOP_P,
                    top_k=config.Model.GENERATION_TOP_K,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the input prompt from response
            if prompt in response:
                response = response.replace(prompt, "").strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            raise e
    
    def update_metrics(self, response_time: float, success: bool):
        """Update performance metrics"""
        self.metrics["total_requests"] += 1
        self.metrics["last_request_time"] = datetime.now().isoformat()
        
        if success:
            self.metrics["successful_requests"] += 1
        else:
            self.metrics["failed_requests"] += 1
        
        # Update average response time
        total_successful = self.metrics["successful_requests"]
        if total_successful > 0:
            current_avg = self.metrics["average_response_time"]
            self.metrics["average_response_time"] = (
                (current_avg * (total_successful - 1) + response_time) / total_successful
            )
    
    def get_system_metrics(self) -> Dict:
        """Get system performance metrics"""
        return {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent,
            "gpu_available": torch.cuda.is_available(),
            "gpu_memory_used": torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0,
            "gpu_memory_total": torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0
        }

# Initialize model server
model_server = ModelServer()

# Create Flask app
app = Flask(__name__)
CORS(app, origins=config.Deployment.CORS_ORIGINS)

# HTML template for the web interface
WEB_INTERFACE_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>EV Charging LLM API</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .container { background: #f5f5f5; padding: 20px; border-radius: 10px; margin: 20px 0; }
        .form-group { margin: 15px 0; }
        label { display: block; margin-bottom: 5px; font-weight: bold; }
        input, textarea, select { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }
        button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
        button:hover { background: #0056b3; }
        .response { background: #e9ecef; padding: 15px; border-radius: 5px; margin-top: 20px; }
        .metrics { background: #d4edda; padding: 10px; border-radius: 5px; margin: 10px 0; }
        .error { background: #f8d7da; color: #721c24; padding: 10px; border-radius: 5px; }
    </style>
</head>
<body>
    <h1>🚗⚡ EV Charging LLM API</h1>
    
    <div class="container">
        <h2>Ask a Question</h2>
        <form id="questionForm">
            <div class="form-group">
                <label for="question">Question:</label>
                <textarea id="question" rows="3" placeholder="Ask about electric vehicle charging..."></textarea>
            </div>
            <div class="form-group">
                <label for="context">Additional Context (optional):</label>
                <textarea id="context" rows="2" placeholder="Provide additional context if needed..."></textarea>
            </div>
            <div class="form-group">
                <label for="temperature">Temperature (0.1-1.0):</label>
                <input type="number" id="temperature" min="0.1" max="1.0" step="0.1" value="0.7">
            </div>
            <button type="submit">Get Answer</button>
        </form>
        
        <div id="response" class="response" style="display: none;">
            <h3>Response:</h3>
            <div id="responseText"></div>
            <div id="responseTime" class="metrics"></div>
        </div>
        
        <div id="error" class="error" style="display: none;"></div>
    </div>
    
    <div class="container">
        <h2>Example Questions</h2>
        <ul>
            <li><a href="#" onclick="setQuestion('What are the different types of EV charging connectors?')">Types of charging connectors</a></li>
            <li><a href="#" onclick="setQuestion('How long does it take to charge an electric vehicle?')">Charging time</a></li>
            <li><a href="#" onclick="setQuestion('What is the difference between AC and DC charging?')">AC vs DC charging</a></li>
            <li><a href="#" onclick="setQuestion('Where can I find public charging stations?')">Finding charging stations</a></li>
            <li><a href="#" onclick="setQuestion('How much does it cost to charge an electric vehicle?')">Charging costs</a></li>
        </ul>
    </div>

    <script>
        function setQuestion(question) {
            document.getElementById('question').value = question;
        }
        
        document.getElementById('questionForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const question = document.getElementById('question').value;
            const context = document.getElementById('context').value;
            const temperature = parseFloat(document.getElementById('temperature').value);
            
            if (!question.trim()) {
                showError('Please enter a question');
                return;
            }
            
            const button = e.target.querySelector('button');
            button.disabled = true;
            button.textContent = 'Generating...';
            
            try {
                const response = await fetch('/generate', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        instruction: question,
                        input: context,
                        temperature: temperature
                    })
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    showResponse(data.response, data.response_time);
                    hideError();
                } else {
                    showError(data.error || 'An error occurred');
                }
            } catch (error) {
                showError('Network error: ' + error.message);
            } finally {
                button.disabled = false;
                button.textContent = 'Get Answer';
            }
        });
        
        function showResponse(text, responseTime) {
            document.getElementById('responseText').textContent = text;
            document.getElementById('responseTime').textContent = `Response time: ${responseTime.toFixed(2)}s`;
            document.getElementById('response').style.display = 'block';
        }
        
        function showError(message) {
            document.getElementById('error').textContent = message;
            document.getElementById('error').style.display = 'block';
        }
        
        function hideError() {
            document.getElementById('error').style.display = 'none';
        }
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    """Web interface for the API"""
    return render_template_string(WEB_INTERFACE_HTML)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model_server.model_loaded,
        "timestamp": datetime.now().isoformat(),
        "version": config.PROJECT_VERSION
    })

@app.route('/metrics', methods=['GET'])
def get_metrics():
    """Get performance metrics"""
    return jsonify({
        "model_metrics": model_server.metrics,
        "system_metrics": model_server.get_system_metrics(),
        "config": config.get_config_summary()
    })

@app.route('/generate', methods=['POST'])
def generate():
    """Generate response endpoint"""
    start_time = time.time()
    
    try:
        # Validate request
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        
        data = request.get_json()
        instruction = data.get('instruction', '').strip()
        
        if not instruction:
            return jsonify({"error": "Instruction is required"}), 400
        
        # Get optional parameters
        input_text = data.get('input', '').strip()
        max_length = data.get('max_length', config.Model.GENERATION_MAX_LENGTH)
        temperature = data.get('temperature', config.Model.GENERATION_TEMPERATURE)
        
        # Validate parameters
        if not (0.1 <= temperature <= 1.0):
            return jsonify({"error": "Temperature must be between 0.1 and 1.0"}), 400
        
        if not (50 <= max_length <= 500):
            return jsonify({"error": "Max length must be between 50 and 500"}), 400
        
        # Format prompt
        prompt = model_server.format_prompt(instruction, input_text)
        
        # Generate response
        response = model_server.generate_response(prompt, max_length, temperature)
        
        response_time = time.time() - start_time
        model_server.update_metrics(response_time, True)
        
        logger.info(f"Generated response in {response_time:.2f}s for instruction: {instruction[:50]}...")
        
        return jsonify({
            "response": response,
            "response_time": response_time,
            "instruction": instruction,
            "input": input_text,
            "parameters": {
                "max_length": max_length,
                "temperature": temperature
            }
        })
        
    except Exception as e:
        response_time = time.time() - start_time
        model_server.update_metrics(response_time, False)
        
        logger.error(f"Generation failed: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/batch_generate', methods=['POST'])
def batch_generate():
    """Batch generation endpoint"""
    start_time = time.time()
    
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        
        data = request.get_json()
        requests_list = data.get('requests', [])
        
        if not requests_list or len(requests_list) > 10:
            return jsonify({"error": "Provide 1-10 requests"}), 400
        
        responses = []
        
        for req in requests_list:
            instruction = req.get('instruction', '').strip()
            if not instruction:
                responses.append({"error": "Instruction is required"})
                continue
            
            input_text = req.get('input', '').strip()
            max_length = req.get('max_length', config.Model.GENERATION_MAX_LENGTH)
            temperature = req.get('temperature', config.Model.GENERATION_TEMPERATURE)
            
            try:
                prompt = model_server.format_prompt(instruction, input_text)
                response = model_server.generate_response(prompt, max_length, temperature)
                
                responses.append({
                    "response": response,
                    "instruction": instruction,
                    "input": input_text
                })
            except Exception as e:
                responses.append({"error": str(e)})
        
        total_time = time.time() - start_time
        
        return jsonify({
            "responses": responses,
            "total_time": total_time,
            "count": len(responses)
        })
        
    except Exception as e:
        logger.error(f"Batch generation failed: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

def initialize_server():
    """Initialize the model server"""
    logger.info("Initializing model server...")
    
    try:
        model_server.load_model()
        logger.info("Model server initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize model server: {str(e)}")
        raise e

if __name__ == '__main__':
    # Initialize server
    initialize_server()
    
    # Run Flask app
    logger.info(f"Starting API server on {config.Deployment.API_HOST}:{config.Deployment.API_PORT}")
    app.run(
        host=config.Deployment.API_HOST,
        port=config.Deployment.API_PORT,
        debug=config.DEBUG,
        threaded=True
    )

