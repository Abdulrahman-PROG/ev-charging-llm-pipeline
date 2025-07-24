"""
Configuration file for EV Charging LLM Pipeline
Centralized configuration for data collection, training, evaluation, and deployment
"""

import os
from datetime import datetime


class Config:
    """Base configuration class"""
    
    # Project Information
    PROJECT_NAME = "EV Charging LLM Pipeline"
    PROJECT_VERSION = "1.0.0"
    TARGET_DOMAIN = "electric vehicle charging stations"
    USE_CASE = "question_answering"
    
    # Environment
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")  # development, staging, production
    DEBUG = os.getenv("DEBUG", "True").lower() == "true"
    
    # Directories
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    OUTPUT_DIR = os.path.join(BASE_DIR, "output_data")
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    LOGS_DIR = os.path.join(BASE_DIR, "logs")
    RESULTS_DIR = os.path.join(BASE_DIR, "evaluation_results")

    class DataCollection:
        PDF_FOLDER = "pdfs"
        WEB_SCRAPING_URLS = [
            "https://afdc.energy.gov/fuels/electricity_charging_home.html",
            "https://afdc.energy.gov/fuels/electricity_charging_public.html",
            "https://www.energy.gov/eere/electricvehicles/charging-home",
            "https://www.energy.gov/eere/electricvehicles/electric-vehicle-charging-infrastructure"
        ]
        EV_KEYWORDS = [
            'charging station', 'electric vehicle', 'ev charging', 'chargepoint',
            'supercharger', 'fast charging', 'charging network', 'charging infrastructure',
            'battery', 'plug-in', 'connector', 'AC charging', 'DC charging'
        ]
        MIN_TEXT_LENGTH = 200
        MAX_TEXT_LENGTH = 50000
        QUALITY_THRESHOLD = 0.5
        REQUEST_TIMEOUT = 10
        DELAY_BETWEEN_REQUESTS = 2
        USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

    class Model:
        BASE_MODEL_NAME = os.getenv("BASE_MODEL_NAME", "microsoft/DialoGPT-small")
        MAX_MODEL_PARAMETERS = "7B"
        MAX_LENGTH = int(os.getenv("MAX_LENGTH", "512"))
        TRAIN_BATCH_SIZE = int(os.getenv("TRAIN_BATCH_SIZE", "4"))
        EVAL_BATCH_SIZE = int(os.getenv("EVAL_BATCH_SIZE", "4"))
        LEARNING_RATE = float(os.getenv("LEARNING_RATE", "5e-4"))
        NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", "3"))
        WARMUP_STEPS = int(os.getenv("WARMUP_STEPS", "100"))
        LORA_R = int(os.getenv("LORA_R", "16"))
        LORA_ALPHA = int(os.getenv("LORA_ALPHA", "32"))
        LORA_DROPOUT = float(os.getenv("LORA_DROPOUT", "0.1"))
        LORA_TARGET_MODULES = ["c_attn", "c_proj"]
        GENERATION_MAX_LENGTH = int(os.getenv("GENERATION_MAX_LENGTH", "200"))
        GENERATION_TEMPERATURE = float(os.getenv("GENERATION_TEMPERATURE", "0.7"))
        GENERATION_TOP_P = float(os.getenv("GENERATION_TOP_P", "0.9"))
        GENERATION_TOP_K = int(os.getenv("GENERATION_TOP_K", "50"))
        FINE_TUNED_MODEL_PATH = ""  
        FINAL_MODEL_PATH = ""       

    class Training:
        TRAIN_SPLIT = 0.8
        VAL_SPLIT = 0.1
        TEST_SPLIT = 0.1
        RANDOM_SEED = 42
        LOGGING_STEPS = int(os.getenv("LOGGING_STEPS", "10"))
        SAVE_STEPS = int(os.getenv("SAVE_STEPS", "500"))
        EVAL_STEPS = int(os.getenv("EVAL_STEPS", "500"))
        SAVE_TOTAL_LIMIT = 2
        WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", "0.01"))
        ADAM_EPSILON = float(os.getenv("ADAM_EPSILON", "1e-8"))
        MAX_GRAD_NORM = float(os.getenv("MAX_GRAD_NORM", "1.0"))
        FP16 = os.getenv("FP16", "True").lower() == "true"
        EXPERIMENT_NAME = f"ev_charging_llm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        WANDB_PROJECT = os.getenv("WANDB_PROJECT", "ev-charging-llm")
        WANDB_ENTITY = os.getenv("WANDB_ENTITY", None)

    class Evaluation:
        BENCHMARK_SIZE = int(os.getenv("BENCHMARK_SIZE", "50"))
        TEST_SET_SIZE = int(os.getenv("TEST_SET_SIZE", "20"))
        ROUGE_METRICS = ['rouge1', 'rouge2', 'rougeL']
        BLEU_SMOOTHING = True
        DOMAIN_QUESTIONS = [
            {
                "question": "What are the different types of EV charging connectors?",
                "expected_keywords": ["Type 1", "Type 2", "CHAdeMO", "CCS", "connector", "charging"]
            },
            {
                "question": "How long does it take to charge an electric vehicle?",
                "expected_keywords": ["time", "hours", "fast", "slow", "charging", "battery"]
            },
            {
                "question": "What is the difference between AC and DC charging?",
                "expected_keywords": ["AC", "DC", "alternating", "direct", "current", "charging"]
            },
            {
                "question": "Where can I find public charging stations?",
                "expected_keywords": ["public", "stations", "location", "map", "app", "network"]
            },
            {
                "question": "What is fast charging for electric vehicles?",
                "expected_keywords": ["fast", "rapid", "DC", "charging", "quick", "speed"]
            },
            {
                "question": "How much does it cost to charge an electric vehicle?",
                "expected_keywords": ["cost", "price", "money", "charging", "electricity", "rate"]
            },
            {
                "question": "Can I charge my EV at home?",
                "expected_keywords": ["home", "residential", "charging", "installation", "outlet"]
            },
            {
                "question": "What is charging infrastructure?",
                "expected_keywords": ["infrastructure", "network", "stations", "grid", "charging"]
            },
            {
                "question": "How do I install a home charging station?",
                "expected_keywords": ["install", "home", "electrician", "charging", "station", "setup"]
            },
            {
                "question": "What are the benefits of electric vehicle charging?",
                "expected_keywords": ["benefits", "advantages", "clean", "environment", "cost", "charging"]
            }
        ]

    class Deployment:
        API_HOST = os.getenv("API_HOST", "0.0.0.0")
        API_PORT = int(os.getenv("API_PORT", "8000"))
        API_WORKERS = int(os.getenv("API_WORKERS", "1"))
        MODEL_CACHE_SIZE = int(os.getenv("MODEL_CACHE_SIZE", "1"))
        MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "10"))
        REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))
        MIN_REPLICAS = int(os.getenv("MIN_REPLICAS", "1"))
        MAX_REPLICAS = int(os.getenv("MAX_REPLICAS", "5"))
        TARGET_CPU_UTILIZATION = int(os.getenv("TARGET_CPU_UTILIZATION", "70"))
        HEALTH_CHECK_INTERVAL = int(os.getenv("HEALTH_CHECK_INTERVAL", "30"))
        HEALTH_CHECK_TIMEOUT = int(os.getenv("HEALTH_CHECK_TIMEOUT", "5"))
        CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
        CORS_METHODS = ["GET", "POST", "OPTIONS"]
        CORS_HEADERS = ["Content-Type", "Authorization"]

    class Monitoring:
        LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
        LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        LOG_FILE = ""  
        METRICS_ENABLED = os.getenv("METRICS_ENABLED", "True").lower() == "true"
        METRICS_PORT = int(os.getenv("METRICS_PORT", "9090"))
        TRACK_LATENCY = True
        TRACK_THROUGHPUT = True
        TRACK_ERROR_RATE = True
        TRACK_MODEL_PERFORMANCE = True
        MAX_RESPONSE_TIME = float(os.getenv("MAX_RESPONSE_TIME", "5.0"))
        MIN_ACCURACY_THRESHOLD = float(os.getenv("MIN_ACCURACY_THRESHOLD", "0.7"))
        MAX_ERROR_RATE = float(os.getenv("MAX_ERROR_RATE", "0.05"))

    class Database:
        DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///ev_charging_llm.db")
        DATABASE_POOL_SIZE = int(os.getenv("DATABASE_POOL_SIZE", "5"))
        DATABASE_TIMEOUT = int(os.getenv("DATABASE_TIMEOUT", "30"))

    class Security:
        SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
        API_KEY_REQUIRED = os.getenv("API_KEY_REQUIRED", "False").lower() == "true"
        RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
        AUTH_ENABLED = os.getenv("AUTH_ENABLED", "False").lower() == "true"
        JWT_SECRET = os.getenv("JWT_SECRET", "jwt-secret-key")
        JWT_EXPIRATION_HOURS = int(os.getenv("JWT_EXPIRATION_HOURS", "24"))

    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist"""
        directories = [
            cls.DATA_DIR,
            cls.OUTPUT_DIR,
            cls.MODELS_DIR,
            cls.LOGS_DIR,
            cls.RESULTS_DIR
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    @classmethod
    def get_config_summary(cls):
        """Get a summary of current configuration"""
        return {
            "project": {
                "name": cls.PROJECT_NAME,
                "version": cls.PROJECT_VERSION,
                "domain": cls.TARGET_DOMAIN,
                "use_case": cls.USE_CASE,
                "environment": cls.ENVIRONMENT
            },
            "model": {
                "base_model": cls.Model.BASE_MODEL_NAME,
                "max_length": cls.Model.MAX_LENGTH,
                "batch_size": cls.Model.TRAIN_BATCH_SIZE,
                "learning_rate": cls.Model.LEARNING_RATE,
                "epochs": cls.Model.NUM_EPOCHS
            },
            "deployment": {
                "host": cls.Deployment.API_HOST,
                "port": cls.Deployment.API_PORT,
                "workers": cls.Deployment.API_WORKERS
            }
        }

    
    @classmethod
    def initialize_paths(cls):
        cls.Model.FINE_TUNED_MODEL_PATH = os.path.join(cls.MODELS_DIR, "fine_tuned_model")
        cls.Model.FINAL_MODEL_PATH = os.path.join(cls.Model.FINE_TUNED_MODEL_PATH, "final_model")
        cls.Monitoring.LOG_FILE = os.path.join(cls.LOGS_DIR, f"pipeline_{datetime.now().strftime('%Y%m%d')}.log")

# Environment-specific configurations
class DevelopmentConfig(Config):
    DEBUG = True
    Config.Model.NUM_EPOCHS = 1
    Config.Training.SAVE_STEPS = 100
    Config.Evaluation.BENCHMARK_SIZE = 10

class ProductionConfig(Config):
    DEBUG = False
    Config.Model.NUM_EPOCHS = 5
    Config.Deployment.API_WORKERS = 4
    Config.Monitoring.METRICS_ENABLED = True
    Config.Security.API_KEY_REQUIRED = True
    Config.Security.AUTH_ENABLED = True

class TestingConfig(Config):
    DEBUG = True
    Config.Model.NUM_EPOCHS = 1
    Config.Training.SAVE_STEPS = 50
    Config.Evaluation.BENCHMARK_SIZE = 5
    Config.Database.DATABASE_URL = "sqlite:///:memory:"

# Configuration factory
def get_config():
    env = os.getenv("ENVIRONMENT", "development").lower()
    if env == "production":
        return ProductionConfig
    elif env == "testing":
        return TestingConfig
    else:
        return DevelopmentConfig

# Initialize configuration
config = get_config()
Config.initialize_paths()  
config.create_directories()