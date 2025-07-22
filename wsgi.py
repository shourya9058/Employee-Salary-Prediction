from app import app, init_model
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the model when the WSGI server starts
try:
    logger.info("Initializing model...")
    init_model()
    logger.info("Model initialization complete")
except Exception as e:
    logger.error(f"Error initializing model: {str(e)}")
    raise

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
