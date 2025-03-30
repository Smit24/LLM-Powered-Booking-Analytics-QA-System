# LLM-Powered-Booking-Analytics-QA-System
## Hotel Analytics and Customer Support API
This Flask application provides two main functionalities:
- Hotel analytics reporting (summary and detailed)
- Customer query answering using Mistral-7B LLM

## Setup Instructions
### Install Python dependencies: 
> pip install -r requirements.txt

Note: Also downloadthe Mistral LLM rfom the link provided in the requirements.txt file.  
### Download required models: 
- Download the Mistral-7B quantized model (mistral-7b.Q4_K_M.gguf) and place it in your project directory.
- The SentenceTransformer model (all-MiniLM-L6-v2) will download automatically on first run.

### Prepare your data:
- Place your hotel analytics CSV file as hotel_analytics_dataset_sorted.csv in the project root.
- The app expects specific columns - refer to the code for exact requirements.

### Running the Application

Start the Flask development server:
> python app.py

The API will be available at: http://localhost:5001

### API Endpoints

1. Analytics Endpoint
- URL: /analytics
- Method: POST
- Request Body (JSON):
> {
  "month": 6,          // optional (1-12)
  "year": 2023,        // optional
  "day": 15,           // optional
  "report_type": "summary"  // "summary" or "detailed"
}

2. Question Answering Endpoint

- URL: /ask
- Method: POST
- Request Body (JSON):

> {
  "query": "What time is check-in?"
}

3. Health Check

- URL: /health
- Method: GET

### System Requirements

- Python 3.8 or higher
- At least 8GB RAM (16GB recommended for LLM)
- For best performance, use a machine with:
a. Multi-core CPU
b. SSD storage
c. 16+ GB RAM if running both the LLM and analytics.

### Notes
- The first run will be slow as it loads the ML models.

### Troubleshooting
- If you encounter issues:
a. Check that the model files are in place.
b. Verify your CSV file has the correct format.
c. Check the /health endpoint for component status.
d. Look for error messages in the console output.
