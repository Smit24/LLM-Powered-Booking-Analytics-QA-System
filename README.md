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
- Input
> curl -X POST http://localhost:5001/analytics -H "Content-Type: application/json" -d '{"month": 12, "year": 2023}'

Expected Output
> {
  "bookings": {
    "cancellation_rate": 16.32208922742111,
    "cancellations": 150,
    "no_shows": 21,
    "total": 769
  },
  "period": "December 2023",
  "revenue": {
    "average_daily_rate": 173.97844732921652,
    "breakdown": {
      "base": 133078.1381354096,
      "extra_services": 2200.0
    },
    "total": 135278.1381354096
  },
  "services": {
    "early_checkins": 21,
    "late_checkouts": 15,
    "laundry_usage": 73,
    "minibar_usage": 45
  }
}


2. Question Answering Endpoint

- URL: /ask
- Method: POST
- Input
> {
  "query": "What is the cancellation policy for hotel bookings?"
}
- Expected Output
> Response:  Our cancellation policy varies depending on the room rate and the time of cancellation. For bookings made at least 72 hours in advance, you can cancel your reservation without any penalty. However, if you cancel within 72 hours, you will incur a 10% fee. Some non-refundable rates may not allow cancellations at all. Refunds are subject to the cancellation timing and the terms of the specific rate you booked. Please review the terms and conditions of your booking for more details on our cancellation policy.

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
