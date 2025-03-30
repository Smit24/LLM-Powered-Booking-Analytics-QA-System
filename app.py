from flask import Flask, jsonify, request
import pandas as pd
from datetime import datetime
import calendar
import numpy as np
import llama_cpp
import chromadb
from sentence_transformers import SentenceTransformer


app = Flask(__name__)                                                                   #Initializing the Flask app

def initialize_components():
    chroma_client = chromadb.PersistentClient(path="./customer_queries_database")       #Initialized Chroma database 
    collection = chroma_client.get_or_create_collection(name="hotel_queries")
    
    
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")                           #Initialize all-Mini embedding model
    
    try:
        llm = llama_cpp.Llama(                                                          #Initialize LLM for Customer Q/A
            model_path="mistral-7b.Q4_K_M.gguf", 
            n_ctx=4096, 
            n_threads=8
        )
    except Exception as e:
        print(f"Error loading LLM model: {e}")
        llm = None
    
    return collection, embedding_model, llm

def load_and_preprocess_data():
    try:
        df = pd.read_csv('hotel_analytics_dataset_sorted.csv')                          #function to pre-process csv according to date
        df['Date'] = pd.to_datetime(df['Date'])
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['IsWeekend'] = df['DayOfWeek'].isin([5, 6])  
        df['Season'] = df['Month'].apply(
            lambda x: 'Winter' if x in [12,1,2] else 
                     'Spring' if x in [3,4,5] else 
                     'Summer' if x in [6,7,8] else 'Fall'
        )
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


collection, embedding_model, llm = initialize_components()
df = load_and_preprocess_data()


def convert_types(obj):                                                                # Was having issue np and pd types, so converting them to Json Serilixzation
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.float, np.float64)):
        return float(obj)
    elif isinstance(obj, pd.Timestamp):
        return obj.strftime('%Y-%m-%d')
    elif isinstance(obj, (pd.Series, pd.DataFrame)):
        return obj.to_dict(orient='records')
    elif isinstance(obj, dict):
        return {k: convert_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_types(x) for x in obj]
    return obj

def generate_summary_report(dataframe, period_info):                                # Summarized report 
    return {
        'period': period_info,
        'bookings': {
            'total': int(dataframe['Bookings'].sum()),
            'cancellations': int(dataframe['Cancellations'].sum()),
            'no_shows': int(dataframe['No Shows'].sum()),
            'cancellation_rate': float(
                dataframe['Cancellations'].sum() / 
                (dataframe['Bookings'].sum() + dataframe['Cancellations'].sum()) * 100
            )
        },
        'revenue': {
            'total': float(dataframe['Total Revenue'].sum()),
            'average_daily_rate': float(dataframe['Room Price'].mean()),
            'breakdown': {
                'base': float(dataframe['Base Revenue'].sum()),
                'extra_services': float(dataframe['Extra Service Revenue'].sum())
            }
        },
        'services': {
            'early_checkins': int(dataframe['Early Check-in Requests'].sum()),
            'late_checkouts': int(dataframe['Late Check-out Requests'].sum()),
            'minibar_usage': int(dataframe['Minibar Usage'].sum()),
            'laundry_usage': int(dataframe['Laundry Usage'].sum())
        }
    }

def generate_detailed_report(dataframe, period_info):                             #Detailed Report 
    """Generate more detailed report with additional metrics"""
    report = generate_summary_report(dataframe, period_info)
    
    if len(dataframe) > 0:
        #Room Statistics
        room_stats = dataframe.groupby('Room Type').agg({
            'Room Price': 'mean',
            'Bookings': 'sum',
            'Total Revenue': 'sum'
        }).reset_index()
        
        #Weekend vs weekday analysis
        day_type_stats = dataframe.groupby('IsWeekend').agg({
            'Bookings': 'sum',
            'Room Price': 'mean'
        }).reset_index()
        day_type_stats['DayType'] = day_type_stats['IsWeekend'].map(
            {True: 'Weekend', False: 'Weekday'}
        )
        
        report.update({
            'room_type_stats': room_stats,
            'day_type_stats': day_type_stats,
            'seasonal_stats': dataframe['Season'].value_counts().to_dict()
        })
    
    return report


@app.route('/analytics', methods=['POST'])
def get_analytics():
    """Endpoint for hotel analytics data"""
    if df is None:
        return jsonify({'error': 'Data not loaded'}), 500
    
    try:                                                                            #Applied time filters
        month = request.json.get('month')
        year = request.json.get('year')
        day = request.json.get('day')
        report_type = request.json.get('report_type', 'summary')
        
        filtered_df = df.copy()                                                     #Filtered data based on time period
        period_info = "All time"
        
        if month:
            filtered_df = filtered_df[filtered_df['Month'] == month]
            period_info = f"All years, Month {calendar.month_name[month]}"
            if year:
                filtered_df = filtered_df[filtered_df['Year'] == year]
                period_info = f"{calendar.month_name[month]} {year}"
                if day:
                    filtered_df = filtered_df[filtered_df['Day'] == day]
                    period_info = f"{calendar.month_name[month]} {day}, {year}"
        
        if report_type == 'summary':
            response = generate_summary_report(filtered_df, period_info)
        elif report_type == 'detailed':
            response = generate_detailed_report(filtered_df, period_info)
        else:
            return jsonify({'error': 'Invalid report type'}), 400
        
        return jsonify(convert_types(response))
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/ask', methods=['POST'])                                                #for customer queries 
def ask_question():
    if not llm:
        return jsonify({'error': 'Language model not loaded'}), 500
    
    try:
        data = request.get_json()
        query = data.get('query')
        
        if not query:
            return jsonify({'error': 'No query provided'}), 400
        
        query_embedding = embedding_model.encode(query).tolist()                 #converting the queries to embeddings and passing them to  
        results = collection.query(
            query_embeddings=[query_embedding]
        )
        
        context = "\n".join(results['documents'][0])                             #preparing context to provide it to the llm
        
        # Generate answer using Mistral
        prompt = f"""You are helpful chatbot and you help the customers regarding queries of this hotel. Use the following information to answer the query:
        {context}
        
        Query: {query}
        Answer:"""
        
        response = llm(
            prompt,
            max_tokens=256,
            temperature=0.2,
            echo=False
        )
        
        return jsonify({
            'question': query,
            'answer': response['choices'][0]['text'].strip(),
            'sources': results['documents'][0]
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/health', methods=['GET'])                                            #Health check endpoint which is going to check all the components
def health_check():
    try:
        status = {
            'api_status': 'running',
            'database_connection': 'connected' if collection else 'disconnected',
            'llm_status': 'loaded' if llm else 'not_loaded',
            'data_loaded': 'yes' if df is not None else 'no',
            'embedding_model': 'ready'
        }
        
        return jsonify(status), 200
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


if __name__ == '__main__':
    app.run(debug=True, port=5001)