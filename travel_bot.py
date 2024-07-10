import json
import logging
import os
import re
import traceback
from collections import Counter

import nltk
from dotenv import load_dotenv
from flask import Flask, request, Response, stream_with_context
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from openai import OpenAI
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Setting up logging
logging.basicConfig(filename='travel_bot.log', level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def ensure_nltk_data():
    """Ensure necessary NLTK data is downloaded."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')


# Ensure necessary NLTK data is available
ensure_nltk_data()


class IntentDetector:
    def __init__(self):
        self.keyword_intents = {
            'package_info': ['package', 'tour', 'vacation', 'holiday', 'itinerary', 'trip', 'travel plan'],
            'flight_info': ['flight', 'airplane', 'airport', 'airline', 'depart', 'arrive'],
            'hotel_info': ['hotel', 'accommodation', 'stay', 'room', 'book', 'reservation']
        }
        self.threshold = 0.3
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()


    def clean_token(self, token):
        # Remove non-alphanumeric characters from the beginning and end of the token
        cleaned = re.sub(r'^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$', '', token)

        # If the cleaned token is numeric, return it as is
        if cleaned.isnumeric():
            return cleaned

        # For non-numeric tokens, remove any remaining non-alphabetic characters
        return re.sub(r'[^a-zA-Z]', '', cleaned)


    def preprocess_text(self, text):
        # Tokenize and convert to lowercase
        tokens = word_tokenize(text.lower())

        # Clean tokens
        tokens = [self.clean_token(token) for token in tokens]

        # Remove empty strings and stop words, then lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens
                  if token and token not in self.stop_words]

        return tokens

    def keyword_match(self, query):
        preprocessed_query = self.preprocess_text(query)
        intent_scores = Counter()

        for intent, keywords in self.keyword_intents.items():
            preprocessed_keywords = set(self.preprocess_text(' '.join(keywords)))
            matches = set(preprocessed_query).intersection(preprocessed_keywords)
            score = len(matches) / len(preprocessed_query) if preprocessed_query else 0
            intent_scores[intent] = score

        top_intent = intent_scores.most_common(1)[0]
        return top_intent[0] if top_intent[1] >= self.threshold else None

    def gpt_classify(self, query, openai_client):
        prompt = f"""
        Classify the following travel-related query into one of these categories:
        1. package_info: Specific travel package inquiries
        2. flight_info: Flight-related questions
        3. hotel_info: Hotel or accommodation queries
        4. general_knowledge: Any other travel-related query or general information

        Query: "{query}"

        Category:
        """
        response = openai_client.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": "You are a travel query classifier. Respond with only the category name and nothing else."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip().lower()

    def detect_intent(self, query, openai_client):
        keyword_intent = self.keyword_match(query)
        if keyword_intent:
            return keyword_intent
        else:
            return self.gpt_classify(query, openai_client)


class Database:
    def __init__(self, db_url):
        self.engine = create_engine(db_url)
        self.Session = sessionmaker(bind=self.engine)

    def execute_sql_query(self, query):
        session = self.Session()
        try:
            result = session.execute(text(query))
            columns = result.keys()
            rows = result.fetchall()
            return columns, rows
        except Exception as e:
            # Return an empty result set with columns in case of an error
            return [], []
        finally:
            session.close()


class OpenAIClient:
    def __init__(self, api_key):
        os.environ['OPENAI_API_KEY'] = api_key
        self.client = OpenAI(api_key=api_key)

    def generate_sql_query(self, message):
        prompt = \
            f"""
            Given the following user question about travel packages, generate an SQL query to fetch the relevant information from the 'travel_packages', 'flights', and 'hotels' tables from the Travel_DB database.
            Assume the database is a SQLite database. Make sure the queries are all in lowercase even if the user input is in uppercase. Use ONLY 'LIKE' for matching string columns. Do not include the table schema or any other information.

            The 'travel_packages' table has columns:
            package_id, package_name, country, city, duration_days, price_usd, package_description, package_type;
            Sample Data in the table:
            | package_id | package_name | country | city | duration_days | price_usd | package_description | package_type |
            |----|--------|-------------|----------|------|-------------|
            |1|tropical paradise|maldives|malé|7|2500|enjoy a week in the beautiful maldives with beachside resorts.|beach|


            The 'flights' table has columns:
            flight_id, airline, departure_city, arrival_city, departure_time, arrival_time, price, days_of_week;
            Logic for days_of_week:
            Weekday      Letter
            -------      ------
            Sunday       S
            Monday       M
            Tuesday      T
            Wednesday    W
            Thursday     R
            Friday       F
            Saturday     U

            Example flight flies on Monday, Sat, and Sunday - then it would be the string: "SMU" 
            
            Sample Data in the table:
            | flight_id | airline | departure_city | arrival_city | departure_time | arrival_time | price | days_of_week |
            |----|--------|-------------|----------|------|-------------|
            |1|air maldives|malé|paris|08:00|14:00|750|mwf|


            The 'hotels' table has columns:
            hotel_id, name, city, country, star_rating, price_per_night, amenities;
            Sample Data in the table:
            | hotel_id | name | city | country | star_rating | price_per_night | amenities |
            |----|--------|-------------|----------|------|-------------|
            |1|hotel paradise|malé|maldives|5|300|"pool, spa, wifi"|


            User question: "{message}"

            SQL query:
        """
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": """You are a SQL query generator for a travel agency database.
                                              Give just the query for a SQLite database. Do not include the table schema or any other information."""},
                {"role": "user", "content": prompt}
            ]
        )

        generated_query = response.choices[0].message.content.strip().replace("```sql", "").replace("```", "").strip()
        return generated_query if generated_query else None

    def query_chatgpt(self, message):
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": """You are a helpful travel assistant.
                                              Answer questions about flight times, distances between cities, and general travel information.
                                              Do not provide specific package details or prices."""},
                {"role": "user", "content": message}
            ]
        )
        return response.choices[0].message.content.strip()

    def query_chatgpt_stream(self, message):
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": """You are a helpful travel assistant.
                                              Answer questions about flight times, distances between cities, and general travel information.
                                              Do not provide specific package details or prices."""},
                {"role": "user", "content": message}
            ],
            stream=True
        )
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content


class TravelAgentChatbot:
    def __init__(self, db, openai_client):
        self.db = db
        self.openai_client = openai_client
        self.intent_detector = IntentDetector()

    def chat(self, message):
        intent = self.intent_detector.detect_intent(message, self.openai_client)
        if intent == 'package_info':
            return self.handle_package_query(message)
        elif intent == 'flight_info':
            return self.handle_flight_query(message)
        elif intent == 'hotel_info':
            return self.handle_hotel_query(message)
        else:
            return self.handle_general_knowledge(message)

    def handle_general_knowledge(self, message):
        return self.openai_client.query_chatgpt(message)

    def handle_package_query(self, message):
        sql_query = self.openai_client.generate_sql_query(message)
        if sql_query:
            columns, results = self.db.execute_sql_query(sql_query)
            response = self.format_results(columns, results)
        else:
            response = "I'm sorry, I couldn't understand your package query. Could you please rephrase it?"
        return response

    def handle_flight_query(self, message):
        sql_query = self.openai_client.generate_sql_query(message)
        if sql_query:
            columns, results = self.db.execute_sql_query(sql_query)
            response = self.format_results(columns, results)
        else:
            response = "I'm sorry, I couldn't understand your flight query. Could you please rephrase it?"
        return response

    def handle_hotel_query(self, message):
        sql_query = self.openai_client.generate_sql_query(message)
        print(sql_query)
        if sql_query:
            columns, results = self.db.execute_sql_query(sql_query)
            response = self.format_results(columns, results)
        else:
            response = "I'm sorry, I couldn't understand your hotel query. Could you please rephrase it?"
        return response

    def format_results(self, columns, results):
        if not results:
            return "I'm sorry, I couldn't find any matching results."

        response = "Here are the results I found:\n\n"
        for row in results:
            row_dict = dict(zip(columns, row))
            for col, val in row_dict.items():
                if isinstance(val, str):
                    if ' ' in val:
                        val = val.title()
                    else:
                        val = val.capitalize()
                response += f"{col.replace('_', ' ').capitalize()}: {val}\n"
            response += "\n" + "-" * 50 + "\n\n"
        return response


# Load environment variables from .env file
load_dotenv()

# Get the database URL from environment variables
db_url = os.getenv('DATABASE_URL')

if not db_url:
    logger.error("Database URL not found. Please set the DATABASE_URL environment variable in the .env file.")
    raise ValueError("Database URL not found. Please set the DATABASE_URL environment variable in the .env file.")

# Initialize components
db = Database(db_url)

# Get the OpenAI API key from environment variables
openai_api_key = os.getenv('OPENAI_API_KEY')

if not openai_api_key:
    logger.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable in the .env file.")
    raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable in the .env file.")

openai_client = OpenAIClient(openai_api_key)
chatbot = TravelAgentChatbot(db, openai_client)

app = Flask(__name__)


@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        message = data.get('message')
        if not message:
            logger.warning("No message provided in the request")
            return Response(json.dumps({"error": "No message provided"}), status=400, mimetype='application/json')

        def generate_response():
            try:
                response = chatbot.chat(message)
                for chunk in response.split():
                    yield json.dumps({"chunk": chunk}) + "\n"
            except Exception as e:
                logger.error(f"Error generating response: {str(e)}")
                yield json.dumps({"error": "An error occurred while generating the response"}) + "\n"

        logger.info(f"Processing message: {message}")
        return Response(stream_with_context(generate_response()), mimetype='application/json')
    except Exception as e:
        logger.error(f"An error occurred while processing the request: {str(e)}")
        logger.error(traceback.format_exc())
        return Response(json.dumps({"error": "An unexpected error occurred"}), status=500, mimetype='application/json')


if __name__ == '__main__':
    app.run(debug=True)
