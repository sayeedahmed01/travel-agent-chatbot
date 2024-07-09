import os
import nltk
from nltk.tokenize import word_tokenize
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from openai import OpenAI
from flask import Flask, request, jsonify
from dotenv import load_dotenv

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')


class Database:
    def __init__(self, db_url):
        self.engine = create_engine(db_url)
        self.Session = sessionmaker(bind=self.engine)

    def execute_sql_query(self, query):
        session = self.Session()
        try:
            result = session.execute(text(query))
            return result.fetchall()
        except Exception as e:
            return str(e)
        finally:
            session.close()

class OpenAIClient:
    def __init__(self, api_key):
        os.environ['OPENAI_API_KEY'] = api_key
        self.client = OpenAI(api_key=api_key)

    def generate_sql_query(self, message):
        prompt = f"""
        Given the following user question about travel packages, generate an SQL query to fetch the relevant information from the 'travel_packages', 'flights', and 'hotels' tables.
        
        The 'travel_packages' table has columns: 
        package_id, package_name, destination, city, duration_days, price_usd, package_description, travel_dates, available_slots, package_type;
        Sample Data in the table:
        | package_id | package_name | destination | city | duration_days | price_usd | package_description | travel_dates | available_slots | package_type |
        |----|--------|-------------|----------|------|-------------|
        |1|tropical paradise|maldives|malé|7|2500|enjoy a week in the beautiful maldives with beachside resorts.|2024-07-01 to 2024-07-08|20|beach|

        The 'flights' table has columns: 
        flight_id, airline, departure_city, arrival_city, departure_time, arrival_time, price, available_seats;
        Sample Data in the table:
        | flight_id | airline | departure_city | arrival_city | departure_time | arrival_time | price | available_seats |
        |----|--------|-------------|----------|------|-------------|
        |1|Air Maldives|Malé|Paris|08:00|14:00|750|150|

        The 'hotels' table has columns:
        hotel_id, name, city, country, star_rating, price_per_night, amenities, available_rooms;
        Sample Data in the table:
        | hotel_id | name | city | country | star_rating | price_per_night | amenities | available_rooms |
        |----|--------|-------------|----------|------|-------------|
        |1|Hotel Paradise|Malé|Maldives|5|300|"Pool, Spa, WiFi"|20|

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

class TravelAgentChatbot:
    def __init__(self, db, openai_client):
        self.db = db
        self.openai_client = openai_client

    def is_package_query(self, message):
        package_keywords = ['book', 'booking', 'cost', 'package', 'price', 'tour', 'trip', 'vacation', 'holiday', 'travel plan', 'itinerary']
        tokens = word_tokenize(message.lower())
        return any(keyword in tokens for keyword in package_keywords)

    def format_package_results(self, results):
        if not results:
            return "I'm sorry, I couldn't find any travel packages matching your criteria."

        def capitalize_words(text):
            return ' '.join(word.capitalize() for word in text.split())

        response = "Here are the travel packages I found:\n\n"
        for package in results:
            package_name, destination, city, duration_days, price_usd, package_description, travel_dates, available_slots, package_type = package[1:]
            response += f"Package Name: {capitalize_words(package_name)}\n"
            response += f"Countries: {capitalize_words(destination)}\n"
            response += f"Cities: {capitalize_words(city)}\n"
            response += f"Duration: {duration_days} days\n"
            response += f"Price: ${price_usd:.2f}\n"
            response += f"Description: {capitalize_words(package_description)}\n"
            response += f"Travel Dates: {capitalize_words(travel_dates)}\n"
            response += f"Available Slots: {available_slots}\n"
            response += f"Package Type: {capitalize_words(package_type)}\n"
            response += "\n" + "-"*50 + "\n\n"
        return response

    def handle_package_query(self, message):
        sql_query = self.openai_client.generate_sql_query(message)
        if sql_query:
            results = self.db.execute_sql_query(sql_query)
            response = self.format_package_results(results)
        else:
            response = "I'm sorry, I couldn't understand your package query. Could you please rephrase it?"
        return response

    def chat(self, message):
        if self.is_package_query(message):
            return self.handle_package_query(message)
        else:
            return self.openai_client.query_chatgpt(message)

# Initialize components
db = Database('sqlite:////Users/sayeedahmed/travelDB.db')
# Load environment variables from .env file
load_dotenv()

# Get the OpenAI API key from environment variables
openai_api_key = os.getenv('OPENAI_API_KEY')

if not openai_api_key:
    raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable in the .env file.")

openai_client = OpenAIClient(openai_api_key)
chatbot = TravelAgentChatbot(db, openai_client)

# Flask API setup
app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get('message')
    if not message:
        return jsonify({"error": "No message provided"}), 400
    response = chatbot.chat(message)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)
