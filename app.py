import streamlit as st
from openai import OpenAI
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from nltk.tokenize import word_tokenize
import nltk
import os

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Database setup
engine = create_engine('sqlite:////Users/sayeedahmed/travelDB.db')
DBSession = sessionmaker(bind=engine)

# OpenAI API setup
os.environ['OPENAI_API_KEY']=

def is_package_query(message):
    package_keywords = ['book', 'booking', 'cost', 'package', 'price', 'tour', 'trip', 'vacation', 'holiday', 'travel plan', 'itinerary']
    tokens = word_tokenize(message.lower())
    print(tokens)
    print(any(keyword in tokens for keyword in package_keywords))
    return any(keyword in tokens for keyword in package_keywords)

def generate_sql_query(message):
    prompt = f"""
    Given the following user question about travel packages, generate an SQL query to fetch the relevant information from a 'travel_packages' table.
    The table has columns: 
    package_id, package_name, destination, city, duration_days, price_usd, package_description, travel_dates, available_slots, package_type;
    Sample Data in the table:
    | package_id | package_name | destination | city | duration_days | price_usd | package_description | travel_dates | available_slots | package_type |
    |----|--------|-------------|----------|------|-------------|
    |1|tropical paradise|maldives|malé|7|2500|enjoy a week in the beautiful maldives with beachside resorts.|2024-07-01 to 2024-07-08|20|beach|
    |2|european explorer|france, germany|paris, berlin|10|3500|explore the cultural and historical wonders of europe.|2024-08-15 to 2024-08-25|15|cultural|
    |3|safari adventure|kenya, tanzania|nairobi, arusha|14|4000|experience the thrill of african wildlife in their natural habitat.|2024-09-10 to 2024-09-24|10|adventure|
    |4|asian delights|japan, south korea|tokyo, seoul|12|3000|discover the rich traditions and modern marvels of east asia.|2024-10-01 to 2024-10-13|25|cultural|
    |5|south american escape|brazil, argentina|rio, buenos aires|10|3200|enjoy the vibrant culture and stunning landscapes of south america.|2024-11-05 to 2024-11-15|18|cultural|
    
    The package_name column contains the name of the package.
    The destination column contains the name of the destination country.
    The city column contains all the cities included in the package, separated by commas.
    The duration_days column contains the number of days the package lasts.
    The price_usd column contains the price per person in USD.
    The package_description column contains a brief description of the package.
    The travel_dates column contains the start and end dates of the package.
    The available_slots column contains the number of available slots for the package.
    The package_type column contains the type of package (e.g., beach, cultural, adventure).
    
    
    User question: "{message}"
    
    SQL query:
    """
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": """You are a SQL query generator for a travel agency database.
                                          Give just the query for a SQLlite database. Do not include the table schema or any other information."""},
            {"role": "user", "content": prompt}
        ]
    )

    generated_query = response.choices[0].message.content.strip().replace("```sql", "").replace("```", "").strip()
    print(generated_query)
    if generated_query:
        return generated_query
    else:
        return None

def execute_sql_query(query):
    session = DBSession()
    try:
        result = session.execute(text(query))
        return result.fetchall()
    except Exception as e:
        st.error(f"Error executing SQL query: {e}")
        return []
    finally:
        session.close()

def format_package_results(results):
    if not results:
        return "I'm sorry, I couldn't find any travel packages matching your criteria."
    print(results)
    def capitalize_words(text):
        return ' '.join(word.capitalize() for word in text.split())

    response = "Here are the travel packages I found:\n\n"
    for package in results:
        package_name, destination, city, duration_days, price_usd, package_description, travel_dates, available_slots, package_type = package[1:]
        response += f"Package Name: {capitalize_words(package_name)}\n"
        response += f"Countires: {capitalize_words(destination)}\n"
        response += f"Cities: {capitalize_words(city)}\n"
        response += f"Duration: {duration_days} days\n"
        response += f"Price: ${price_usd:.2f}\n"
        response += f"Description: {capitalize_words(package_description)}\n"
        response += f"Travel Dates: {capitalize_words(travel_dates)}\n"
        response += f"Available Slots: {available_slots}\n"
        response += f"Package Type: {capitalize_words(package_type)}\n"
        response += "\n" + "-"*50 + "\n\n"
    print(response)
    return response


def query_chatgpt(message):
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": """You are a helpful travel assistant.
                                          Answer questions about flight times, distances between cities, and general travel information.
                                          Do not provide specific package details or prices."""},
            {"role": "user", "content": message}
        ]
    )
    print(response.choices[0].message.content.strip())
    return response.choices[0].message.content.strip()
def handle_package_query(message):
    sql_query = generate_sql_query(message)
    print(sql_query)
    if sql_query:
        results = execute_sql_query(sql_query)
        response = format_package_results(results)
    else:
        response = "I'm sorry, I couldn't understand your package query. Could you please rephrase it?"
    print(response)
    return response

def chat(message):
    if is_package_query(message):
        return handle_package_query(message)
    else:
        return query_chatgpt(message)

# Streamlit UI
st.title("Travel Agent Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What would you like to know?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = chat(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
