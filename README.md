# Travel Agent Chatbot

This is a travel agent chatbot that interacts with users to provide travel-related information, 
such as distances or flight times between two cities and travel package options. 
The chatbot uses a combination of direct queries and a custom Query-SQL prompts to ChatGPT to fetch travel package information from a database.

## Requirements

- Python 3.8 or above
- An OpenAI API key

## Installation

1. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

2. Create a `.env` file in the root directory and add your OpenAI API key:
    ```env
    OPENAI_API_KEY=your_openai_api_key
    ```

   Alternatively, you can set the OpenAI API key directly in the code (not recommended for security reasons). Open `travel_bot.py` and replace the placeholder with your actual API key:
    ```python
    openai_api_key = "your_openai_api_key"
    ```

## Running the Application

1. Start the Flask backend server:
    ```sh
    python travel_bot.py
    ```

2. In a new terminal, start the Streamlit frontend application:
    ```sh
    streamlit run app.py
    ```

3. Open your web browser and go to `http://localhost:8501` to access the chatbot interface.

## Usage

- Enter your travel-related questions in the input box provided in the web interface.
- The chatbot will respond with the relevant information, whether it's from ChatGPT for general knowledge queries or from the travel agent database for specific package information.

## Logging

- All interactions and errors are logged in the `travel_bot.log` file.

## Sample Dataset

- The `TravelDB` directory contains the database used for querying travel packages. Ensure that it is in the correct format and has the necessary tables (`travel_packages`, `flights`, `hotels`).

