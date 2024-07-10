import base64
import json

import requests
import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Travel Agent Chatbot",
    page_icon="✈️",
    layout="wide"
)

# Custom CSS to improve the appearance
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&family=Roboto:wght@300;400&display=swap');
    
    body {
        font-family: 'Roboto', sans-serif;
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: #ffffff;
    }
    .stApp {
        max-width: 1000px;
        margin: 0 auto;
    }
    .custom-header {
        background: rgba(30, 60, 114, 0.9);
        padding: 10px 0;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 20px;
    }
    .header {
        display: flex;
        align-items: center;
        max-width: 1000px;
        width: 100%;
        padding-left: 20px;
    }
    .header img {
        width: 50px;
        margin-right: 20px;
    }
    .header h1 {
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        color: #ffffff;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .chat-container {
        padding: 20px;
    }
    .stTextInput > div > div > input {
        background-color: rgba(255,255,255,0.1);
        color: #ffffff;
        border: none;
        border-radius: 20px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stTextInput > div > div > input::placeholder {
        color: rgba(255,255,255,0.5);
    }
    .package-info {
        margin-bottom: 15px;
        border-bottom: 1px solid rgba(255,255,255,0.2);
        padding-bottom: 15px;
    }
    .package-info:last-child {
        border-bottom: none;
    }
    </style>
    """, unsafe_allow_html=True)

def format_response(response):
    if "Here are the results I found:" in response:
        packages = response.split("Here are the results I found:")[1].strip().split("--------------------------------------------------")
        formatted_output = "Here are the results I found:\n\n"
        for package in packages:
            if package.strip():
                package_info = package.strip().split('\n')
                formatted_output += "\n".join(package_info)
                formatted_output += "\n\n"
        return formatted_output.strip()
    return response

# Function to load and encode the image
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

logo_base64 = get_base64_of_bin_file('logo.webp')

custom_header = st.container()
with custom_header:
    st.markdown(f"""
    <div class="custom-header">
        <div class="header">
            <img src="data:image/webp;base64,{logo_base64}" alt="Logo">
            <h1>Travel Agent Chatbot</h1>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What would you like to know about your travels?"):
    # Add user message to chat history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        try:
            # Send request to Flask API
            with requests.post("http://127.0.0.1:5000/chat", json={"message": prompt}, stream=True) as r:
                r.raise_for_status()  # Raise an exception for bad status codes
                for chunk in r.iter_lines():
                    if chunk:
                        chunk_data = json.loads(chunk)
                        if 'error' in chunk_data:
                            full_response = f"I'm sorry, but an error occurred: {chunk_data['error']}"
                            break
                        full_response += chunk_data['chunk'] + " "
                        formatted_response = format_response(full_response)
                        message_placeholder.text(formatted_response)
        except requests.exceptions.RequestException as e:
            full_response = f"I'm sorry, but I'm having trouble connecting to my knowledge base right now. Error: {str(e)}"
        except json.JSONDecodeError:
            full_response = "I apologize, but I received an unexpected response format. Please try again."
        except Exception as e:
            full_response = f"An unexpected error occurred: {str(e)}"

        formatted_response = format_response(full_response)
        message_placeholder.markdown(formatted_response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})