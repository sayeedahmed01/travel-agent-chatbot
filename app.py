import streamlit as st
import requests

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

    try:
        # Send request to Flask API
        response = requests.post("http://127.0.0.1:5000/chat", json={"message": prompt})
        response.raise_for_status()  # Check if the request was successful
        response_data = response.json()
        api_response = response_data.get("response", "I'm sorry, I couldn't process your request at the moment.")
    except requests.exceptions.RequestException as e:
        api_response = f"Error: {e}"
    except ValueError:
        api_response = "Error: Received an invalid response from the API."

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(api_response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": api_response})
