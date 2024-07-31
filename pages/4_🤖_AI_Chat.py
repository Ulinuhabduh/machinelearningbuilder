import streamlit as st
import google.generativeai as gen_ai
import os
import time
import openai


GOOGLE_API_KEY = "AIzaSyDAcIxULry0OG0JrRPRypnl8hEWfNQrcw4"

    # Set up Google Gemini-Pro AI model
gen_ai.configure(api_key=GOOGLE_API_KEY)
model = gen_ai.GenerativeModel('gemini-pro')


def translate_role_for_streamlit(user_role):
    if user_role == "model":
        return "assistant"
    else:
        return user_role


if "chat_session" not in st.session_state:
    st.session_state.chat_session = model.start_chat(history=[])


st.title("🤖 Gemini Pro - ChatBot")

for message in st.session_state.chat_session.history:
    with st.chat_message(translate_role_for_streamlit(message.role)):
        st.markdown(message.parts[0].text)

# Input field for user's message
user_prompt = st.chat_input("Ask Gemini-Pro...")
if user_prompt:
    st.chat_message("user").markdown(user_prompt)

    # Create a placeholder for the assistant's response
    assistant_placeholder = st.chat_message("assistant")
    assistant_text_placeholder = assistant_placeholder.empty()

    gemini_response = st.session_state.chat_session.send_message(user_prompt)

    response_text = gemini_response.text
    typing_speed = 0.005  
        
    for i in range(len(response_text) + 1):
        current_text = response_text[:i]
        assistant_text_placeholder.markdown(current_text)
        time.sleep(typing_speed)
