import streamlit as st
import google.generativeai as gen_ai
import os
import time
import openai

select_api = st.selectbox("Select API", [" ","Gemini-Pro", "GPT-3.5"])

if select_api == "Gemini-Pro":
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
elif select_api == "GPT-3.5":
    OPENAI_API_KEY = 'sk-proj-XMEp1dvnmPaTm80zJ160T3BlbkFJ4S3OAYVIh5YZy57NAnVu'
    openai.api_key = OPENAI_API_KEY

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.title("🤖 GPT-3.5 - ChatBot")

    # display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


    user_prompt = st.chat_input("Ask GPT-3.5...")

    if user_prompt:
        st.chat_message("user").markdown(user_prompt)
        st.session_state.chat_history.append({"role": "user", "content": user_prompt})

        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                *st.session_state.chat_history
            ]
        )

        assistant_response = response.choices[0].message.content
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

        # display GPT-4o's response
        with st.chat_message("assistant"):
            st.markdown(assistant_response)