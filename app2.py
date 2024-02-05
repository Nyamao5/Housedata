import streamlit as st
import pandas as pd
import openai
from openai import OpenAIAPI


# Function to perform data analysis
def perform_data_analysis(dataset):
    analysis_result = {
        "description": dataset.describe(),
        "null_values": dataset.isnull().sum()
    }
    return analysis_result

# Function to handle user queries
def handle_user_query(query, dataset):
    if query.lower() == "description":
        return dataset.describe().to_string()
    elif query.lower() == "null values":
        return dataset.isnull().sum().to_string()
    else:
        return "I'm sorry, I don't understand that query. You can ask for 'description' or 'null values'."

# Main Streamlit app
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    dataset = pd.read_csv(uploaded_file)

    st.title("💬 Chatbot with Data Analysis")
    st.caption("🚀 A Streamlit chatbot powered by OpenAI LLM for data analysis")

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you with the dataset?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()

        client = OpenAI(api_key=openai_api_key)
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        # Check if the user is asking for data analysis
        if "analyze" in prompt.lower():
            analysis_result = perform_data_analysis(dataset)
            response = f"Here is the data analysis:\n\nDescription:\n{analysis_result['description']}\n\nNull Values:\n{analysis_result['null_values']}"
        else:
            # Use OpenAI API for general queries
            response = client.chat.completions.create(model="gpt-3.5-turbo", messages=st.session_state.messages).choices[0].message.content

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)
else:
    st.warning("Please upload a CSV file to get started.")
