import streamlit as st
from openai import OpenAI
import tiktoken
from dotenv import load_dotenv
import os
import json
from datetime import datetime

# Load environment variables
load_dotenv()


# Function to count tokens
def count_tokens(text, model="gpt-4"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


# Function to save chat history
def save_chat_history():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_filename = f"chat_history_{timestamp}.json"
    md_filename = f"chat_history_{timestamp}.md"

    # Save as JSON
    with open(json_filename, "w") as f:
        json.dump(st.session_state.messages, f)

    # Save as Markdown
    with open(md_filename, "w") as f:
        for message in st.session_state.messages:
            f.write(f"**{message['role'].capitalize()}:** {message['content'].encode()}\n\n")

    return json_filename, md_filename


# Function to load chat history
def load_chat_history(filename):
    with open(filename, "r") as f:
        loaded_messages = json.load(f)
    return loaded_messages


# ---------------------------
# Initialize Session State
# ---------------------------
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'token_count' not in st.session_state:
    st.session_state.token_count = 0
if 'api_key' not in st.session_state:
    st.session_state.api_key = os.getenv("OPENAI_API_KEY", "")
if 'model' not in st.session_state:
    st.session_state.model = 'gpt-4o-mini'
if 'temperature' not in st.session_state:
    st.session_state.temperature = 0.1
if 'max_tokens' not in st.session_state:
    st.session_state.max_tokens = 4096
if 'history_window' not in st.session_state:
    st.session_state.history_window = 10

# ---------------------------
# Sidebar Configuration
# ---------------------------
st.sidebar.header("OpenAI Configuration")

# API Key Input
st.session_state.api_key = st.sidebar.text_input(
    "OpenAI API Key",
    type="password",
    value=st.session_state.api_key,
    help="Enter your OpenAI API key."
)

# Model Selection
available_models = ["gpt-4o-mini", "gpt-4o", "o1-mini", "o1-preview"]
st.session_state.model = st.sidebar.selectbox(
    "Select Model",
    available_models,
    index=available_models.index(st.session_state.model),
    help="Choose the OpenAI model to use for chatting."
)

# Temperature Slider
st.session_state.temperature = st.sidebar.slider(
    "Temperature",
    min_value=0.0,
    max_value=1.0,
    value=st.session_state.temperature,
    step=0.1,
    help="Controls the randomness of the model's output."
)

# Max Tokens Input
st.session_state.max_tokens = st.sidebar.number_input(
    "Max Tokens",
    min_value=100,
    max_value=4096,
    value=st.session_state.max_tokens,
    step=100,
    help="The maximum number of tokens to generate in the response."
)

# History Window Slider
st.session_state.history_window = st.sidebar.slider(
    "History Window",
    min_value=1,
    max_value=20,
    value=st.session_state.history_window,
    step=1,
    help="Number of recent messages to include in the context."
)

# ---------------------------
# Set OpenAI API Key
# ---------------------------
if st.session_state.api_key:
    client = OpenAI(api_key=st.session_state.api_key)
else:
    st.sidebar.warning("⚠️ Please enter your OpenAI API key to use the chat.")

# ---------------------------
# Main Chat Interface
# ---------------------------
st.title(f"🗨️ Chat with {st.session_state.model}")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What would you like to ask?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # Prepare the base parameters
        completion_params = {
            "model": st.session_state.model,
            "messages": [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[-st.session_state.history_window:]
            ],
        }

        # Add parameters based on model type
        if st.session_state.model.startswith("o1"):
            # For o1 models, don't use streaming
            with st.spinner("Thinking..."):
                response = client.chat.completions.create(**completion_params)
                full_response = response.choices[0].message.content
            message_placeholder.markdown(full_response)
        else:
            # For non-o1 models, use streaming and include temperature and max_tokens
            completion_params.update({
                "stream": True,
                "temperature": st.session_state.temperature,
                "max_tokens": st.session_state.max_tokens
            })
            for response in client.chat.completions.create(**completion_params):
                full_response += (response.choices[0].delta.content or "")
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
    st.session_state.token_count = sum(count_tokens(m["content"]) for m in st.session_state.messages)


# ---------------------------
# Display Token Count and Usage
# ---------------------------
st.sidebar.markdown("----")
st.sidebar.write(f"**Total Messages:** {len(st.session_state.messages)}")
st.sidebar.write(f"**Total Tokens:** {st.session_state.token_count}")

# Save Chat History Button
if st.sidebar.button("Save Chat History"):
    json_filename, md_filename = save_chat_history()
    st.sidebar.success(f"Chat history saved as {json_filename} and {md_filename}")

# Load Chat History Button
uploaded_file = st.sidebar.file_uploader("Load Chat History", type="json")
if uploaded_file is not None:
    loaded_messages = json.load(uploaded_file)
    st.session_state.messages = loaded_messages
    st.session_state.token_count = sum(count_tokens(m["content"]) for m in st.session_state.messages)
    st.sidebar.success("Chat history loaded successfully!")
    st.rerun()

# Optional: Reset Chat History
if st.sidebar.button("Reset Conversation"):
    st.session_state.messages = []
    st.session_state.token_count = 0
    st.rerun()
