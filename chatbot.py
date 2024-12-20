import os
import boto3
import tempfile
import botocore
import streamlit as st
from langchain_community.llms import Bedrock
from langchain.chains import ConversationChain
from langchain.schema import Document
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import (
    CSVLoader,
    TextLoader,
    UnstructuredExcelLoader,
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredImageLoader
)

# Initialize session states
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "model_kwargs" not in st.session_state:
    st.session_state.model_kwargs = {}

# Model mapping dictionary with friendly names
MODEL_MAPPING = {
    # Anthropic Models
    "anthropic.claude-3-sonnet-20240229-v1:0": "Anthropic - Claude 3 Sonnet",
    "anthropic.claude-3-haiku-20240307-v1:0": "Anthropic - Claude 3 Haiku",
    "anthropic.claude-3-opus-20240229-v1:0": "Anthropic - Claude 3 Opus",
    "anthropic.claude-3-5-sonnet-20240620-v1:0": "Anthropic - Claude 3.5 Sonnet",
    "anthropic.claude-3-5-sonnet-20241022-v2:0": "Anthropic - Claude 3.5 Sonnet v2",
    "anthropic.claude-3-5-haiku-20241022-v1:0": "Anthropic - Claude 3.5 Haiku",
    "anthropic.claude-instant-v1": "Anthropic - Claude Instant",
    "anthropic.claude-v2:1": "Anthropic - Claude v2.1",

    # Amazon Models
    "amazon.titan-text-lite-v1": "Amazon - Titan Text Lite",
    "amazon.titan-text-express-v1": "Amazon - Titan Text Express",
    "amazon.titan-text-premier-v1:0": "Amazon - Titan Text Premier",
    "amazon.titan-image-generator-v1": "Amazon - Titan Image Generator",
    "amazon.nova-pro-v1:0": "Amazon - Nova Pro",
    "amazon.nova-lite-v1:0": "Amazon - Nova Lite",
    "amazon.nova-canvas-v1:0": "Amazon - Nova Canvas",
    "amazon.nova-reel-v1:0": "Amazon - Nova Reel",
    "amazon.nova-micro-v1:0": "Amazon - Nova Micro",

    # Cohere Models
    "cohere.command-text-v14": "Cohere - Command",
    "cohere.command-light-text-v14": "Cohere - Command Light",
    "cohere.command-r-v1:0": "Cohere - Command R",
    "cohere.command-r-plus-v1:0": "Cohere - Command R+",

    # Meta Models
    "meta.llama3-8b-instruct-v1:0": "Meta - Llama 3 8B Instruct",
    "meta.llama3-70b-instruct-v1:0": "Meta - Llama 3 70B Instruct",
    "meta.llama3-1-8b-instruct-v1:0": "Meta - Llama 3.1 8B Instruct",
    "meta.llama3-1-70b-instruct-v1:0": "Meta - Llama 3.1 70B Instruct",
    "meta.llama3-2-1b-instruct-v1:0": "Meta - Llama 3.2 1B Instruct",
    "meta.llama3-2-3b-instruct-v1:0": "Meta - Llama 3.2 3B Instruct",
    "meta.llama3-2-11b-instruct-v1:0": "Meta - Llama 3.2 11B Instruct",
    "meta.llama3-2-90b-instruct-v1:0": "Meta - Llama 3.2 90B Instruct",
    "meta.llama3-3-70b-instruct-v1:0": "Meta - Llama 3.3 70B Instruct",

    # Mistral AI Models
    "mistral.mistral-7b-instruct-v0:2": "Mistral AI - Mistral 7B Instruct",
    "mistral.mixtral-8x7b-instruct-v0:1": "Mistral AI - Mixtral 8x7B Instruct",
    "mistral.mistral-large-2402-v1:0": "Mistral AI - Mistral Large",
    "mistral.mistral-small-2402-v1:0": "Mistral AI - Mistral Small"
}


def get_model_friendly_name(model_id):
    """Get friendly name for a model ID."""
    return MODEL_MAPPING.get(model_id, model_id)


def get_model_id_from_friendly_name(friendly_name):
    """Get model ID from friendly name."""
    for model_id, name in MODEL_MAPPING.items():
        if name == friendly_name:
            return model_id
    return friendly_name


def get_bedrock_models():
    """Get available Bedrock models."""
    try:
        # Try to create a Bedrock client
        bedrock = boto3.client(
            service_name='bedrock',
            region_name='us-east-1'
        )
        response = bedrock.list_foundation_models()
        models = [model['modelId'] for model in response['modelSummaries']]

        # Convert to friendly names and sort
        friendly_models = [get_model_friendly_name(model) for model in models]
        friendly_models.sort()
        return friendly_models
    except (botocore.exceptions.UnknownServiceError, botocore.exceptions.ClientError) as e:
        try:
            # Try bedrock-runtime as fallback
            bedrock = boto3.client(
                service_name='bedrock-runtime',
                region_name='us-east-1'
            )
            # If we can create a client but can't list models, use default list
            st.warning("Unable to list models. Using default model list.")
            friendly_models = [get_model_friendly_name(
                model) for model in MODEL_MAPPING.keys()]
            friendly_models.sort()
            return friendly_models
        except Exception as e:
            st.error(f"Error connecting to Bedrock: {str(e)}")
            st.warning("Using default model list.")
            friendly_models = [get_model_friendly_name(
                model) for model in MODEL_MAPPING.keys()]
            friendly_models.sort()
            return friendly_models


def load_document(file):
    """Load document based on file type."""
    if file is None:
        return None

    file_extension = os.path.splitext(file.name)[1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
        tmp_file.write(file.getvalue())
        file_path = tmp_file.name

    try:
        if file_extension == '.pdf':
            loader = PyPDFLoader(file_path)
        elif file_extension == '.csv':
            loader = CSVLoader(file_path)
        elif file_extension == '.txt':
            loader = TextLoader(file_path)
        elif file_extension in ['.xlsx', '.xls']:
            loader = UnstructuredExcelLoader(file_path)
        elif file_extension == '.docx':
            loader = Docx2txtLoader(file_path)
        elif file_extension in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
            loader = UnstructuredImageLoader(file_path)
        elif file_extension in ['.mp4', '.avi', '.mov', '.wmv']:
            # For video files, return a simple description as we can't process them directly
            return [Document(page_content=f"Video file uploaded: {file.name}")]
        else:
            st.error(f"Unsupported file type: {file_extension}")
            return None

        documents = loader.load()
        os.unlink(file_path)
        return documents

    except Exception as e:
        st.error(f"Error loading document: {str(e)}")
        if os.path.exists(file_path):
            os.unlink(file_path)
        return None


def process_uploaded_content(documents, file_extension):
    """Process uploaded content based on file type."""
    if documents:
        if file_extension in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
            return "Image content: " + "\n".join([doc.page_content for doc in documents])
        elif file_extension in ['.mp4', '.avi', '.mov', '.wmv']:
            return "Video content: " + "\n".join([doc.page_content for doc in documents])
        else:
            return "\n".join([doc.page_content for doc in documents])
    return None


def initialize_conversation(friendly_model_name, system_prompt, model_kwargs):
    """Initialize or reinitialize the conversation with new settings."""
    try:
        # Convert friendly name back to model ID
        model_id = get_model_id_from_friendly_name(friendly_model_name)

        # Try bedrock-runtime first
        client = boto3.client('bedrock-runtime', region_name='us-east-1')
    except:
        try:
            # Try bedrock as fallback
            client = boto3.client('bedrock', region_name='us-east-1')
        except Exception as e:
            st.error(f"Failed to initialize Bedrock client: {str(e)}")
            return None

    try:
        llm = Bedrock(
            model_id=model_id,
            model_kwargs=model_kwargs,
            client=client
        )

        memory = ConversationBufferMemory()
        conversation = ConversationChain(
            llm=llm,
            memory=memory,
            verbose=True
        )

        if system_prompt:
            conversation.predict(input=f"System: {system_prompt}")

        return conversation

    except Exception as e:
        st.error(f"Error initializing conversation: {str(e)}")
        return None


def main():
    st.title("🤖 Neil'sChatbot")

    # Sidebar settings
    with st.sidebar:
        st.header("Settings")

        # Model selection with friendly names and default selection
        friendly_models = get_bedrock_models()
        default_index = friendly_models.index(
            "Anthropic - Claude 3.5 Sonnet v2") if "Anthropic - Claude 3.5 Sonnet v2" in friendly_models else 0
        selected_friendly_model = st.selectbox(
            "Select Model",
            friendly_models,
            index=default_index,
            help="Select the AI model you want to chat with"
        )

        # Display model ID for reference
        model_id = get_model_id_from_friendly_name(selected_friendly_model)
        st.caption(f"Model ID: {model_id}")

        # System prompt
        system_prompt = st.text_area(
            "System Prompt",
            "You are a helpful AI assistant."
        )

        # Model parameters
        st.subheader("Model Parameters")
        temperature = st.slider(
            "Temperature",
            0.0, 1.0, 1.0,
            help="Higher values make the output more random, lower values make it more focused"
        )
        max_tokens = st.number_input(
            "Max Tokens",
            1, 4096, 4096,
            help="Maximum number of tokens in the response"
        )

        # Update model parameters
        model_kwargs = {
            "temperature": temperature,
            "max_tokens_to_sample": max_tokens,
        }

        # File upload with expanded file types
        uploaded_file = st.file_uploader(
            "Upload Document",
            type=['pdf', 'csv', 'txt', 'xlsx', 'xls', 'docx',
                  'jpg', 'jpeg', 'png', 'gif', 'bmp',
                  'mp4', 'avi', 'mov', 'wmv']
        )

        # Initialize/Reinitialize button
        if st.button("Initialize/Reinitialize Chat"):
            st.session_state.conversation = initialize_conversation(
                selected_friendly_model,
                system_prompt,
                model_kwargs
            )
            st.session_state.messages = []
            if st.session_state.conversation:
                st.success("Chat initialized!")
    # Main chat interface
    if st.session_state.conversation is None:
        st.info("Please initialize the chat using the sidebar settings.")
        return

    # Process uploaded file
    if uploaded_file:
        documents = load_document(uploaded_file)
        if documents:
            file_extension = os.path.splitext(uploaded_file.name)[1].lower()
            document_content = process_uploaded_content(
                documents, file_extension)
            st.session_state.conversation.predict(
                input=f"Please process this content: {document_content}"
            )
            st.success(f"{uploaded_file.name} processed!")

    # Chat interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What would you like to know?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response = st.session_state.conversation.predict(input=prompt)
            st.markdown(response)
            st.session_state.messages.append(
                {"role": "assistant", "content": response}
            )


if __name__ == "__main__":
    main()
