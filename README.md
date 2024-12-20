# Neilflix's Bedrock Chatbot

A Streamlit-based chatbot interface for Amazon Bedrock models.

## Features
- Multiple model selection from Amazon Bedrock
- Adjustable model parameters
- File upload support (PDF, DOCX, CSV, TXT, Images, etc.)
- Conversation memory
- Customizable system prompts

## Setup
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Configure AWS credentials
4. Run the app: `streamlit run chatbot.py`

## Deployment
This app is deployed on Streamlit Community Cloud. Visit [your-app-url] to use it.

## Configuration
Configure AWS credentials in Streamlit Cloud:
1. Go to your app settings
2. Add the following secrets:
   - AWS_ACCESS_KEY_ID
   - AWS_SECRET_ACCESS_KEY
   - AWS_DEFAULT_REGION
