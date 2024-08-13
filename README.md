## Multi-Function Chat App
# Overview
The Multi-Function Chat App is a versatile web application built using Streamlit, integrating advanced AI models for various functionalities. It allows users to interact with text, PDFs, websites, and images in multiple ways, such as language translation, summarization, and content analysis.

## Features
#Language Translator: 
Translate your text or content from a PDF document into multiple languages.
# PDF Summarizer: Upload a PDF document and receive a concise summary of its content.
# Chat with Website: 
Input a website URL and ask questions to get answers directly from the website's content.
# Image Analysis: 
Upload an image and ask questions about it to receive AI-generated insights.
# Chat with PDF: 
Upload PDFs and interact with them by asking questions about their content.
# Installation
To set up the application locally, follow these steps:

# Clone the repository:

git clone https://github.com/your-username/multi-function-chat-app.git
cd multi-function-chat-app
# Install the required dependencies:

pip install -r requirements.txt
Running the Application
Start the Streamlit app by running the following command:

streamlit run app.py
This will open the application in your default web browser, where you can explore its various features.

# Environment Variables
To enable the Google Generative AI functionalities, set the following environment variable:


export GOOGLE_API_KEY='your-google-api-key'
Replace 'your-google-api-key' with your actual Google API key.

# Usage
# Language Translator:
Select "Language Translator" from the sidebar.
Input your text or upload a PDF.
Choose the target language and click "Translate."
# PDF Summarizer:
Select "PDF Summarizer" from the sidebar.
Upload a PDF and click "Summarize" to get a summary of the document.
# Chat with Website:
Select "Chat with Website" from the sidebar.
Enter the website URL and ask questions about its content.
# Image Analysis:
Select "Image Analysis" from the sidebar.
Upload an image and interact with it by asking questions.
# Chat with PDF:
Select "Chat with PDF" from the sidebar.
Upload PDFs and ask questions to receive responses based on the content.
