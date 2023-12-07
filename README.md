# TL;DR Bot - Chat with your documents

A chatbot powered by LangCHain that augments GPT 3.5 with the contents of your own documents.

## Overview of the App

<img src="">

- Takes user queries via Streamlit's `st.chat_input` and displays both user queries and model responses with `st.chat_message`
- Uses RAG LangChain to load and index data and create a chat engine that will retrieve context from that data to respond to each user query

## Demo App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://llm-for-good-aghutwvnl6suwgsz683vst.streamlit.app/)

## Get an OpenAI API key

You can get your own OpenAI API key by following the following instructions:
1. Go to https://platform.openai.com/account/api-keys.
2. Click on the `+ Create new secret key` button.
3. Next, enter an identifier name (optional) and click on the `Create secret key` button.

## Try out the app

Once the app is loaded, enter your question about your document, enter your OpenAI API key and wait for a response.
