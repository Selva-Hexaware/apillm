from fastapi import FastAPI, UploadFile, File
import io
import os
from fastapi.middleware.cors import CORSMiddleware
import openai

from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.prompts.chat import(
 ChatPromptTemplate,
 HumanMessagePromptTemplate,
)
from azure.cosmos import CosmosClient, PartitionKey, exceptions

import pandas as pd
import numpy as np
import re
import openai
import os
import re
import requests
import sys
import tiktoken

from num2words import num2words

from openai.embeddings_utils import get_embedding, cosine_similarity

app = FastAPI()

# Allow all origins, methods, and headers (not recommended in production)
app.add_middleware(
CORSMiddleware,
allow_origins=["*"],
allow_credentials=True,
allow_methods=["*"],
allow_headers=["*"],
)

@app.post("/speech-to-text")
async def speech_to_text(audio: UploadFile = File(...)):
    openai.api_key = "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" 
    contents = await audio.read()

    with open(audio.filename, "wb") as f:
        f.write(contents)

    with open(audio.filename, 'rb') as file:
        transcript = openai.Audio.transcribe("whisper-1", file)

    os.remove(audio.filename)

    return {"transcript": transcript.text}


def embed_model_contents(model):
    def query_cosmos(cosmos_url, cosmos_key, database_name, container_name):
        client = CosmosClient(cosmos_url, cosmos_key)
        database = client.get_database_client(database_name)
        container = database.get_container_client(container_name)

        query = f"SELECT * FROM modelprompt WHERE modelprompt.Model = '{model}'"
        options = {
            'enableCrossPartitionQuery': True,
            'maxItemCount': -1
        }
        print(query)

        result_iterable = container.query_items(
            query=query,
            feed_options=options
        )

        items = []

        for result_list in result_iterable.by_page(continuation_token=None):
            for result in result_list:
                if 'content' in result:
                    items.append(result)
        return items, container

    cosmos_url = "https://cosmos-hexainsights.documents.azure.com:443/"
    cosmos_key = "HJppOUqtKwDhvRpPi7ys3FIXEbKCXCItgfWuV4ZKsWa49nyZ1b1qwlXqbbUrZiL63JVzbvHcYH3JACDb2afTdQ=="
    database_name = "hexainsightsdb"
    container_name = "modelprompt"

    items, container = query_cosmos(cosmos_url, cosmos_key, database_name, container_name)

    API_KEY = "f700af5638614eed9c63d85695011af9"
    RESOURCE_ENDPOINT = "https://rapidxopenai.openai.azure.com/"

    openai.api_type = "azure"
    openai.api_key = API_KEY
    openai.api_base = RESOURCE_ENDPOINT
    openai.api_version = "2022-12-01"

    def normalize_text(s, sep_token = " \n "):
        s = re.sub(r'\s+', ' ', s).strip()
        s = re.sub(r". ,","",s)
        s = s.replace("..",".")
        s = s.replace(". .",".")
        s = s.replace("\n", "")
        s = s.strip()
        return s

    tokenizer = tiktoken.get_encoding("cl100k_base")

    for item in items:
        if 'content' in item:
            content = item['content']
            normalized_text = normalize_text(content)
            n_tokens = len(tokenizer.encode(normalized_text))

        if n_tokens < 8192:
            embedding = get_embedding(normalized_text, engine="ada2embed")
            print(type(embedding))
            item['embedding'] = embedding 
            container.upsert_item(item)


def run_llm(question, model):
    def query_cosmos(cosmos_url, cosmos_key, database_name, container_name, model):
        client = CosmosClient(cosmos_url, cosmos_key)
        database = client.get_database_client(database_name)
        container = database.get_container_client(container_name)
        

        query = f"SELECT * FROM modelprompt WHERE modelprompt.Model = '{model}'"
        options = {
            'enableCrossPartitionQuery': True,
            'maxItemCount': -1
        }

        result_iterable = container.query_items(
            query=query,
            feed_options=options
        )

        embeddings = []
        documents = []

        for result_list in result_iterable.by_page(continuation_token=None):
            for result in result_list:
                if 'embedding' in result and 'content' in result:
                    embeddings.append(np.array(result['embedding']))
                    documents.append(result['content'])

        return embeddings, documents

    def get_embeddings_and_documents_for_model(model):
        
        cosmos_url = "https://cosmos-hexainsights.documents.azure.com:443/"
        cosmos_key = "HJppOUqtKwDhvRpPi7ys3FIXEbKCXCItgfWuV4ZKsWa49nyZ1b1qwlXqbbUrZiL63JVzbvHcYH3JACDb2afTdQ=="
        database_name = "hexainsightsdb"
        container_name = "modelprompt"

        embeddings, documents = query_cosmos(cosmos_url, cosmos_key, database_name, container_name, model)

        return embeddings, documents

    API_KEY = "f700af5638614eed9c63d85695011af9"
    RESOURCE_ENDPOINT = "https://rapidxopenai.openai.azure.com/"

    openai.api_type = "azure"
    openai.api_key = API_KEY
    openai.api_base = RESOURCE_ENDPOINT
    openai.api_version = "2022-12-01"

    embeddings, documents = get_embeddings_and_documents_for_model(model)

    user_query_embedding = get_embedding(question, engine="ada2embed")

    cosine_similarities = cosine_similarity(embeddings, user_query_embedding)

    most_relevant_document_index = np.argmax(cosine_similarities)

    most_relevant_document_text = documents[most_relevant_document_index]

    human_message_prompt = HumanMessagePromptTemplate(
        prompt = PromptTemplate(
            input_variables=["question", "document_text"],
            template="Answer this: {question}. Base your answer off of this Document: {document_text}",
        )
    )

    chat_prompt_template = ChatPromptTemplate.from_messages([human_message_prompt])

    openai.api_type = "open_ai"
    openai.api_key = None
    openai.api_base = "https://api.openai.com/v1"
    openai.api_version = None

    chat = ChatOpenAI(temperature=0.9, openai_api_key="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")

    chain = LLMChain(llm=chat, prompt=chat_prompt_template)

    result = chain.run({
        'question': question,
        'document_text': most_relevant_document_text
    })

    print(result)

    return result


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/askai/{model}/{question}")
def run_llm_endpoint(model: str, question: str):
    result = run_llm(question, model)
    return {"result": result}

@app.get("/embed_model_contents/{model}")
def run_embeddings_endpoint(model: str):
    embed_model_contents(model)
    return {"status": "success"}
