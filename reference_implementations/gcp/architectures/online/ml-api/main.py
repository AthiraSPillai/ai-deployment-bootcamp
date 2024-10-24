import streamlit as st
import requests
import json
from google.cloud import storage
from google.cloud import documentai_v1beta3 as documentai
from google.cloud import aiplatform_v1, aiplatform
from fastapi import FastAPI, File, UploadFile, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.wsgi import WSGIMiddleware
import os
import uvicorn

# Configuration
ENDPOINT_ID = os.getenv("ENDPOINT_ID")
MODEL_NAME = os.getenv("MODEL_NAME")
ZONE = os.getenv("ZONE")
REGION = f"{ZONE.split('-')[0]}-{ZONE.split('-')[1]}"
PROJECT_ID = os.getenv("PROJECT_ID")
PROJECT_NUMBER = os.getenv("PROJECT_NUMBER")
ENV = os.getenv("ENV")
PROJECT_PREFIX = PROJECT_ID.replace("-", "_")
BUCKET_NAME = os.getenv("BUCKET_NAME")
AUTH_TOKEN = os.getenv("AUTH_TOKEN")

# Initialize FastAPI
app = FastAPI()

@app.post("/predict")
async def predict(text: str = Body(None), num_tokens: int = Body(...), file: UploadFile = File(None)):
    if file:
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(file.filename)
        blob.upload_from_string(file.file.read())
        file_url = f"gs://{BUCKET_NAME}/{file.filename}"
        docai_client = documentai.DocumentUnderstandingServiceClient()
        input_config = documentai.DocumentUnderstandingService.DocumentInputConfig(
            gcs_source={"uri": file_url}, mime_type="application/pdf"
        )
        request = documentai.ProcessDocumentRequest(input_config=input_config)
        document = docai_client.process_document(request=request)
        text = document.text

    if not text:
        return JSONResponse(content={"error": "Text input is required."}, status_code=400)

    input_data = {
        "text": text,
        "num_tokens": num_tokens,
        "model": MODEL_NAME
    }
    prediction_client = aiplatform_v1.PredictionServiceClient(
        client_options={
            "api_endpoint": f"{REGION}-aiplatform.googleapis.com",
        }
    )
    json_data = json.dumps(input_data)
    http_body = httpbody_pb2.HttpBody(data=json_data.encode("utf-8"), content_type="application/json")
    request = aiplatform_v1.RawPredictRequest(
        endpoint=f"projects/{PROJECT_NUMBER}/locations/{REGION}/endpoints/{ENDPOINT_ID}",
        http_body=http_body,
    )
    response = prediction_client.raw_predict(request)
    return {"prediction": json.loads(response.data)}

# Mount FastAPI app to Streamlit
st_app = FastAPI()
st_app.mount("/api", app)

# Streamlit UI
st.set_page_config(page_title="BMO Document Summarizer", page_icon=":bank:")
st.markdown("""
    <style>
    .main {
        background-color: #e6eff9;
    }
    h1 {
        color: #004d98;
    }
    .sidebar .sidebar-content {
        background-color: #004d98;
    }
    .css-1d391kg {
        color: #ffffff;
    }
    .css-1d391kg:focus {
        background-color: #e6eff9;
    }
    </style>
""", unsafe_allow_html=True)

# BMO Logo
# logo_url = "https://upload.wikimedia.org/wikipedia/en/8/88/BMO_Financial_Group_logo.svg"
# st.image(logo_url, width=200)

st.title("Document Summarizer")

uploaded_file = st.file_uploader('Please upload a PDF file')
news_text = st.text_area('Or paste news text here')
num_tokens = st.slider('Select number of tokens for summarization', min_value=50, max_value=1000, value=200)

def call_backend(text, num_tokens, uploaded_file=None):
    if uploaded_file:
        files = {'file': uploaded_file}
    else:
        files = None

    payload = {
        "text": text,
        "num_tokens": num_tokens
    }

    response = requests.post("http://localhost:8080/api/predict", files=files, data=payload)
    if response.status_code == 200:
        return response.json().get("prediction")
    else:
        return "Error: Something went wrong."

if uploaded_file is not None:
    st.write("File uploaded successfully.")
    summary = call_backend(text="", num_tokens=num_tokens, uploaded_file=uploaded_file)
    st.subheader('Summarized PDF Content')
    st.write(summary)

if news_text:
    summary = call_backend(text=news_text, num_tokens=num_tokens)
    st.subheader('Summarized Content')
    st.write(summary)
