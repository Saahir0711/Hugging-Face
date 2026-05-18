import requests
import random

from config import HF_API_KEY

MODEL = "facebook/bart-large-mnli"
API_URL = f"https://router.huggingface.co/hf-inference/models/{MODEL}"
HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}
LABELS = ["Spam", "Safe"]

def classify_message(message):
    payload = {
        "inputs": message,
        "parameters": {"candidate_labels": LABELS}
    }

    response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=40)

    if not response.ok:
        raise RuntimeError(f"API error: {response.status_code}")
    
    data = response.json()
    
    results = list(zip([data[0]['label'], data[1]['label']], [data[0]['score'], data[1]['score']]))
    return sorted(results, key=lambda x: x[1], reverse=True)

while True:

    message = input("Type in your chosen message or type q to quit")
    if message != "q":
        results = classify_message(message)
        print("Welcome to Safe vs Spam.")
        print ("The message was: ", message)
        if results[0][1] > results[1][1]:
            print ("The message is classified as SAFE")
        else:
            print("The message is classified as SPAM")
        for label, score in results:
            print (f"{label}: Score ={score*100} ")
    else:
        break
