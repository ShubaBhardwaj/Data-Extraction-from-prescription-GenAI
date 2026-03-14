import base64
import mimetypes
import json
import os
from dotenv import load_dotenv
from openai import OpenAI
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# --- Configuration ---

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# --- FastAPI Setup ---

app = FastAPI(title="Gemini Image Analyzer API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins; change in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Helper Function ---

def encode_image_file(file: UploadFile):
    """Encodes uploaded image to base64 and determines MIME type."""
    try:
        mime_type = file.content_type
        if not mime_type or not mime_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Uploaded file is not an image.")

        file_bytes = file.file.read()
        encoded_string = base64.b64encode(file_bytes).decode("utf-8")
        return encoded_string, mime_type
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {e}")


# --- System Prompt ---

system_prompt = """
You are a helpful AI Assistant specialized in analyzing images.
You are helpful AI assistant specialized in analysing medical prescription images .
1)User gives image of prescription.
2)System decodes the messy handwriting of doctor and find out the dosage timimg and how to take(like after breakfast or before breakfast) and medicine name .
3)If prescription says twice a day(two dots in front of medicine name ) ,then assign medicine at 8 am in morning and 8pm in evening. 
4)If thrice(three dots in front of medicine name), assign at 8 am in morning , 2 pm in morning and 8 pm in evening .
Rules:
1) Doctors generally use the following mentioned abbreviations-
OD (Once a day) -> 8:00 AM
BD/BDS (Twice a day) -> 8:00 AM & 8:00 PM
TDS/TID (Thrice a day) -> 8:00 AM, 2:00 PM, & 8:00 PM
QID (Four times a day) -> 8:00 AM, 12:00 PM, 4:00 PM, & 8:00 PM
SOS -> "Only when needed" (e.g., for pain or fever).
2)There should be atleast 90percent accuracy in reading the prescription
3)Keep dosage clear and language simple.
4)Always show the Medicine Name and Dosage back to the user with a "Does this look correct?" prompt before adding it to the timetable.
Output:
Output should be in json format 
Morning (08:00 AM): [Medicine Name] - [Dosage] - [Before/After Food]

Afternoon (02:00 PM): [Medicine Name] - [Dosage] - [Before/After Food]

Evening (08:00 PM): [Medicine Name] - [Dosage] - [Before/After Food]

Return the result as JSON with a single key 'Conclusion' containing the full analysis.
"""

# --- API Endpoint ---

@app.post("/analysis")
async def analyze_image(file: UploadFile = File(...)):
    """
    Upload an image for AI analysis.
    The image can be either food or a medical prescription.
    Returns structured JSON response.
    """
    try:
        encoded_image, mime_type = encode_image_file(file)

        content_parts = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{encoded_image}"
                }
            },
            {
                "type": "text",
                "text": "Analyze this image according to the system prompt's instructions and provide the result as JSON."
            }
        ]

        message = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content_parts},
        ]

        response = client.chat.completions.create(
            model="gemini-2.5-flash",
            response_format={"type": "json_object"},
            temperature=0.1,
            messages=message
        )

        response_content = response.choices[0].message.content
        parsed_output = json.loads(response_content)

        if "Conclusion" in parsed_output:
            return JSONResponse(content={"status": "success", "data": parsed_output["Conclusion"]})
        else:
            return JSONResponse(content={"status": "partial", "raw": parsed_output})

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Invalid JSON response from AI.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Root Route ---

@app.get("/")
def home():
    return {"message": "Welcome to Gemini Image Analyzer API! Use POST /analysis to upload an image."}