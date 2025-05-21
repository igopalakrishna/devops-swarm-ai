# test_gcp_direct.py
import os
from google.cloud import logging as google_cloud_logging # Renamed to avoid conflict if you have a local 'logging' module
from vertexai.preview.language_models import TextGenerationModel
import vertexai
from vertexai.generative_models import GenerativeModel # Updated import

# --- (!!!) IMPORTANT: EDIT THESE TWO LINES (!!!) ---
GCP_PROJECT_ID = "devops-swarm-ai-project"  # <<< REPLACE "devops-swarm-ai-project" with YOUR ACTUAL GCP Project ID if different (but yours seems to be this)
GCP_LOCATION = "us-central1"               # <<< REPLACE "us-central1" with your preferred Vertex AI region if different (e.g., "europe-west1", "asia-southeast1", etc.)
# --- Make sure the region you choose supports Vertex AI and the Gemini models ---

print(f"--- Starting GCP Direct Test ---")
print(f"Using Project ID: {GCP_PROJECT_ID}")
print(f"Using Location/Region: {GCP_LOCATION}")

# Check if GOOGLE_APPLICATION_CREDENTIALS is set (useful for debugging if you are using a service account)
gac = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
if gac:
    print(f"GOOGLE_APPLICATION_CREDENTIALS is SET to: {gac}")
else:
    print(f"GOOGLE_APPLICATION_CREDENTIALS is NOT SET (Application Default Credentials will be used).")

# Initialize Vertex AI
try:
    print(f"\nAttempting to initialize Vertex AI for project '{GCP_PROJECT_ID}' in location '{GCP_LOCATION}'...")
    vertexai.init(project=GCP_PROJECT_ID, location=GCP_LOCATION)
    print("✅ Vertex AI initialized successfully.")
except Exception as e:
    print(f"❌ ERROR initializing Vertex AI: {e}")
    print("--- GCP Direct Test Failed (Vertex AI Init) ---")
    exit() # Exit if Vertex AI can't initialize, as the model call will also fail

# Test Cloud Logging
try:
    print("\nAttempting to write to Google Cloud Logging...")
    logging_client = google_cloud_logging.Client(project=GCP_PROJECT_ID)
    logger_name = "my-direct-gcp-test-log" # Use a distinct logger name
    logger = logging_client.logger(logger_name)
    logger.log_text("Direct GCP test log entry from test_gcp_direct.py script.")
    print(f"✅ Successfully wrote a test log entry to Google Cloud Logging in project '{GCP_PROJECT_ID}'. Check for logger named '{logger_name}' in Logs Explorer.")
except Exception as e:
    print(f"❌ ERROR writing to Cloud Logging: {e}")

# Test Vertex AI (Gemini Pro model using GenerativeModel)
try:
    print("\nAttempting to call Vertex AI (Gemini Pro using GenerativeModel)...")
    # Use GenerativeModel for Gemini
    model = GenerativeModel("gemini-2.5-pro-preview-05-06") 
    
    prompt = "What is the capital of France? Respond with only the city name."
    response = model.generate_content(prompt)
    
    # Accessing the text response correctly from GenerativeModel
    # The response object structure is different from TextGenerationModel
    # Typically, it's response.text or iterating through response.candidates[0].content.parts
    # For a simple text prompt, response.text should work.
    # If response.text doesn't work directly, try:
    # summary = "".join([part.text for part in response.candidates[0].content.parts])

    summary = response.text # Try this first for simple text output
    print(f"✅ Successfully called Vertex AI (Gemini Pro using GenerativeModel). Response: {summary.strip()}")

except Exception as e:
    print(f"❌ ERROR calling Vertex AI (Gemini Pro using GenerativeModel): {e}")
    import traceback
    traceback.print_exc() # Print full traceback for better debugging

print("\n--- GCP Direct Test Complete ---")
