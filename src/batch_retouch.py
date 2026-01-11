import json
import os

from dotenv import load_dotenv
from google import genai
from google.cloud import storage
from google.genai import types
from tqdm import tqdm

# Load environment variables from .env file
load_dotenv()

# 1. Initialize the client
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise ValueError(
        "API key not provided. Set GEMINI_API_KEY environment variable in .env file"
    )
client = genai.Client(api_key=api_key)

# 2. Configuration
# Note: In 2026, Batch API requires images to be in a GCS bucket
GCS_INPUT_PATH = "gs://gemini-image-retouch/raw/"
GCS_OUTPUT_PATH = "gs://gemini-image-retouch/proccessed/"
MODEL_ID = "models/gemini-2.5-flash"

RAW_DIR = "./data/raw/piano_full"


# Helper: Upload images to GCS
def upload_images_to_gcs(local_dir, gcs_path):
    print(f"Uploading images from {local_dir} to {gcs_path}...")

    # Parse bucket and prefix
    if not gcs_path.startswith("gs://"):
        raise ValueError("GCS_INPUT_PATH must start with gs://")

    path_parts = gcs_path[5:].split("/", 1)
    bucket_name = path_parts[0]
    prefix = path_parts[1] if len(path_parts) > 1 else ""
    if prefix and not prefix.endswith("/"):
        prefix += "/"

    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
    except Exception as e:
        print(f"Error initializing GCS client: {e}")
        print(
            "Ensure you have GOOGLE_APPLICATION_CREDENTIALS set or are authenticated via gcloud."
        )
        raise

    # Get list of files
    if not os.path.exists(local_dir):
        raise FileNotFoundError(f"Local directory {local_dir} does not exist")

    files = [
        f
        for f in os.listdir(local_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    ]

    if not files:
        print("No images found to upload.")
        return files

    for filename in tqdm(files, desc="Uploading to GCS"):
        local_file_path = os.path.join(local_dir, filename)
        blob_path = f"{prefix}{filename}"
        blob = bucket.blob(blob_path)
        blob.upload_from_filename(local_file_path)

    return files


# 3. Create the Batch Request File (.jsonl)
# Format required by Gemini API
def create_batch_file(image_names, prompt):
    print(f"Creates batch file for {len(image_names)} images...")
    with open("batch_requests.jsonl", "w") as f:
        for img_name in image_names:
            # Construct the line
            line = {
                "custom_id": f"retouch_{img_name}",
                "method": "POST",
                "request": {
                    "model": MODEL_ID,
                    "contents": [
                        {"parts": [{"text": prompt}]},
                        {
                            "parts": [
                                {
                                    "file_data": {
                                        "mime_type": "image/jpeg",
                                        "file_uri": f"{GCS_INPUT_PATH}{img_name}",
                                    }
                                }
                            ]
                        },
                    ],
                    "generation_config": {"response_modalities": ["IMAGE"]},
                },
            }
            f.write(json.dumps(line) + "\n")


# 4. Define the Retouching Prompt
master_prompt = (
    "Analyze this batch of images taken in the same location. Your goal is to apply a uniform professional retouch across all photos, ensuring consistent lighting, color grading, and sharpness on the people.\n\n"
    "1. Lighting & Exposure Correction: The subjects are currently heavily backlit. Apply a flattering fill light to the people in the foreground to brighten their faces and reveal details in their black clothing. Balance this new foreground exposure with the bright window background so the images look naturally bright, not washed out. Remove atmospheric haze caused by the window light.\n\n"
    "2. Color Consistency: Establish a consistent white balance across the batch. The sheer curtains and white textured furniture should render as clean, neutral white (no blue or yellow tints). Skin tones for all individuals must be natural, warm, and consistent from photo to photo, regardless of their position relative to the window.\n\n"
    "3. Texture & Sharpening: Apply a uniform high-quality sharpening pass to the entire batch. The textures of the dried grass, the fabric of the sofa, and hair details should be crisp and visually identical in sharpness across all images.\n\n"
    "Output the processed batch maintaining high resolution."
)

# 5. Main Execution
if __name__ == "__main__":
    try:
        # Upload images
        uploaded_files = upload_images_to_gcs(RAW_DIR, GCS_INPUT_PATH)

        if not uploaded_files:
            print("No files to process.")
            exit()

        # Create batch file
        create_batch_file(uploaded_files, master_prompt)
        print("✓ Created batch_requests.jsonl\n")

        # Upload batch file to Gemini API
        print("Uploading batch_requests.jsonl to Gemini API...")
        uploaded_file = client.files.upload(
            file="batch_requests.jsonl",
            config=types.UploadFileConfig(
                display_name="image-retouch-batch", mime_type="application/jsonl"
            ),
        )
        print(f"File uploaded: {uploaded_file.name}")

        # Submit Batch Job
        print("Creating batch job...")

        try:
            # Try to set destination
            batch_job = client.batches.create(
                model=MODEL_ID,
                src=uploaded_file.name,
                config=types.CreateBatchJobConfig(
                    dest=types.BatchJobDestination(gcs_uri=GCS_OUTPUT_PATH)
                ),
            )
            print(f"  Output Dest: {GCS_OUTPUT_PATH}")

        except Exception as e:
            print(
                f"⚠️ Warning: Could not set GCS destination ({e}). Falling back to default storage."
            )
            batch_job = client.batches.create(model=MODEL_ID, src=uploaded_file.name)
            print("  Output Dest: Default (Cloud Storage)")

        print("\n✓ Batch Job Created!")
        print(f"  Job ID: {batch_job.name}")
        print(f"  Status: {batch_job.state}")

        print("\nTo check status later, use:")
        print(f"  job = client.batches.get(name='{batch_job.name}')")
        print("  print(job.state)")

    except Exception as e:
        print(f"\n❌ info: An error occurred: {e}")
