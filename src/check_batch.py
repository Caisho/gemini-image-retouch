import base64
import json
import os

from dotenv import load_dotenv
from google import genai

# Load environment variables
load_dotenv()

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not set")
client = genai.Client(api_key=api_key)

JOB_NAME = "batches/worgu6z6dqyqjktv1znie2bv8mo38yjvsqo8"
OUTPUT_DIR = "./data/processed"


def save_image_from_part(part, output_path):
    # Verify if part has inline_data or similar
    # Structure usually: part.inline_data.data (bytes) or part.text
    # For images, verify how SDK returns it.
    pass


def main():
    print(f"Checking status for job: {JOB_NAME}")
    job = client.batches.get(name=JOB_NAME)
    print(f"Status: {job.state}")

    if job.state == "JOB_STATE_ACTIVE" or job.state == "JOB_STATE_PENDING":
        print("Job is still running. Please wait.")
        return

    if job.state == "JOB_STATE_FAILED":
        print(f"Job failed: {job.error}")
        return

    if (
        job.state == "JOB_STATE_SUCCEEDED" or job.state == "JOB_STATE_COMPLETED"
    ):  # Check exact enum
        print("Job completed successfully. Downloading results...")

        # Ensure output dir exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Download the output file
        # The job usually has 'output_file' which is a File resource name
        output_file_name = job.output_file
        print(f"Output file: {output_file_name}")

        # Get content
        # Note: client.files.content returns bytes
        content = client.files.content(name=output_file_name)

        # Parse JSONL
        lines = content.strip().split(b"\n")

        success_count = 0
        for i, line in enumerate(lines):
            try:
                result = json.loads(line)
                # Structure: {"custom_id": "...", "response": {...}}
                custom_id = result.get("custom_id", f"image_{i}")

                # Extract image
                # Response -> candidates -> content -> parts -> inline_data
                # We need to robustly find the image data
                response = result.get("response", {})

                # Need to handle potential errors in individual items
                if "error" in response:
                    print(f"Error for {custom_id}: {response['error']}")
                    continue

                # Navigate to the image
                # This depends on the exact JSON structure of the Batch output
                # Usually it mimics the generateContent response
                # Let's write the raw JSON for the first one to debug if we fail
                # But assume standard path: candidates[0].content.parts[0]

                # NOTE: Since we requested IMAGE modality, likely B64 encoded in inline_data
                # We'll do a robust check

                candidates = response.get("candidates", [])
                if not candidates:
                    print(f"No candidates for {custom_id}")
                    continue

                parts = candidates[0].get("content", {}).get("parts", [])
                for part in parts:
                    # Check for inline_data
                    if "inline_data" in part:
                        mime_type = part["inline_data"].get("mime_type", "image/png")
                        data_b64 = part["inline_data"]["data"]

                        # Determine extension
                        ext = ".png"
                        if "jpeg" in mime_type:
                            ext = ".jpg"

                        # Filename
                        # custom_id was "retouch_filename.jpg"
                        # Clean it up
                        base_name = custom_id.replace("retouch_", "")
                        # Split existing extension if present so we don't duplicate
                        name_part = os.path.splitext(base_name)[0]
                        out_name = f"{name_part}_retouched{ext}"
                        out_path = os.path.join(OUTPUT_DIR, out_name)

                        with open(out_path, "wb") as img_f:
                            img_f.write(base64.b64decode(data_b64))

                        print(f"Saved: {out_path}")
                        success_count += 1

            except Exception as e:
                print(f"Failed to process line {i}: {e}")

        print(f"\nProcessing complete. Saved {success_count} images to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
