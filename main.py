import os
import io
import argparse
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

# Load environment variables from .env file
load_dotenv()

# Available Gemini models for image processing
AVAILABLE_MODELS = [
    "gemini-2.5-flash-image",
    "gemini-3-pro-image-preview",
]


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Batch process images using Google Gemini API for professional retouching."
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="./data/raw",
        help="Directory containing raw images to process (default: ./data/raw)",
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default="./data/processed",
        help="Directory to save processed images (default: ./data/processed)",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=AVAILABLE_MODELS,
        default="gemini-3-pro-image-preview",
        help=f"Gemini model to use (default: gemini-3-pro-image-preview)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Apply a professional color grade. Brighten subjects to counter backlighting. Ensure consistent skin tones and high sharpness.",
        help="Retouch prompt for the AI model",
    )
    return parser.parse_args()


def main():
    """Main processing function."""
    args = parse_args()

    # Validate API key
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "API key not provided. Set GEMINI_API_KEY environment variable in .env file"
        )

    # Create output directory if it doesn't exist
    os.makedirs(args.processed_dir, exist_ok=True)

    # Initialize the client
    client = genai.Client(api_key=api_key)

    print(f"Processing images from: {args.raw_dir}")
    print(f"Saving processed images to: {args.processed_dir}")
    print(f"Using model: {args.model}")
    print(f"Prompt: {args.prompt}\n")

    # Process the batch
    processed_count = 0
    for filename in os.listdir(args.raw_dir):
        if filename.endswith((".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG")):
            img_path = os.path.join(args.raw_dir, filename)

            print(f"Processing: {filename}...")

            with open(img_path, "rb") as f:
                image_bytes = f.read()

            # Send request to Gemini
            response = client.models.generate_content(
                model=args.model,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                    args.prompt,
                ],
                config=types.GenerateContentConfig(response_modalities=["IMAGE"]),
            )

            # Save the returned image
            for part in response.candidates[0].content.parts:
                if part.inline_data:
                    edited_img = Image.open(io.BytesIO(part.inline_data.data))
                    output_path = os.path.join(
                        args.processed_dir, f"retouched_{filename}"
                    )
                    edited_img.save(output_path)
                    print(f"Saved: {output_path}")
                    processed_count += 1

    print(f"\nProcessing complete! {processed_count} images processed.")


if __name__ == "__main__":
    main()