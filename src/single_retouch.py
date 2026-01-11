import os
import io
import argparse
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image
from tqdm import tqdm

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
        default="gemini-2.5-flash-image",
        help=f"Gemini model to use (default: gemini-2.5-flash-image)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Retouch prompt for the AI model (uses default professional batch retouch prompt if not specified)",
    )
    return parser.parse_args()


def main():
    """Main processing function."""
    args = parse_args()

    # Set default prompt if not provided
    if args.prompt is None:
        args.prompt = (
            "Analyze this batch of images taken in the same location. Your goal is to apply a uniform professional retouch across all photos, ensuring consistent lighting, color grading, and sharpness on the people.\n\n"
            "1. Lighting & Exposure Correction: The subjects are currently heavily backlit. Apply a flattering fill light to the people in the foreground to brighten their faces and reveal details in their black clothing. Balance this new foreground exposure with the bright window background so the images look naturally bright, not washed out. Remove atmospheric haze caused by the window light.\n\n"
            "2. Color Consistency: Establish a consistent white balance across the batch. The sheer curtains and white textured furniture should render as clean, neutral white (no blue or yellow tints). Skin tones for all individuals must be natural, warm, and consistent from photo to photo, regardless of their position relative to the window.\n\n"
            "3. Texture & Sharpening: Apply a uniform high-quality sharpening pass to the entire batch. The textures of the dried grass, the fabric of the sofa, and hair details should be crisp and visually identical in sharpness across all images.\n\n"
            "Output the processed batch maintaining high resolution."
        )

    # Validate API key
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "API key not provided. Set GEMINI_API_KEY environment variable in .env file"
        )

    # Determine output directory preserving subdirectory structure
    raw_path = Path(args.raw_dir)
    processed_base = Path(args.processed_dir)
    
    # If raw_dir has subdirectories after 'raw', preserve them in processed
    raw_parts = raw_path.parts
    if "raw" in raw_parts:
        raw_idx = raw_parts.index("raw")
        subdir_parts = raw_parts[raw_idx + 1:]
        output_dir = processed_base.joinpath(*subdir_parts)
    else:
        output_dir = processed_base
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Initialize the client
    client = genai.Client(api_key=api_key)

    print(f"Processing images from: {args.raw_dir}")
    print(f"Saving processed images to: {output_dir}")
    print(f"Using model: {args.model}")
    print(f"Prompt: {args.prompt}\n")

    # Get list of image files
    image_files = [
        f for f in os.listdir(args.raw_dir)
        if f.endswith((".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG"))
    ]

    # Process the batch with progress bar
    processed_count = 0
    for filename in tqdm(image_files, desc="Processing images", unit="image"):
        img_path = os.path.join(args.raw_dir, filename)

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
                # Save as PNG
                output_path = os.path.join(output_dir, f"retouched_{Path(filename).stem}.png")
                edited_img.save(output_path, format='PNG')
                processed_count += 1

    print(f"\nProcessing complete! {processed_count} images processed.")


if __name__ == "__main__":
    main()