from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications import ResNet50
from services.cloudinary import upload_to_cloudinary
from PIL import Image
import numpy as np
from flask import Flask, request, jsonify
from qdrant_client.http.models import VectorParams, Distance
from qdrant_client import QdrantClient
import os
import logging
import time  # For retry logic
import base64
# Disable oneDNN optimizations
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


def qdrant_request_with_retries(func, *args, retries=3, backoff_multiplier=2, **kwargs):
    """Custom retry logic for Qdrant requests."""
    for attempt in range(retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt < retries - 1:
                wait_time = backoff_multiplier ** attempt
                logging.warning(
                    f"Retrying Qdrant request in {wait_time} seconds... (Attempt {attempt + 1})")
                time.sleep(wait_time)
            else:
                logging.error(
                    f"Qdrant request failed after {retries} attempts: {e}")
                raise


# Initialize Qdrant client
qdrant_client = QdrantClient(
    url="https://f46fd696-7a66-49eb-9c0e-fdf0e3426942.europe-west3-0.gcp.cloud.qdrant.io:6333",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwiZXhwIjoxOTE2MTMxMzM2fQ.pe7nAz1bxGnQIrBsevDslsMCFmcssGY0HaKr6mhlOFw",
    timeout=60.0,  # Set timeout to 60 seconds
)

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained ResNet50 model for feature extraction
model = ResNet50(weights="imagenet", include_top=False, pooling="avg")

# Function to extract features from an image


def extract_features(image_path):
    try:
        image = Image.open(image_path).convert("RGB").resize((224, 224))
        image_array = img_to_array(image)
        image_array = np.expand_dims(image_array, axis=0)
        image_array = preprocess_input(image_array)
        features = model.predict(image_array)
        return features.flatten()
    except Exception as e:
        logging.error(f"Error processing image {image_path}: {e}")
        return None


# Create a collection in Qdrant
collection_name = "image_search"
if not qdrant_request_with_retries(qdrant_client.collection_exists, collection_name=collection_name):
    qdrant_request_with_retries(
        qdrant_client.create_collection,
        collection_name=collection_name,
        vectors_config=VectorParams(size=2048, distance=Distance.COSINE),
    )


def clean_qdrant_collection():
    """Delete and recreate the Qdrant collection."""
    try:
        if qdrant_request_with_retries(qdrant_client.collection_exists, collection_name=collection_name):
            logging.info(f"Deleting existing collection: {collection_name}")
            qdrant_request_with_retries(
                qdrant_client.delete_collection, collection_name=collection_name)
        logging.info(f"Creating collection: {collection_name}")
        qdrant_request_with_retries(
            qdrant_client.create_collection,
            collection_name=collection_name,
            vectors_config=VectorParams(size=2048, distance=Distance.COSINE),
        )
    except Exception as e:
        logging.error(f"Error cleaning Qdrant collection: {e}")


def image_to_base64(image_path):
    """Convert an image to a Base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        logging.error(f"Error converting image to Base64: {e}")
        return None


def process_images_folder():
    """Clean Qdrant and process images in the './images' folder."""
    image_folder = "./images2"
    if not os.path.exists(image_folder):
        logging.error(f"Error: Folder '{image_folder}' does not exist.")
        return
    if not os.listdir(image_folder):
        logging.warning(f"Warning: Folder '{image_folder}' is empty.")
        return

    clean_qdrant_collection()  # Clean the collection before processing
    for idx, image_file in enumerate(os.listdir(image_folder)):
        image_path = os.path.join(image_folder, image_file)
        if os.path.isfile(image_path):
            # Upload image to Cloudinary
            image_url = upload_to_cloudinary(image_path)
            if not image_url:
                logging.error(f"Failed to upload {image_file} to Cloudinary.")
                continue

            # Extract features
            vector = extract_features(image_path)
            if vector is not None:  # Only upsert if feature extraction was successful
                try:
                    qdrant_request_with_retries(
                        qdrant_client.upsert,
                        collection_name=collection_name,
                        points=[{
                            "id": idx,
                            "vector": vector,
                            "payload": {
                                "filename": image_file,
                                "image_url": image_url
                            }
                        }],
                    )
                    logging.info(
                        f"Inserted image '{image_file}' into Qdrant with URL: {image_url}")
                except Exception as e:
                    logging.error(
                        f"Error inserting image '{image_file}' into Qdrant: {e}")


process_images_folder()


@app.route("/process-images", methods=["POST"])
def process_images():
    """Endpoint to trigger processing of the images folder."""
    try:
        process_images_folder()
        return jsonify({"message": "Images processed successfully."})
    except Exception as e:
        logging.error(f"Error processing images: {e}")
        return jsonify({"error": "Failed to process images."}), 500


@app.route("/search", methods=["POST"])
def search():
    """Search endpoint to find similar images."""
    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    # query_image = request.files["image"]
    query_path = "./images2/5_0_945522561097668.jpg"
    # query_image.save(query_path)

    query_vector = extract_features(query_path)
    if query_vector is None:
        return jsonify({"error": "Failed to extract features from the query image."}), 400

    try:
        search_result = qdrant_request_with_retries(
            qdrant_client.search,
            collection_name=collection_name,
            query_vector=query_vector,
            limit=5,
        )
        results = [
            {
                "id": hit.id,
                "score": hit.score,
                "filename": hit.payload["filename"],
                "image_url": hit.payload["image_url"]
            }
            for hit in search_result
        ]
        return jsonify(results)
    except Exception as e:
        logging.error(f"Error during search: {e}")
        return jsonify({"error": "Search operation failed."}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
