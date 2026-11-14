"""
Packs the data the CI evaluation needs into one zip and uploads it to S3.

CI needs only what answering a question reads, not the PDFs or the parsed markdown:
    data/chroma_db/       vector database
    data/chunked_docs/    chunk text (BM25 index)
    data/chunk_pages/     page numbers for citations
    data/companies.json   company names

Run it again whenever you re-ingest documents, so CI evaluates the same data you do.

Usage (from the project root):
    python src/ingestion/upload_ci_data.py
"""
import os
import zipfile

import boto3
from dotenv import load_dotenv

load_dotenv()

BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
S3_KEY = "ci/ci_data.zip"
ZIP_PATH = "ci_data.zip"
INCLUDE = ["data/chroma_db", "data/chunked_docs", "data/chunk_pages", "data/companies.json"]


def build_zip():
    missing = [p for p in INCLUDE if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"Missing: {missing}. Run the ingestion steps first.")

    count = 0
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in INCLUDE:
            files = [path] if os.path.isfile(path) else [
                os.path.join(root, name) for root, _, names in os.walk(path) for name in names
            ]
            for file in files:
                # forward slashes, so the zip unpacks correctly on the Linux CI machine
                zf.write(file, arcname=file.replace(os.sep, "/"))
                count += 1
    size_mb = os.path.getsize(ZIP_PATH) / 1_000_000
    print(f"Packed {count} files into {ZIP_PATH} ({size_mb:.1f} MB)")


def upload():
    if not BUCKET_NAME:
        raise SystemExit("S3_BUCKET_NAME is not set in .env")
    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    )
    s3.upload_file(ZIP_PATH, BUCKET_NAME, S3_KEY)
    print(f"Uploaded to s3://{BUCKET_NAME}/{S3_KEY}")


if __name__ == "__main__":
    build_zip()
    upload()
    os.remove(ZIP_PATH)