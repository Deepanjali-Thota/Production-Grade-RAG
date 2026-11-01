import os
from docling.document_converter import DocumentConverter

# Target one specific file to test
TEST_FILE = "data/production_baseline/MSFT.pdf"

def parse_test_document():
    if not os.path.exists(TEST_FILE):
        print(f"Error: Could not find {TEST_FILE}. Make sure the S3 sync downloaded it!")
        return

    print(f"Starting Docling extraction for: {TEST_FILE}")
    print("This might take a minute or two for a large 10-K...")

    # Initialize Docling and convert the document
    converter = DocumentConverter()
    result = converter.convert(TEST_FILE)

    # Export the parsed document to Markdown
    markdown_content = result.document.export_to_markdown()

    print("\n✅ Parsing Complete!")
    print("--- Snippet of Extracted Text ---")
    print(markdown_content[:1000]) # Print the first 1000 characters
    print("---------------------------------")

if __name__ == "__main__":
    parse_test_document()