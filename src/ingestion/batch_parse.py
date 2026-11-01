import os
import glob
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption

INPUT_DIR = "data/production_baseline"
OUTPUT_DIR = "data/parsed_docs"

def batch_parse_documents():
    # Make sure output folder exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Find all PDFs in the input folder
    pdf_files = glob.glob(os.path.join(INPUT_DIR, "*.pdf"))
    
    if not pdf_files:
        print("No PDF files found to parse!")
        return

    print(f"Found {len(pdf_files)} documents. Booting up Docling...")

    # Speed settings: 10-K PDFs already contain text, so OCR is not needed.
    # FAST table mode still extracts tables, just with a lighter model.
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.mode = TableFormerMode.FAST

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )
    
    for i, pdf_path in enumerate(pdf_files, 1):
        filename = os.path.basename(pdf_path)
        markdown_filename = filename.replace(".pdf", ".md")
        output_path = os.path.join(OUTPUT_DIR, markdown_filename)
        
        # Skip if already parsed
        if os.path.exists(output_path):
            print(f"[{i}/{len(pdf_files)}] Skipping {filename} (already parsed)")
            continue
            
        print(f"[{i}/{len(pdf_files)}] Parsing {filename}...")
        
        try:
            result = converter.convert(pdf_path)
            markdown_content = result.document.export_to_markdown()
            
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(markdown_content)
            print(f"    -> Saved to {output_path}")
            
        except Exception as e:
            print(f"    -> Error parsing {filename}: {e}")

if __name__ == "__main__":
    batch_parse_documents()