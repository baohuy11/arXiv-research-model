from PyPDF2 import PdfReader
from io import BytesIO
import requests

def summarize_pdf(url: str, summarizer, tokenizer) -> str:
    response = requests.get(url)
    response.raise_for_status()
        
    # Read the PDF content from bytes
    pdf_file = BytesIO(response.content)
    reader = PdfReader(pdf_file)
        
    # Extract text from each page of the PDF
    extracted_text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            extracted_text += page_text + "\n"
        
    if not extracted_text.strip():
        return "No text could be extracted from the PDF."
    
    # Tokenize the full text and truncate to the model's max token length.
    inputs = tokenizer(
        extracted_text,
        max_length=tokenizer.model_max_length - 2,
        truncation=True,
        return_tensors="pt"
    )
    # Convert tokens back to text if the summarizer expects raw text.
    truncated_text = tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)
    # Generate a summary. You can tweak max_length and min_length to control output.
    summary = summarizer(truncated_text, max_length=150, min_length=50, do_sample=False)
    return summary[0]['summary_text']