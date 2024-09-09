import fitz  # aka PyMuPDF
from typing import Optional

def pdf_text_extractor(filepath: str, start_page: int = 0, end_page: Optional[int] = None) -> str:
    """
    Extracts text from specified pages of a PDF file.

    Args:
        filepath (str): Path to the PDF file.
        start_page (int): The first page to start extraction from (0-indexed).
        end_page (Optional[int]): The last page to stop extraction at (0-indexed). If None, processes to 
                                  the end of the document.

    Returns:
        str: Extracted text as a single string.
    """
    text = ""
    with fitz.open(filepath) as doc:
        # Determine the last page if end_page is not specified
        if end_page is None or end_page >= doc.page_count:
            end_page = doc.page_count - 1
        
        # Extract text from each page given a specific page range
        for page_num in range(start_page - 1, end_page + 1):
            page = doc.load_page(page_num)
            text += page.get_text("text")
    
    return text

def save_text_to_file(text: str, output_filepath: str) -> None:
    """
    Saves extracted text to a text file.

    Args:
        text (str): String containing the text to be saved.
        output_filepath (str): Path where the text file will be saved.
    """
    with open(output_filepath, 'w', encoding='utf-8') as file:
        file.write(text)