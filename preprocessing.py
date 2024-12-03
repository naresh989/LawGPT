import os
from PyPDF2 import PdfReader
from concurrent.futures import ThreadPoolExecutor

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a single PDF file.
    """
    output = ""
    reader = PdfReader(pdf_path)
    for page in reader.pages:
        output += page.extract_text() + "\n"
    return output

def process_pdfs_in_parallel(pdf_folder, output_file):
    """
    Processes multiple PDFs in parallel and saves the combined text to a file.
    """
    pdf_files = [os.path.join(pdf_folder, file) for file in os.listdir(pdf_folder) if file.endswith(".pdf")]
    with ThreadPoolExecutor() as executor:
        results = executor.map(extract_text_from_pdf, pdf_files)

    combined_text = "\n".join(results)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(combined_text)

def reformat_to_paragraphs(input_file, output_file, max_words_per_paragraph=50):
    """
    Reformats a text file into paragraphs with a maximum number of words.
    """
    with open(input_file, "r", encoding="utf-8") as infile:
        words = [line.strip() for line in infile if line.strip()]

    text = " ".join(words)
    sentences = text.split(". ")
    sentences = [sentence.strip().capitalize() + "." for sentence in sentences if sentence]

    paragraphs = []
    current_paragraph = []
    word_count = 0

    for sentence in sentences:
        word_count += len(sentence.split())
        current_paragraph.append(sentence)
        if word_count >= max_words_per_paragraph:
            paragraphs.append(" ".join(current_paragraph))
            current_paragraph = []
            word_count = 0

    if current_paragraph:
        paragraphs.append(" ".join(current_paragraph))

    with open(output_file, "w", encoding="utf-8") as outfile:
        for paragraph in paragraphs:
            outfile.write(paragraph + "\n\n")
