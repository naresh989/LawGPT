from preprocessing import process_pdfs_in_parallel, reformat_to_paragraphs, merge_and_preprocess

# Paths
pdf_folder = "path_to_pdf_folder"
combined_text_file = "path_to_combined_text.txt"
formatted_text_file = "path_to_formatted_text.txt"
merged_file = "path_to_merged_text.txt"

# Extract and process PDFs
process_pdfs_in_parallel(pdf_folder, combined_text_file)
reformat_to_paragraphs(combined_text_file, formatted_text_file)
merge_and_preprocess([formatted_text_file], merged_file)
