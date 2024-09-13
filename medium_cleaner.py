import os

# Define the phrases to remove from headers and the cutoff points in the content
header_phrases = ["Open in app", "Sign up", "Sign in", "Write"]
cutoff_phrases = ['![](https://miro.medium.com/v2/da:true/resize:fit:0/5c50caa54067fd622d2f0fac18392213bf92f6e2fae89b691e62bceb40885e74)', "Sign up to discover human stories that deepen your understanding of the world."]

def clean_markdown_file(input_file_path, output_file_path):
    with open(input_file_path, 'r') as file:
        content = file.readlines()

    # Remove header lines containing specific phrases
    cleaned_content = [line for line in content if not any(phrase in line for phrase in header_phrases)]
    
    # Join the cleaned content
    cleaned_text = ''.join(cleaned_content)
    
    # Find the earliest cutoff point and truncate the content after it
    for phrase in cutoff_phrases:
        cutoff_index = cleaned_text.find(phrase)
        if cutoff_index != -1:
            cleaned_text = cleaned_text[:cutoff_index]
            break
    
    # Write the cleaned content to the output file path in the "cleaned/" directory
    with open(output_file_path, 'w') as file:
        file.write(cleaned_text)

def clean_all_markdown_files(directory):
    # Create the "cleaned/" directory if it doesn't exist
    cleaned_dir = os.path.join(directory, 'cleaned')
    os.makedirs(cleaned_dir, exist_ok=True)
    
    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.md'):  # Process only markdown files
            input_file_path = os.path.join(directory, filename)
            output_file_path = os.path.join(cleaned_dir, filename)
            clean_markdown_file(input_file_path, output_file_path)
            print(f"Cleaned: {filename} -> {output_file_path}")

# Specify the directory containing the markdown files
directory_path = '/Users/tre/Documents/aset/projects/gpt/nootroscripts/output/mds/ul-mcclure-medium/'

# Apply cleaning to all markdown files in the directory
clean_all_markdown_files(directory_path)