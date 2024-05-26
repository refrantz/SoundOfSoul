import os

def remove_last_empty_line(directory):
    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        
        # Check if it's a file
        if os.path.isfile(filepath):
            # Open the file, read lines, and strip any trailing newline or whitespace
            with open(filepath, 'r', encoding='utf-8') as file:
                lines = [line.rstrip('\n') for line in file]

            # Remove the last line if it is empty
            if lines and lines[-1] == '':
                lines.pop()

            # Write the modified lines back to the file, ensuring no extra newline is added at the end
            with open(filepath, 'w', encoding='utf-8') as file:
                for i, line in enumerate(lines):
                    # Add a newline character to all but the last line
                    if i < len(lines) - 1:
                        file.write(f"{line}\n")
                    else:
                        file.write(line)

# Usage
directory_path = 'appended_tokens/double_bass_output/test'
remove_last_empty_line(directory_path)
