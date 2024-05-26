import os

def convert_directory(input_dir, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Iterate over each file in the input directory
    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        # Read all lines, strip whitespace, and join into a single line
        with open(input_path, 'r') as file:
            lines = file.read().splitlines()
        single_line = ' '.join(lines)
        
        # Write the single line to a new file in the output directory
        with open(output_path, 'w') as file:
            file.write(single_line)

# Example usage
convert_directory('appended_tokens/double_bass_input/test', 'appended_tokens/double_bass_input/test_appended')
