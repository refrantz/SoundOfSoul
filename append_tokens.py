import os


def load_tokens(filename):
    with open(filename, 'r') as file:
        tokens = file.read().splitlines()
    return tokens

def write_combined_file(semantic_tokens, acoustic_tokens, output_file):
    with open(output_file, 'w') as out_file:
        # Write the global start of sequence
        out_file.write('9217\n')

        # Write semantic tokens with their start sentinel
        out_file.write('9218\n')
        out_file.writelines(f"{token}\n" for token in semantic_tokens)
        
        # Process each dimension of acoustic tokens
        for dim in range(8):
            out_file.write(f"{9219 + dim}\n")  # Start of each acoustic dimension
            for token_line in acoustic_tokens:
                tokens = token_line.split()
                print_value = int(float(tokens[dim]))+((dim+1)*1024)
                out_file.write(f"{print_value}\n")

def process_songs(semantic_tokens_dir, acoustic_tokens_dir, output_dir):
    for filename in os.listdir(semantic_tokens_dir):
        semantic_tokens_file = os.path.join(semantic_tokens_dir, filename)
        acoustic_tokens_file = os.path.join(acoustic_tokens_dir, filename)
        output_file = os.path.join(output_dir, filename)

        if os.path.exists(acoustic_tokens_file):
            semantic_tokens = load_tokens(semantic_tokens_file)
            acoustic_tokens = load_tokens(acoustic_tokens_file)
            write_combined_file(semantic_tokens, acoustic_tokens, output_file)
        else:
            print(f"No acoustic file found for {filename}")

# Example usage
semantic_dir = 'C:/Users/renan/Desktop/PUCRS/9_SEMESTRE/TCC2/semantical_tokens/double_bass_output'
acoustic_dir = 'C:/Users/renan/Desktop/PUCRS/9_SEMESTRE/TCC2/acoustical_tokens/double_bass_output'
output_dir = 'C:/Users/renan/Desktop/PUCRS/9_SEMESTRE/TCC2/appended_tokens/double_bass_output'

process_songs(semantic_dir, acoustic_dir, output_dir)