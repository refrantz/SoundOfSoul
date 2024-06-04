import os

print("hello")
def load_tokens(filename):
    with open(filename, 'r') as file:
        tokens = file.read().splitlines()
    return tokens

def write_combined_file(semantic_tokens, acoustic_tokens, output_file):
    with open(output_file, 'w') as out_file:
        out_file.writelines(f"{token}\n" for token in semantic_tokens)
        
        # Process each dimension of acoustic tokens
        for token_line in acoustic_tokens:
            tokens = token_line.split()
            for dim in range(4):
                print_value = int(float(tokens[dim]))+((dim+1)*1024)
                out_file.write(f"{print_value}\n")

def process_songs(semantic_tokens_dir, acoustic_tokens_dir, output_dir):
    for filename in os.listdir(semantic_tokens_dir):
        semantic_tokens_file = os.path.join(semantic_tokens_dir, filename)
        acoustic_tokens_file = os.path.join(acoustic_tokens_dir, filename)
        output_file = os.path.join(output_dir, filename)

        if os.path.exists(acoustic_tokens_file):
            semantic_tokens = load_tokens(semantic_tokens_file)
            if(len(semantic_tokens) == 500):
                acoustic_tokens = load_tokens(acoustic_tokens_file)
                write_combined_file(semantic_tokens, acoustic_tokens, output_file)
        else:
            print(f"No acoustic file found for {filename}")

# Example usage
semantic_dir = 'C:/Users/renan/Desktop/PUCRS/9_SEMESTRE/TCC2/semantical_tokens/drums_output_toGuit'
acoustic_dir = 'C:/Users/renan/Desktop/PUCRS/9_SEMESTRE/TCC2/acoustical_tokens/drums_output_toGuit'
output_dir = 'C:/Users/renan/Desktop/PUCRS/9_SEMESTRE/TCC2/appended_tokens/drums_output_toGuit/'

process_songs(semantic_dir, acoustic_dir, output_dir)