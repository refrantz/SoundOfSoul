import subprocess
import os
import shutil
from tempfile import NamedTemporaryFile

def resample_wav(input_path, sample_rate=16000):
    """
    Resample a WAV file to the specified sample rate using FFmpeg,
    overwrites the original file with the resampled file.
    """
    # Create a temporary file with .wav extension for compatibility
    with NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        tmp_file_path = tmp_file.name

    # Build the FFmpeg command to adjust the sample rate
    cmd = [
        'ffmpeg',
        '-y',  # Overwrite without asking
        '-i', input_path,  # Input file path
        '-ar', str(sample_rate),  # Set the audio sample rate
        '-acodec', 'pcm_s16le',  # Convert audio to PCM 16-bit little-endian
        '-ac', '1',  # Convert to mono if desired
        tmp_file_path  # Output to temporary file
    ]

    try:
        subprocess.run(cmd, check=True, stderr=subprocess.PIPE)
        # Replace the original file with the new file
        shutil.move(tmp_file_path, input_path)
        print(f"File resampled and overwritten successfully: {input_path}")
    except subprocess.CalledProcessError as e:
        # Clean up temporary file on failure
        os.unlink(tmp_file_path)
        print(f"Error processing {input_path}: {e.stderr.decode()}")

def adjust_sample_rate_of_directory(root_directory, sample_rate=16000):
    """
    Adjust the sample rate of all WAV files in the specified directory and its subdirectories.
    """
    for root, dirs, files in os.walk(root_directory):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                print(f"Resampling: {file_path}")
                resample_wav(file_path, sample_rate)

# Example usage
if __name__ == "__main__":
    root_dir = 'C:/Users/renan/Desktop/PUCRS/9_SEMESTRE/TCC2/MedleyDB_V1/V1'  # Specify the root directory containing the WAV files
    adjust_sample_rate_of_directory(root_dir)
