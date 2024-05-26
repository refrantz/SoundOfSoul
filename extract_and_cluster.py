import torch
from audiolm_pytorch import EncodecWrapper
from transformers import AutoFeatureExtractor, Wav2Vec2BertModel
import soundfile as sf
from sklearn.cluster import KMeans
from pydub import AudioSegment
import yaml
import os
import numpy as np
from transformers import T5ForConditionalGeneration

torch.cuda.empty_cache()

encodec = EncodecWrapper()

print(torch.cuda.get_device_name(0))

# Assuming the model and processor have been loaded
processor = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
model = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0").to(device='cuda')

previous = "empty"
all_output_size = []

def load_metadata(yaml_path):
    with open(yaml_path, 'r') as file:
        metadata = yaml.safe_load(file)
    return metadata

def find_input_stem(metadata, instrument_name):
    for stem_key, stem_value in metadata['stems'].items():
        if stem_value['instrument'] == instrument_name:
            return stem_value['filename']  # Return the filename of the stem for the desired instrument
    return None

def audio_segment_to_tensor(audio_segment):
    """Convert AudioSegment to a PyTorch tensor."""
    # Get the waveform as a numpy array (data is a list of arrays for each channel)
    samples = np.array(audio_segment.get_array_of_samples())
    if audio_segment.channels == 2:
        samples = samples.reshape((-1, 2))
    # Convert to float32 for PyTorch, normalize by max int value for the sample width
    samples = samples.astype(np.float32) / (2**(8 * audio_segment.sample_width) / 2)
    return torch.from_numpy(samples)

def audio_segment_to_numpy(audio_segment):
    """Convert AudioSegment to a numpy array."""
    # Get the waveform as a numpy array (data is a list of arrays for each channel)
    samples = np.array(audio_segment.get_array_of_samples())
    if audio_segment.channels == 2:
        samples = samples.reshape((-1, 2))
    else:
        samples = samples.reshape((-1, 1))  # Ensure 2D array even for mono
    # Normalize by max int value for the sample width to convert to float
    samples = samples.astype(np.float32) / float(2**(8 * audio_segment.sample_width) - 1)
    return samples

def mix_and_process_segments(stems_dir, input_filename):
    segments = []
    # Load and mix the audio stems as before
    mixed_track = None
    input_track = None
    for stem_file in os.listdir(stems_dir):
        path_to_file = os.path.join(stems_dir, stem_file)
        if stem_file.endswith('.wav'):
            audio = AudioSegment.from_file(path_to_file)
            if stem_file == input_filename:
                input_track = audio
            elif mixed_track is None:
                mixed_track = audio
            else:
                mixed_track = mixed_track.overlay(audio)

    # Segment the mixed and input tracks into 10-second slices
    if mixed_track:
        for i in range(0, len(mixed_track), 10005):  # 10000 ms = 10 seconds
            segment_mixed = mixed_track[i:i+10005]
            segment_input = input_track[i:i+10005] if input_track else None
            if segment_input:
                segments.append((audio_segment_to_tensor(segment_mixed), audio_segment_to_tensor(segment_input)))
    
    return segments

def extract_layer_embeddings(waveform, layer_num=6):
    #print("started")
    # Process the audio and prepare for the model
    # Ensure we're using the appropriate method for your model and processor setup
    inputs = processor(waveform, sampling_rate=16000, return_tensors="pt")
    
    with torch.no_grad():
        inputs = inputs.to(device='cuda')
        # Ensure we're accessing the model outputs correctly
        outputs = model(**inputs, output_hidden_states=True)
        # Extract the embeddings from the specified layer
        layer_embeddings = outputs.hidden_states[layer_num]

    #print("finished embedding")
    
    return layer_embeddings.squeeze(0)

def train_and_assign_kmeans(points, num_clusters, centroids_file):
    """
    Trains a k-means model on the provided points and writes the centroids and cluster indices to files.

    :param points: A (n_points, 1024) array containing the points.
    :param num_clusters: The number of clusters for the k-means algorithm.
    :param centroids_file: The file path to save the centroids.
    :param indices_file: The file path to save the cluster indices.
    """
    mean = np.mean(points, axis=0)
    std = np.std(points, axis=0)
    normalized_embeddings = (points - mean) / std
    # Train k-means on the provided points
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(normalized_embeddings)
    
    # Get the centroids and cluster indices
    centroids = kmeans.cluster_centers_
    cluster_indices = kmeans.predict(normalized_embeddings)
    global all_output_size
    start_index = 0

    with open(centroids_file, 'w') as f:
        for centroid in centroids:
            f.write(" ".join(map(str, centroid)) + "\n")

    for output_size in all_output_size:
        num_clusters_in_segment, segment_file_name = output_size

        # Select the slice of centroids and indices for this segment
        end_index = start_index + num_clusters_in_segment
        segment_indices = cluster_indices[start_index:end_index]

        # Write cluster indices to the file for this segment
        indices_file = f"semantical_tokens/{segment_file_name}.txt"
        with open(indices_file, 'w') as f:
            for index in segment_indices:
                f.write(str(index) + "\n")

        # Update start index for the next segment
        start_index = end_index

def process_song(metadata_path, stems_dir):
    metadata = load_metadata(metadata_path)
    song_name = os.path.basename(metadata_path).split('_METADATA')[0]
    input_stem = find_input_stem(metadata, "double bass")

    if not input_stem:
        print(f"No 'double bass' stem found for {song_name}. Skipping...")
        return

    segments = mix_and_process_segments(stems_dir, input_stem)
    for idx, (mixed_audio, input_audio) in enumerate(segments):
        semantical_input = extract_layer_embeddings(mixed_audio)
        acoustical_output = encodec(mixed_audio, input_sample_hz=16000)[1]

        global previous

        if previous is not "empty":
            previous = np.concatenate((previous, semantical_input.cpu()), axis=0)
        else:
            previous = semantical_input.cpu()

        all_output_size.append([semantical_input.shape[0], f"{song_name}_segment_{idx}"])

        # Save to text files, one token per line, appending the segment index to filenames
        np.savetxt(f"acoustical_tokens/{song_name}_segment_{idx}.txt", acoustical_output, fmt='%f')
        #print(f"Processed and saved data for {song_name} segment {idx}")

def process_dataset(dataset_dir):
    metadata_dir = os.path.join(dataset_dir, "Metadata")
    stems_version_dir = os.path.join(dataset_dir, "V1")

    for song_folder in os.listdir(stems_version_dir):
        song_path = os.path.join(stems_version_dir, song_folder)
        if os.path.isdir(song_path):
            stems_dir = os.path.join(song_path, f"{song_folder}_STEMS")
            metadata_file = f"{song_folder}_METADATA.yaml"
            metadata_path = os.path.join(metadata_dir, metadata_file)
            if os.path.exists(metadata_path):
                process_song(metadata_path, stems_dir)
            else:
                print(f"Metadata not found for {song_folder}")

dataset_dir = "C:/Users/renan/Desktop/PUCRS/9_SEMESTRE/TCC2/MedleyDB_V1"
process_dataset(dataset_dir)


# Define the number of clusters
num_clusters = 1024

# Define the file paths
centroids_file = "centroids.txt"
points = previous

# Call the function
train_and_assign_kmeans(points, num_clusters, centroids_file)