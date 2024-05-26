import tensorflow as tf
import os
import seqio
from frechet_audio_distance import FrechetAudioDistance
import random

frechet = FrechetAudioDistance(model_name="vggish", sample_rate=16000)

vocab_size=9344

def fad_metric(targets, predictions):
    # Assuming `targets` and `predictions` are paths to audio files or directories containing audio files
    # You will need to adjust this based on how your data is structured
    fad_score = frechet.score(targets, predictions)
    return {"FAD": fad_score}

def load_data(split, shuffle_files=False, seed=10):
    base_dir = "appended_tokens"
    
    def generator():
        # Optionally shuffle input files
        input_files = tf.io.gfile.listdir(os.path.join(base_dir, "double_bass_input", split))
        if shuffle_files:
            random.shuffle(input_files)
        
        for file_name in input_files:
            input_path = os.path.join(base_dir, "double_bass_input", split, file_name)
            output_path = os.path.join(base_dir, "double_bass_output", split, file_name)
            
            with tf.io.gfile.GFile(input_path, "r") as input_file, \
                 tf.io.gfile.GFile(output_path, "r") as output_file:
                input_text = input_file.read().splitlines()
                output_text = output_file.read().splitlines()
                yield {"inputs": input_text, "targets": output_text}

    # Create TensorFlow Dataset from the generator
    output_types = {"inputs": tf.int32, "targets": tf.int32}
    output_shapes = {"inputs": tf.TensorShape([None]), "targets": tf.TensorShape([None])}
    dataset = tf.data.Dataset.from_generator(generator, output_types=output_types, output_shapes=output_shapes)

    #for sample in dataset.take(5):  # Modify the number in take() to see more or fewer examples
    #    print("Inputs:", sample['inputs'].numpy())
    #    print("Targets:", sample['targets'].numpy())

    return dataset



seqio.TaskRegistry.add(
    name="SoundOfSoul",
    source=seqio.FunctionDataSource(
        dataset_fn=load_data,
        splits={"train", "test", "validation"}
    ),
    output_features={
        "inputs": seqio.Feature(vocabulary=seqio.PassThroughVocabulary(vocab_size), add_eos=True, dtype=tf.int32),
        "targets": seqio.Feature(vocabulary=seqio.PassThroughVocabulary(vocab_size), add_eos=True, dtype=tf.int32)
    }
)
