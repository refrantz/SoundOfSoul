git clone https://github.com/refrantz/text-to-text-transfer-transformer_singsong
git clone https://github.com/refrantz/t5x_singsong
git clone https://github.com/google/airio
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install --upgrade clu
git clone https://github.com/google/flax.git
pip install --user flax
pip install -e ./airio
pip install -e ./t5x_singsong
pip install -e ./text-to-text-transfer-transformer_singsong
