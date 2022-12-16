import os, tensorflow as tf

print("Setting up GCS access...")
# Use legacy GCS authentication method.
os.environ['USE_AUTH_EPHEM'] = '0'
import tensorflow_gcs_config
# from google.colab import auth
# Set credentials for GCS reading/writing from Colab and TPU.
TPU_TOPOLOGY = "v2-8"
try:
tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
TPU_ADDRESS = tpu.get_master()
print('Running on TPU:', TPU_ADDRESS)
except ValueError:
raise BaseException('ERROR: Not connected to a TPU runtime; please see the previous cell in this notebook for instructions!')
# auth.authenticate_user()
tf.enable_eager_execution()
tf.config.experimental_connect_to_host(TPU_ADDRESS)
tensorflow_gcs_config.configure_gcs_from_colab_auth()

# if ON_CLOUD:
#   %reload_ext tensorboard
# %tensorboard --logdir="$MODEL_DIR" --port=0