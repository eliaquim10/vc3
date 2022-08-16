cd /opt/notebooks/
conda activate super
python Samsung_SuperResolution_Dataset.py
# docker exec super1 /bin/bash -i /opt/notebooks/run.sh
#tensorboard --port 35781  --logdir logs/gradient_tape --load_fast=false  --bind_all