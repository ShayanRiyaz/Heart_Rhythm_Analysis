import os
from types import SimpleNamespace

# ----------------------------- configuration -----------------------------



# config = SimpleNamespace(
#     WIN_SEC=WIN_SEC,
#     FS_ORIGINAL=FS_ORIGINAL,
#     DEC_FACTOR=DEC_FACTOR,
#     FS=FS,
#     WIN_LEN=WIN_LEN,
#     BATCH=BATCH,
#     EPOCHS=EPOCHS,
#     VAR_MULT=VAR_MULT
# )

config = SimpleNamespace()


config.WIN_SEC      = 8
config.FS_ORIGINAL  = 125
config.DEC_FACTOR   = 6
config.FS           = round(config.FS_ORIGINAL / config.DEC_FACTOR, 2)
config.WIN_LEN      = int(round(config.WIN_SEC * config.FS))
config.BATCH        = 32
config.EPOCHS       = 1000
config.VAR_MULT     = 500  # for any variance scaling if needed
# Paths
config.paths = SimpleNamespace()
config.paths.FOLDER_PATH           = 'length_full'
config.paths.BASE_DATASET_DIR      = os.path.join('data','development_dataset', config.paths.FOLDER_PATH)
config.paths.TRAIN_LOADER_PATH     = os.path.join(config.paths.BASE_DATASET_DIR,'train_dataset.h5')
config.paths.TEST_LOADER_PATH      = os.path.join(config.paths.BASE_DATASET_DIR,'test_dataset.h5')

# Checkpoint directory (for trainer.py)
config.checkpoint = SimpleNamespace()
config.checkpoint.MODEL_NAME            = "nn_new_v1_processed"
config.checkpoint.CKPT_DIR              = os.path.join('trained_model',f"{config.checkpoint.MODEL_NAME}",'checkpoints')
config.checkpoint.SAVE_EVERY            = 5  # epochs
config.checkpoint.RESUME                = False

config.model = SimpleNamespace()
config.model.C_IN = 1
config.model.BASE_CHANNELS = 32
config.model.MODEL_DEPTH = 3
config.model.DROP_OUT = 0.2
config.model.LossPosWeights =None

config.plotting = SimpleNamespace()
config.plotting.save_dir =os.path.join(os.getcwd(),'trained_model',config.checkpoint.MODEL_NAME)
config.plotting.make_plots = True
config.plotting.return_fig = True

if not os.path.exists(config.checkpoint.CKPT_DIR):
    os.makedirs(config.checkpoint.CKPT_DIR)