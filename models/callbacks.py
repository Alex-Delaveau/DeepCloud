
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
import os

from utils.config import Config

def get_callbacks(model_name='unet', monitor='val_loss'):
    callbacks = [
        EarlyStopping(
            monitor=monitor,
            patience=10,
            verbose=1,
            mode='min',
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor=monitor,
            factor=0.1,
            patience=5,
            verbose=1,
            mode='min',
            min_delta=0.0001,
            cooldown=0,
            min_lr=1e-7
        ),
        ModelCheckpoint(
            filepath=os.path.join(Config.SAVE_PATH,f'best_{model_name}.keras'),
            monitor=monitor,
            verbose=1,
            save_best_only=True,
            mode='min'
        ),
        TensorBoard(
            log_dir=os.path.join(Config.LOG_DIR, model_name),
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            update_freq='epoch',
            profile_batch=2,
            embeddings_freq=1
        )
    ]
    return callbacks
