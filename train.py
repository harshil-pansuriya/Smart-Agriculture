import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from pathlib import Path
from utils.preprocessor import create_data_generator

def build_model(num_classes):
    
    base_model=MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_model.trainable=False
    
    x=base_model.output 
    x = GlobalAveragePooling2D()(x)
    x=Dense(256, activation='relu')(x)
    x=Dropout(0.2)(x)
    predictions= Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model=Model(inputs=base_model.input, outputs=predictions)
    
    # compile
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_model():
    # Paths
    data_dir = Path('data')
    train_dir = data_dir / 'train'
    test_dir = data_dir / 'test'
    
    # Parameters
    batch_size = 32
    image_size = (224, 224)
    epochs = 20
    
    # create data generator
    datagen=create_data_generator()
    
    # create generators
    train_generator=datagen.flow_from_directory(
        train_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    test_generator=datagen.flow_from_directory(
        test_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    # Build Models
    num_classes=len(train_generator.class_indices)
    model=build_model(num_classes)
    
    # callbacks
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint(
            'models/best_model.h5',
            save_best_only=True,
            monitor='val_accuracy'
        ),
        tf.keras.callbacks.EarlyStopping(
            patience=5,
            restore_best_weights=True
        )
    ]
    
    # train
    history=model.fit(
        train_generator,
        validation_data=test_generator,
        epochs=epochs,
        callbacks=callbacks
    )
    
    # Save class indices
    import json
    with open('models/class_indices.json', 'w') as f:
        json.dump(train_generator.class_indices, f)

    return history

if __name__ == '__main__':
    train_model()