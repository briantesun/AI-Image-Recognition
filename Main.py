from tensorflow import keras

base_model = keras.applications.VGG16(
    weights='imagenet',
    input_shape=(224, 224, 3),
    include_top=False)

# Model creation and compilation
base_model.trainable = False

inputs = keras.Input(shape=(224, 224, 3))

x = base_model(inputs, training=False)

x = keras.layers.GlobalAveragePooling2D()(x)

num_classes = 6
outputs = keras.layers.Dense(num_classes, activation='softmax')(x)

model = keras.Model(inputs, outputs)

model.compile(loss='categorical_crossentropy', metrics=['accuracy'])

# Data augmentation
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

# Dataset loading
train_it = datagen.flow_from_directory('data/train/', 
                                       target_size=(224, 224), 
                                       color_mode='rgb', 
                                       class_mode='categorical')

valid_it = datagen.flow_from_directory('data/valid/', 
                                       target_size=(224, 224), 
                                       color_mode='rgb', 
                                       class_mode='categorical')


# Model training
model.fit(train_it,
          validation_data=valid_it,
          steps_per_epoch=train_it.samples/train_it.batch_size,
          validation_steps=valid_it.samples/valid_it.batch_size,
          epochs=10)

# Fine-tuning
base_model.trainable = True

model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=1e-5),
              loss='categorical_crossentropy', metrics=['accuracy'])

history_fine_tuning = model.fit(train_it,
                                validation_data=valid_it,
                                steps_per_epoch=train_it.samples/train_it.batch_size,
                                validation_steps=valid_it.samples/valid_it.batch_size,
                                epochs=10)

model.evaluate(valid_it, steps=valid_it.samples/valid_it.batch_size)