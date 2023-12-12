import os
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16  # Using VGG16 for transfer learning
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from sklearn.utils import class_weight  # For handling class imbalance
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import BatchNormalization
import PIL

# Load the data into a Pandas DataFrame
data = pd.read_csv('d:/fashion/styles.csv', error_bad_lines=False)
data['image'] = data.apply(lambda row: str(row['id']) + ".jpg", axis=1)

# Filter out the rows without an image file
image_dir = 'd:/fashion/images'
data = data[data['image'].apply(lambda x: os.path.isfile(os.path.join(image_dir, x)))]

# Define your categories of interest here
categories_of_interest = ['Men', 'Women', 'Boys', 'Girls', 'Unisex']

# Filter the dataframe to include only the rows with the article types you're interested in
data = data[data['gender'].isin(categories_of_interest)]

# Check class distribution and balance if necessary
class_counts = data['gender'].value_counts()
print(class_counts)

# Handling class imbalance
class_weights = class_weight.compute_class_weight(
    'balanced', classes=np.unique(data['gender']), y=data['gender'].values)
class_weights = dict(enumerate(class_weights))

# Preprocessing and Augmentation
datagen = ImageDataGenerator(
    rescale=1./255.,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Model Definition
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(60, 80, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)

# Prepare the data generator for training
train_generator = datagen.flow_from_dataframe(
    dataframe=data,
    directory=image_dir,
    x_col="image",
    y_col="gender",
    subset="training",
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode="categorical",
    target_size=(60,80)
)

# Prepare the data generator for validation
valid_generator = datagen.flow_from_dataframe(
    dataframe=data,
    directory=image_dir,
    x_col="image",
    y_col="gender",
    subset="validation",
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode="categorical",
    target_size=(60,80)
)

predictions = Dense(len(train_generator.class_indices), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile the Model
from tensorflow.keras.metrics import Precision, Recall

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy', Precision(), Recall()]
)

# Train the Model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=valid_generator,
    validation_steps=valid_generator.samples // valid_generator.batch_size,
    epochs=10,
    class_weight=class_weights
)

# Save the Model and the Class Indices
model.save('gender_classifier_complex_model.h5')
# Save class indices to a file for later use
class_indices = train_generator.class_indices
import json
with open('class_indices.json', 'w') as file:
    json.dump(class_indices, file)
