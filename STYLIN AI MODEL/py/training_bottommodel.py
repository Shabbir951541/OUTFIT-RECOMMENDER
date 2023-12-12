import os
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
 
# Load the data
data = pd.read_csv('d:/fashion/styles.csv', error_bad_lines=False)
data['image'] = data.apply(lambda row: str(row['id']) + ".jpg", axis=1)
 
# Filter for bottom-wear items
bottom_wear_types = ['Jeans', 'Shorts','Track Pants', 'Trousers', 'Skirts', 'Leggings','Salwar and Dupatta']  # Adjust as necessary to match your dataset
data = data[data['subCategory'] == 'Bottomwear']
data = data[data['articleType'].isin(bottom_wear_types)]
 
# Ensure there are no missing image files
image_dir = 'd:/fashion/images'
data = data[data['image'].apply(lambda x: os.path.isfile(os.path.join(image_dir, x)))]
 
 
# Set up data generators
datagen = ImageDataGenerator(rescale=1./255., validation_split=0.2)
 
train_generator = datagen.flow_from_dataframe(
    dataframe=data,
    directory=image_dir,
    x_col="image",
    y_col="articleType",  # Use the column that contains the type of bottom-wear
    subset="training",
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode="categorical",
    target_size=(60,80)
)
print(train_generator.class_indices)
valid_generator = datagen.flow_from_dataframe(
    dataframe=data,
    directory=image_dir,
    x_col="image",
    y_col="articleType",  # Use the column that contains the type of bottom-wear
    subset="validation",
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode="categorical",
    target_size=(60,80)
)
 
 
 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
 
model = Sequential([
    Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(60, 80, 3)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
   
    # Second convolutional block
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
   
    # Third convolutional block
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
   
    # Fourth convolutional block
    Conv2D(512, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
   
    # Flattening the layers
    Flatten(),
   
    # Fully connected layers
    Dense(1024, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
   
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
   
    # The output layer with softmax activation for classification
    Dense(len(train_generator.class_indices), activation='softmax') # Set to the number of color classes
])
 
# Compile the model# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
 
 
 
# Print out the model summary
model.summary()
 
 
# Train the model
history = model.fit(train_generator,
                    steps_per_epoch=train_generator.samples // train_generator.batch_size,
                    validation_data=valid_generator,
                    validation_steps=valid_generator.samples // valid_generator.batch_size,
                    epochs=50)  # Adjust the number of epochs as needed
 
 
# Save the model
model.save('bottomwear_classifier_model.h5')
 
