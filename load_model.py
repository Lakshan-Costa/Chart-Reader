from tensorflow import keras
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model = keras.models.load_model("Chart_Classification_79.h5")

train_datagen = ImageDataGenerator(rescale=1/255, validation_split=0.1)
train_generator = train_datagen.flow_from_directory(
    'Images',
    target_size = (300, 300),
    batch_size = 256,
    class_mode = 'categorical',
    subset = 'training'
)

# Load and preprocess the image
img = Image.open("test/1381.png")
img = img.resize((300, 300))
x = np.array(img)
x = x[:, :, :3] # Remove the 4th channel
x = x / 255. # Scale pixel values to [0, 1]
x = np.expand_dims(x, axis=0) # Add an extra dimension

# Pass the image through the model
preds = model.predict(x)

# Get the class with the highest predicted probability
class_idx = np.argmax(preds[0])

# Get the class labels from the generator
class_labels = train_generator.class_indices
class_labels = {v: k for k, v in class_labels.items()}

# Display the predicted class
print(f"The image is predicted to be a {class_labels[class_idx]}")


