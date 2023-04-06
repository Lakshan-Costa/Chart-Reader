import numpy as np
from sklearn.metrics import f1_score
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#load model
model = load_model("newModel.h5")

#directory of dataset
data_dir = "Images"

# Define the size of your input 
img_size = (300, 300)

# Define the batch size for training and validation data generators
batch_size = 256

# Create an instance of the ImageDataGenerator class for data augmentation and preprocessing
train_data_gen = ImageDataGenerator(
    rescale=1./255,          # rescale pixel values to [0,1]
    validation_split=0.2     # set aside 20% of data for validation
)

# Create the training data generator
train_generator = train_data_gen.flow_from_directory(
    data_dir,                        # directory of the dataset
    target_size=img_size,
    batch_size=batch_size,    
    class_mode='categorical',
    shuffle=True,                  
    subset='training'
)

# Create the validation data generator
validation_generator = train_data_gen.flow_from_directory(
    data_dir,           
    target_size=img_size,            
    batch_size=batch_size,           
    class_mode='categorical',       
    shuffle=False,                   
    subset='validation'              
)

# truth labels
y_true = validation_generator.classes

# Define the predicted labels from the model
# This is just an example, you would need to replace this with the actual predicted labels
y_pred_probs = model.predict(validation_generator)

# Convert the predicted class probabilities to class labels
y_pred = np.argmax(y_pred_probs, axis=1)

# Calculate the F1 score for each class
f1_scores = f1_score(y_true, y_pred, average=None)

# Print the F1 score for each class
num_classes = len(validation_generator.class_indices)
for i in range(num_classes):
    class_name = list(validation_generator.class_indices.keys())[i]
    print(f"F1 score for class {class_name}: {f1_scores[i]}")
