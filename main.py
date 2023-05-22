import tkinter as tk
from tkinter import filedialog
import nibabel as nib
import numpy as np
from PIL import Image
import tensorflow as tf

# Load the trained TensorFlow Lite model
model_path = 'pd_mri_model.tflite'

# Load the model
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("input_details",input_details)
print("op",output_details)

# Define the class labels
class_labels = ['pd', 'no']

# Function to handle file upload and processing


def process_file():
    # Open file dialog to select ".nii.gz" file
    file_path = filedialog.askopenfilename(filetypes=[("NIfTI files", ".gz")])

    if file_path:
        # Read the ".nii.gz" file using nibabel
        img = nib.load(file_path)
        img_data = img.get_fdata()

        # Get the middle 5 slices
        num_slices = img_data.shape[2]
        start_slice = max(0, num_slices // 2 - 2)
        end_slice = min(num_slices, num_slices // 2 + 3)
        middle_slices = img_data[:, :, start_slice:end_slice]

        # Save the middle slices as images
        image_paths = []
        for i, slice_data in enumerate(middle_slices.transpose()):
            image = Image.fromarray(slice_data)
            # image = image.convert('greyscale')  # Convert the image to grayscale format
            image_path = f"middle_slice_{i+1}.png"
            image.save(image_path)
            image_paths.append(image_path)

        # Perform PD detection on the middle slices using the model
        test_model(image_paths)

        # Display the results
        result_label.config(text="PD detection completed")


def load_and_preprocess_images(image_paths):
    images = []
    for image_path in image_paths:
        image = Image.open(image_path)
        # Resize the image to match the input size of the model
        image = image.resize((256, 256))
        # Convert the image to grayscale and normalize pixel values to the range [0, 1]
        image = np.array(image.convert('L')) / 255.0
        # Convert the image data type to FLOAT32
        image = image.astype(np.float32)
        images.append(image)
    images = np.array(images)
    # Adjust the shape of the images array to (batch_size, num_slices, height, width, channels)
    # Add batch and channel dimensions
    images = np.expand_dims(images, axis=(0, 4))
    # Repeat the images to match the expected num_slices
    images = np.repeat(images, repeats=10, axis=1)
    return images


def test_model(image_paths):
    images = load_and_preprocess_images(image_paths)
    interpreter.set_tensor(input_details[0]['index'], images)

    # Run inference
    interpreter.invoke()

    # Get the output predictions
    predictions = interpreter.get_tensor(output_details[0]['index'])
    conf_2 = 0
    for i, pred in enumerate(predictions):
        predicted_class_index = np.argmax(pred)
        predicted_class_label = class_labels[predicted_class_index]
        confidence = pred[predicted_class_index] * 100
        conf_2 = conf_2 + confidence
        # print('Image', i+1)
        # print('Predicted Class:', predicted_class_label)
        # print('Confidence:', confidence)
        # print()
    conf_3 = conf_2 / 5
    if conf_3 > 70:
        result_label.config(text="The person has Parkinson")
    else:
        result_label.config(text="The person does not have Parkinson")


# Create the Tkinter window
window = tk.Tk()
window.title("PD Detection App")

# Create a button to upload the file
upload_button = tk.Button(window, text="Upload File", command=process_file)
upload_button.pack(pady=10)

# Create a label to display the results
result_label = tk.Label(window, text="Predictions: ")
result_label.pack(pady=10)

# Run the Tkinter event loop
window.mainloop()
