import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint
from sklearn.metrics import classification_report
from PIL import Image
import io
import cv2
import ipywidgets as widgets
from IPython.display import display, clear_output

image_size = 224
categories = ['glioma', 'meningioma', 'pituitary', 'notumor']


X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')


y_train = tf.keras.utils.to_categorical(y_train, num_classes=len(categories))
y_test = tf.keras.utils.to_categorical(y_test, num_classes=len(categories))


model_path = 'effnet.keras'
if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
    print("Loaded existing model from", model_path)
else:
    effnet = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
    model = effnet.output
    model = tf.keras.layers.GlobalAveragePooling2D()(model)
    model = tf.keras.layers.Dropout(rate=0.5)(model)
    model = tf.keras.layers.Dense(len(categories), activation='softmax')(model)
    model = tf.keras.models.Model(inputs=effnet.input, outputs=model)
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

# Define callbacks
tensorboard = TensorBoard(log_dir='logs')
checkpoint = ModelCheckpoint("effnet.keras", monitor="val_accuracy", save_best_only=True, mode="auto", verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.3, patience=2, min_delta=0.001, mode='auto', verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, validation_split=0.1, epochs=12, verbose=1, batch_size=32,
                   callbacks=[tensorboard, checkpoint, reduce_lr, early_stopping])

# Evaluate the model
pred = model.predict(X_test)
pred = np.argmax(pred, axis=1)
y_test_new = np.argmax(y_test, axis=1)

# Print classification report
print(classification_report(y_test_new, pred, target_names=categories))

# Define the image prediction function
def img_pred(upload):
    for name, file_info in uploader.value.items():
        img = Image.open(io.BytesIO(file_info['content']))
        opencvImage = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = cv2.resize(opencvImage, (image_size, image_size))
        img = img.reshape(1, image_size, image_size, 3)
        p = model.predict(img)
        p = np.argmax(p, axis=1)[0]

        if p == 0:
            p = 'Glioma Tumor'
        elif p == 1:
            p = 'No Tumor'
        elif p == 2:
            p = 'Meningioma Tumor'
        else:
            p = 'Pituitary Tumor'

        print(f'The Model predicts that it is a {p}')

uploader = widgets.FileUpload()
display(uploader)

button = widgets.Button(description='Predict')
out = widgets.Output()
def on_button_clicked(_):
    with out:  
        clear_output()
        try:
            img_pred(uploader)
        except:
            print('No Image Uploaded/Invalid Image File')
button.on_click(on_button_clicked)
widgets.VBox([button, out])
