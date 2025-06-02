import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import pandas as pd
import keras
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
import shutil
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from flask import Flask, render_template, request, redirect, url_for, flash, session
from tensorflow.keras.optimizers import Adam, Adagrad
from sklearn.metrics import classification_report, confusion_matrix
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'models'
app.secret_key = 'secret123'

inp_num_classes = 7
inp_epochs = 30
inp_batch_size_train = 32
inp_batch_size_test = 12
inp_initial_learning_rate = 0.001

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)
model_path = os.path.join(MODEL_FOLDER, "improved_image_model.keras")

allowed_extensions = {'png'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

@app.route("/", methods=["GET", "POST"])
def upload_dataset():
    if request.method == "POST":
        file = request.files.get("dataset")
        model_type = request.form.get("model_type")
        session['model_type'] = model_type
        if not file or file.filename == "":
            flash("No dataset selected")
            return redirect(request.url)
        shutil.rmtree(UPLOAD_FOLDER, ignore_errors=True)
        os.makedirs(UPLOAD_FOLDER)
        dataset_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(dataset_path)
        with zipfile.ZipFile(dataset_path, "r") as zip_ref:
            zip_ref.extractall(UPLOAD_FOLDER)
        os.remove(dataset_path)
        extracted_root = os.path.join(UPLOAD_FOLDER, os.listdir(UPLOAD_FOLDER)[0])
        if os.path.exists(extracted_root):
            for subfolder in os.listdir(extracted_root):
                shutil.move(os.path.join(extracted_root, subfolder), UPLOAD_FOLDER)
            shutil.rmtree(extracted_root)
        flash("Dataset uploaded successfully. Training started.")
        return redirect(url_for("train_model"))
    return render_template("index.html")

@app.route("/train", methods=["GET"])
def train_model():
    model_type = session.get('model_type', 'cnn')

    if model_type == 'dcnn':
        trdata = ImageDataGenerator(
            validation_split=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            rescale=1.0/255.0
        )
        traindata = trdata.flow_from_directory(
            directory=UPLOAD_FOLDER,
            target_size=(200, 500),
            color_mode="grayscale",
            batch_size=inp_batch_size_train,
            subset="training",
            class_mode="categorical",
            shuffle=True
        )
        testdata = trdata.flow_from_directory(
            directory=UPLOAD_FOLDER,
            target_size=(200, 500),
            color_mode="grayscale",
            batch_size=inp_batch_size_test,
            subset="validation",
            class_mode="categorical",
            shuffle=False
        )
        model = Sequential()
        model.add(Conv2D(input_shape=(200, 500, 1), filters=2, kernel_size=(3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(filters=4, kernel_size=(3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(filters=1, kernel_size=(3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(filters=2, kernel_size=(3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Flatten())
        model.add(Dense(units=64))
        model.add(Activation("relu"))
        model.add(Dropout(0.1))
        model.add(Dense(units=inp_num_classes))
        model.add(Activation("softmax"))

        opt = Adagrad(learning_rate=0.1)
        model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

        history = model.fit(
            traindata,
            validation_data=testdata,
            epochs=inp_epochs,
            steps_per_epoch=traindata.samples // inp_batch_size_train,
            validation_steps=testdata.samples // inp_batch_size_test,
        )
        model.save(os.path.join(MODEL_FOLDER, "dcnn_model.keras"))

        scores = model.evaluate(testdata, steps=testdata.samples // inp_batch_size_test, verbose=1)
        accuracy = "Accuracy:%.2f%%" % (scores[1] * 100)
        Y_pred = model.predict(testdata, testdata.samples // inp_batch_size_test, verbose=1)
        y_pred = np.argmax(Y_pred, axis=1)
        class_labels = list(testdata.class_indices.keys())
        conf_matrix = confusion_matrix(testdata.classes, y_pred)
        class_report = classification_report(testdata.classes, y_pred, target_names=class_labels)
        return render_template('train.html', accuracy=accuracy,
                               confusion_matrix=conf_matrix,
                               classification_report=class_report,
                               show_result=True)

    elif model_type == 'vit':
        IMAGE_SIZE = 128
        PATCH_SIZE = 16
        NUM_CLASSES = inp_num_classes
        BATCH_SIZE = inp_batch_size_train
        EPOCHS = inp_epochs
        projection_dim = 64
        transformer_layers = 4
        num_heads = 4
        mlp_dim = 128

        trdata = ImageDataGenerator(
            validation_split=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            rescale=1.0/255.0
        )
        traindata = trdata.flow_from_directory(
            directory=UPLOAD_FOLDER,
            target_size=(IMAGE_SIZE, IMAGE_SIZE),
            color_mode="grayscale",
            batch_size=BATCH_SIZE,
            subset="training",
            class_mode="categorical",
            shuffle=True
        )
        testdata = trdata.flow_from_directory(
            directory=UPLOAD_FOLDER,
            target_size=(IMAGE_SIZE, IMAGE_SIZE),
            color_mode="grayscale",
            batch_size=BATCH_SIZE,
            subset="validation",
            class_mode="categorical",
            shuffle=False
        )

        class PatchEmbedding(layers.Layer):
            def __init__(self, num_patches, projection_dim):
                super(PatchEmbedding, self).__init__()
                self.projection = layers.Dense(projection_dim)
                self.position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)

            def call(self, patch):
                positions = tf.range(start=0, limit=tf.shape(patch)[-2], delta=1)
                embedded = self.projection(patch) + self.position_embedding(positions)
                return embedded

        def build_vit(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1),
                      patch_size=PATCH_SIZE,
                      num_classes=NUM_CLASSES,
                      projection_dim=projection_dim,
                      transformer_layers=transformer_layers,
                      num_heads=num_heads,
                      mlp_dim=mlp_dim):

            inputs = layers.Input(shape=input_shape)
            x = layers.Resizing(IMAGE_SIZE, IMAGE_SIZE)(inputs)
            x = layers.Conv2D(filters=projection_dim,
                              kernel_size=patch_size,
                              strides=patch_size,
                              padding='valid')(x)
            x = layers.Reshape((-1, projection_dim))(x)

            embedding_layer = PatchEmbedding(num_patches=x.shape[1], projection_dim=projection_dim)
            x = embedding_layer(x)

            for _ in range(transformer_layers):
                x1 = layers.LayerNormalization(epsilon=1e-6)(x)
                attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim)(x1, x1)
                x2 = layers.Add()([attention, x])

                x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
                mlp = layers.Dense(mlp_dim, activation='gelu')(x3)
                mlp = layers.Dense(projection_dim)(mlp)
                x = layers.Add()([mlp, x2])

            x = layers.LayerNormalization(epsilon=1e-6)(x)
            x = layers.GlobalAveragePooling1D()(x)
            x = layers.Dense(mlp_dim, activation='relu')(x)
            x = layers.Dropout(0.3)(x)
            outputs = layers.Dense(num_classes, activation='softmax')(x)

            return models.Model(inputs=inputs, outputs=outputs)

        vit_model = build_vit()
        vit_model.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])

        checkpoint_cb = ModelCheckpoint(os.path.join(MODEL_FOLDER, 'vit_model.keras'),
                                        save_best_only=True, monitor='val_loss')
        early_stopping_cb = EarlyStopping(monitor='val_loss', patience=10)

        history = vit_model.fit(
            traindata,
            validation_data=testdata,
            epochs=EPOCHS,
            callbacks=[checkpoint_cb, early_stopping_cb],
            steps_per_epoch=traindata.samples // BATCH_SIZE,
            validation_steps=testdata.samples // BATCH_SIZE
        )

        scores = vit_model.evaluate(testdata, steps=testdata.samples // BATCH_SIZE)
        accuracy = "Accuracy:%.2f%%" % (scores[1] * 100)
        Y_pred = vit_model.predict(testdata, testdata.samples // BATCH_SIZE)
        y_pred = np.argmax(Y_pred, axis=1)
        class_labels = list(testdata.class_indices.keys())
        conf_matrix = confusion_matrix(testdata.classes, y_pred)
        class_report = classification_report(testdata.classes, y_pred, target_names=class_labels)
        return render_template('train.html', accuracy=accuracy,
                               confusion_matrix=conf_matrix,
                               classification_report=class_report,
                               show_result=True)

    else:  # default CNN
        trdata = ImageDataGenerator(
            validation_split=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            rescale=1.0/255.0
        )
        traindata = trdata.flow_from_directory(
            directory=UPLOAD_FOLDER,
            target_size=(200, 500),
            color_mode="grayscale",
            batch_size=inp_batch_size_train,
            subset="training",
            class_mode="categorical",
            shuffle=True
        )
        testdata = trdata.flow_from_directory(
            directory=UPLOAD_FOLDER,
            target_size=(200, 500),
            color_mode="grayscale",
            batch_size=inp_batch_size_test,
            subset="validation",
            class_mode="categorical",
            shuffle=False
        )

        model = Sequential()
        model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(200, 500, 1), activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(inp_num_classes, activation='softmax'))

        model.compile(optimizer=Adam(inp_initial_learning_rate),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        history = model.fit(
            traindata,
            validation_data=testdata,
            epochs=inp_epochs,
            steps_per_epoch=traindata.samples // inp_batch_size_train,
            validation_steps=testdata.samples // inp_batch_size_test,
        )

        model.save(model_path)

        scores = model.evaluate(testdata, steps=testdata.samples // inp_batch_size_test)
        accuracy = "Accuracy:%.2f%%" % (scores[1] * 100)
        Y_pred = model.predict(testdata, testdata.samples // inp_batch_size_test)
        y_pred = np.argmax(Y_pred, axis=1)
        class_labels = list(testdata.class_indices.keys())
        conf_matrix = confusion_matrix(testdata.classes, y_pred)
        class_report = classification_report(testdata.classes, y_pred, target_names=class_labels)

        return render_template('train.html', accuracy=accuracy,
                               confusion_matrix=conf_matrix,
                               classification_report=class_report,
                               show_result=True)

@app.route("/test", methods=["GET", "POST"])
def test_model():
    if request.method == "POST":
        file = request.files.get("test_image")
        if not file or file.filename == "":
            flash("No image selected for testing")
            return redirect(request.url)
        if not allowed_file(file.filename):
            flash("Only PNG images allowed")
            return redirect(request.url)
        filename = secure_filename(file.filename)
        test_image_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(test_image_path)

        model_type = session.get('model_type', 'cnn')

        if model_type == 'dcnn':
            model = load_model(os.path.join(MODEL_FOLDER, "dcnn_model.keras"))
            img = load_img(test_image_path, color_mode="grayscale", target_size=(200, 500))
            x = img_to_array(img) / 255.0
            x = np.expand_dims(x, axis=0)  # batch dimension
            preds = model.predict(x)
            predicted_class = np.argmax(preds, axis=1)[0]
            # You can map predicted_class to class names if you save class indices in session or file.
            return f"Predicted class index: {predicted_class}"

        elif model_type == 'vit':
            model = load_model(os.path.join(MODEL_FOLDER, 'vit_model.keras'))
            img = load_img(test_image_path, color_mode="grayscale", target_size=(128, 128))
            x = img_to_array(img) / 255.0
            x = np.expand_dims(x, axis=0)
            preds = model.predict(x)
            predicted_class = np.argmax(preds, axis=1)[0]
            return f"Predicted class index: {predicted_class}"

        else:  # default CNN
            model = load_model(model_path)
            img = load_img(test_image_path, color_mode="grayscale", target_size=(200, 500))
            x = img_to_array(img) / 255.0
            x = np.expand_dims(x, axis=0)
            preds = model.predict(x)
            predicted_class = np.argmax(preds, axis=1)[0]
            return f"Predicted class index: {predicted_class}"

    return render_template("test.html")

if __name__ == "__main__":
    app.run(debug=True)
