from telegram.ext import *
from io import BytesIO
import cv2
import numpy as np
import tensorflow as tf
TOKEN = "6142858557:AAG7kVdoeMY9shi5V56BZ4jlA4_SQJDiJPg"


# ML part of bot's logic:
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255, x_test / 255

# Ten class from well-known cifar10 dataset for algorithm training
class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# working with model itself
# Images in cifar10 are size of 32 by 32, RGB
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(32, 32, 3)))
model.add(tf.keras.layers.MaxPool2D((2,2)))
model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu'))
model.add(tf.keras.layers.MaxPool2D((2,2)))
model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))


# Bot's backend logic implemented
def start(update, context):
    update.message.reply_text("Welcome to the Final Project's Bot")
    update.message.reply_text("This bot created for Advanced Programming course as Final Project")
    update.message.reply_text("Instructor Mr. Sultanmurat Yeleu, y.sultanmurat@astanait.edu.kz")
    update.message.reply_text("Bot will classify images sent by user")


def help(update, context):
    update.message.reply_text("""
    /start - To start bot
    /train - To start training CNN(Convolutional Neural Networks)
    /help - To show this message
    
    Students: IT-2105 (Computer Science)
    Serikov Yerasyl
    Serik Sultan
    """)


def train(update, context):
    #where model is actually used
    update.message.reply_text("Model in process of training _._._.")
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
    model.save('yera_image_classf.model')
    update.message.reply_text("Training process is done, please send a photo")

def handle_message(update, context):
    update.message.reply_text("Firstly train the model, then send a picture")


# Getting image from user
def handle_photo(update, context):
    file = context.bot.get_file(update.message.photo[-1].file_id)
    f = BytesIO(file.download_as_bytearray())
    file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)

    #usage of OpenCV to convert color RGB -> BGR
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (32,32), interpolation=cv2.INTER_AREA)

    #prediction part and given class of image
    prediction = model.predict(np.array([img / 255]))
    update.message.reply_text(f"In this image bot saw a/an {class_names[np.argmax(prediction)]}")


updater = Updater(TOKEN, use_context=True)
dp = updater.dispatcher

dp.add_handler(CommandHandler("start", start))
dp.add_handler(CommandHandler("train", train))
dp.add_handler(CommandHandler("help", help))
dp.add_handler(MessageHandler(Filters.text, handle_message))
dp.add_handler(MessageHandler(Filters.photo, handle_photo))

updater.start_polling()
updater.idle()