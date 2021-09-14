import streamlit as st
import cv2
import numpy as np
import pandas as pd
from tensorflow import keras
import matplotlib.pyplot as plt
from keras.utils import np_utils
from PIL import Image
import tensorflow as tf
from asrtoolkit import cer
from streamlit_drawable_canvas import st_canvas

def page3():
    st.title("Démonstration")
    st.title("Test de l'outil")
    canvas_result = st_canvas(
        stroke_width = 25,
        stroke_color = "#000",
        background_color = "#fff",
        height = 280,
        width = 1120,
        drawing_mode = "freedraw",
        key = "canvas",
        )
    
    @st.cache
    def get_data():
        df = pd.read_csv('min100.csv') # modifier le nom du fichier
        return df
    
    
    path_images = '...' # à modifier

    data = get_data()

    y = data["text"]
    y = np.array(y)
    y = y.reshape(-1,1)
    from sklearn import preprocessing

    enc = preprocessing.OrdinalEncoder(categories='auto')
    enc.fit(y)
    target = enc.transform(y)

    st.write("1. Choisir un modèle dans la liste déroulante.")
    st.write("2. Enfin, cliquer sur le bouton ci-dessous pour lancer le modèle choisi et afficher les résultats.")

    choix_modele = st.selectbox("Choisissez votre modèle", options = ["LeNet", "Détection d'objet"])

    

    click = st.button("Lancer le modèle et afficher les résultats")
    
    if click:
        if choix_modele == "LeNet":
            
            my_initial_image = canvas_result.image_data
            #st.write(my_initial_image.shape)
            img =cv2.resize(my_initial_image.astype(np.uint8),(128,32))
            img_rescalling = (cv2.resize(img, dsize=(128,32),interpolation=cv2.INTER_NEAREST))
            #st.write(img_rescalling.shape)
            my_image=cv2.cvtColor(img_rescalling,cv2.COLOR_BGR2GRAY)
            
            #st.write(my_image.shape)
            st.write("Vous avez écrit:")
            st.image(my_image, width=400)
            X_test=[]
            X_test.append(my_image)
            X_test = np.array(X_test)
 
            X_test = X_test.reshape([-1,32,128,1])
            
            model = keras.models.load_model("my_model_128x32.h5",custom_objects={'tf': tf})
            y_pred = model.predict(X_test/1.0)
            
            y_pred_class = y_pred.argmax(axis=1)

            target_new = target.reshape([-1])
            y_new = y.reshape([-1])

            index_pred = list(target_new).index(y_pred_class)


            st.write("Et voici les résultats de votre modèle :")
            st.write('**Prediction**')
            st.write(str(y_new[index_pred]))
        else:
            my_initial_image = canvas_result.image_data
            #st.write(my_initial_image.shape)
            img =cv2.resize(my_initial_image.astype(np.uint8),(132,28))
            img_rescalling = (cv2.resize(img, dsize=(132,28),interpolation=cv2.INTER_NEAREST))
            #st.write(img_rescalling.shape)
            my_image=cv2.cvtColor(img_rescalling,cv2.COLOR_BGR2GRAY)
            
            #st.write(my_image.shape)
            st.write("Vous avez écrit:")
            st.image(my_image, width=400)
            X_test=[]
            X_test.append(my_image)
            X_test = np.array(X_test)

            
            model = tf.keras.models.load_model('htr_model_cer.h5', custom_objects={'tf': tf})
            
            def loss(labels, logits):
                return tf.reduce_mean(
                        tf.nn.ctc_loss(
                            labels = labels,
                            logits = logits,
                            logit_length = [logits.shape[1]]*logits.shape[0],
                            label_length = None,
                            logits_time_major = False,
                            blank_index=-1
                        )
                    )

            import string
            charList = list(string.ascii_letters)+[' ']

            def encode_labels(labels, charList):
                # Hash Table
                table = tf.lookup.StaticHashTable(
                    tf.lookup.KeyValueTensorInitializer(
                        charList,
                        np.arange(len(charList)),
                        value_dtype=tf.int32
                    ),
                    0,
                    name='char2id'
                )
                return table.lookup(
                tf.compat.v1.string_split(labels, delimiter=''))   


            def decode_codes(codes, charList):
                table = tf.lookup.StaticHashTable(
                    tf.lookup.KeyValueTensorInitializer(
                        np.arange(len(charList)),
                        charList,
                        key_dtype=tf.int32
                    ),
                    '',
                    name='id2char'
                )
                return table.lookup(codes)

            def greedy_decoder(logits):
            # ctc beam search decoder
                predicted_codes, _ = tf.nn.ctc_greedy_decoder(
                    # shape of tensor [max_time x batch_size x  num_classes] 
                    tf.transpose(logits, (1, 0, 2)),
                    [logits.shape[1]]*logits.shape[0]
                )
                
                # convert to int32
                codes = tf.cast(predicted_codes[0], tf.int32)
                
                # Decode the index of caracter
                text = decode_codes(codes, charList)
                
                # Convert a SparseTensor to string
                text = tf.sparse.to_dense(text).numpy().astype(str)
                
                return list(map(lambda x: ''.join(x), text))


            X_test = tf.cast(X_test, dtype = tf.float32)
            l = greedy_decoder(model(np.expand_dims(X_test, -1)))
            
            st.write("Et voici les résultats de votre modèle :")

            st.write('**Prediction**')
            st.write(str(l[0]))
            
            
