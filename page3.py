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

    canvas_result = st_canvas(
        stroke_width = 15,
        stroke_color = "#000",
        background_color = "#fff",
        height = 320,
        width = 1280,
        drawing_mode = "freedraw",
        key = "canvas",
        )
    
    @st.cache
    def get_data():
        df = pd.read_csv('min100.csv') # mettre min100.csv
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
            img =cv2.resize(my_initial_image.astype(np.uint8),(128,32)) #mettre astype(np.uint8),(128,32))
            img_rescalling = (cv2.resize(img, dsize=(128,32),interpolation=cv2.INTER_NEAREST)) #mettre dsize=(128,32)
            #st.write(img_rescalling.shape)
            my_image=cv2.cvtColor(img_rescalling,cv2.COLOR_BGR2GRAY)
            
            #st.write(my_image.shape)
            st.write("Vous avez écrit:")
            st.image(my_image, width=400)
            X_test=[]
            X_test.append(my_image)
            X_test = np.array(X_test)
 
            X_test = X_test.reshape([-1,32,128,1]) #mettre reshape([-1,32,128,1])
            
            model = tf.keras.models.load_model("model_min100.h5") # mettre le 1er modèle
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

            
            model = tf.keras.models.load_model('model_ocr.h5', custom_objects={'tf': tf})
            
         

            import string
            charList = list(string.ascii_letters)+[' ']

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
            
            #st.write("X_test[:1]", X_test[:1])
            #st.write("np.expand_dims(X_test[:1], -1)",np.expand_dims(X_test[:1], -1).shape,np.expand_dims(X_test[:1], -1))
            X_test_new = np.expand_dims(X_test[:1], -1)
            
            X_test_prediction = model(X_test_new)
            #st.write("model(X_test_new)", X_test_prediction)
            
            X_test_pred_transp = tf.transpose(X_test_prediction, (1, 0, 2))
            #st.write("tf.transpose(model(X_test_new), (1, 0, 2))", X_test_pred_transp)
            
            X_test_seq = [X_test_prediction.shape[1]]*X_test_prediction.shape[0]
            #st.write("[X_test_prediction.shape[1]]*X_test_prediction.shape[0]", X_test_seq)
            
            predicted_codes, _  = tf.nn.ctc_greedy_decoder(X_test_pred_transp, X_test_seq)
            #st.write("predicted_codes[0]", predicted_codes[0])
            
            predicted_codes_str = tf.sparse.to_dense(predicted_codes[0]).numpy().astype(str)
            #st.write("predicted_codes_str", predicted_codes_str)
            
            codes = tf.cast(predicted_codes[0], tf.int32)
            #st.write("codes = tf.cast(predicted_codes[0], tf.int32)", codes)
            
            text = decode_codes(codes, charList)
            #st.write("text = decode_codes(codes, charList)", text)
           
            text = tf.sparse.to_dense(text).numpy().astype(str)
            #st.write("text = tf.sparse.to_dense(text).numpy().astype(str)", text)
            
            predi = list(map(lambda x: ''.join(x), text))
            #st.write("predi = list(map(lambda x: ''.join(x), text))", predi)
            
            st.write("Et voici les résultats de votre modèle :")
            
            st.write('**Prediction**')
            st.write(str(predi[0]))
            
            
