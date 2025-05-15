#!/usr/bin/python3
import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.applications.mobilenet_v2 import( # type: ignore
    MobileNetV2,
    preprocess_input,
    decode_predictions
)
from PIL import Image

#we are using a light weight pretrained model from tensor flow
def load_model():
    #this below is a CNN(machine learning)
    model=MobileNetV2(weights="imagenet")
    return model

#we are processing the image
def preprocess_image(image):
    #convert any image uploaded and convert to correct format
    #converts the image into a bunch of numbers
    img=np.array(image)
    img=cv2.resize(img,(224, 224))
    img=preprocess_input(img)
    #to expect multiple images
    img=np.expand_dims(img, axis=0)
    return img

def classify_image(model, image):
    try:
        processed_image=preprocess_image(image)
        predictions=model.predict(processed_image)
        #we want to take the top 3 predictions
        decoded_predictions=decode_predictions(predictions, top=3)[0]
        return decoded_predictions
    except Exception as e:
        st.error(f"Error classifying image: {str(e)}")
        return None
    
def main():
    st.set_page_config(page_title="AI Image Classifier", page_icon="🖼", layout="centered")

    st.title("AI Image Classifier 🤖")
    st.write("Upload an image and let AI tell you what is in it!")

    #when the model is loaded its not reloaded as it takes time
    @st.cache_resource
    def load_cached_model():
        return load_model()
    
    model=load_cached_model()

    uploaded_file=st.file_uploader("Choose an image..", type=["jpg", "png"])

    if uploaded_file is not None:
        image=st.image(
            uploaded_file, caption="Uploaded Image", use_container_width=True
        )

        btn=st.button("Classify Image")

        if btn:
            with st.spinner("Analyzing Image.."):
                image=Image.open(uploaded_file)
                predictions=classify_image(model,image)

                if predictions:
                    st.subheader("Predictions")
                    for _,label, score in predictions:
                        st.write(f"**{label}**:{score:.2%}")


if __name__=="__main__":
    main()
