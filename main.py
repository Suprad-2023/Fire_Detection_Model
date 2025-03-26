import streamlit as st
import tensorflow as tf
import numpy as np

# Predict Function
def firedetection(img):
    model= tf.keras.models.load_model("fire_detection_model.keras")
    image= tf.keras.preprocessing.image.load_img(img, target_size=(224,224))
    input_arr= tf.keras.preprocessing.image.img_to_array(image)/255.0
    input_arr = np.expand_dims(input_arr, axis=0)
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Fire Detection"])

#HomePage
if(app_mode=='Home'):
    st.header("FOREST FIRE DETECTION SYSTEM")
    image_path = "HomePage.jpeg"
    st.image(image_path,use_container_width=True)
    st.markdown("""
    Welcome to the Forest Fire Detection System! 🌿🔍
    
    Our mission is to help Identify forest fire from the images taken from survilance cameras.
    ### How It Works
    1. **Upload Image:** Go to the **Fire Detection** page and upload an image of a suspected fire or smoke.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify if its a fire or a smoke or a non fire image.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate fire detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Fire Detection** page in the sidebar to upload an image and experience the power of our Fire Detection Model!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

#AboutPage
elif(app_mode=="About"):
    st.header("About")
    st.markdown("""
                #### About Dataset
                This dataset is provided by upGrad. This is the part of the Capstone project where we detect a forest fire using survilance cameras.
                This dataset consists of about 31500 images of Fire, Smoke and Non Fire which is categorized into 3 different classes.The total dataset is divided into 70/30 ratio of training and testing set preserving the directory structure.
                The test dataset is present in a new test directory contating 10500 images of Fire, Smoke and Non Fire images.
                #### Content
                1. train (31500 images)
                2. test (10500 images)

                """)

#PredictionPage
elif(app_mode=="Fire Detection"):
    st.header("Fire Detection")
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_image,width=4,use_container_width=True)
    #Predict button
    if(st.button("Predict")):
        st.snow()
        st.write("Our Prediction")
        result_index = firedetection(test_image)
    
        #ReadingLabels
        class_name=['Non Fire', 'Fire', 'Smoke']
        st.success("Model is Predicting it's a {}".format(class_name[result_index]))