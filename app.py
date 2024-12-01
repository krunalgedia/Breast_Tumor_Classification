import os
import sys
import numpy as np
import pandas as pd
import cv2
import PIL
import tensorflow as tf
from tf_explain.core.grad_cam import GradCAM
from tensorflow.keras import backend as K
#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # This tells TensorFlow to ignore the GPU


import tumorClassify.utils.metrics as metrics

import streamlit as st
from tumorClassify.config.config import ConfigurationManager
#from tumorClassify.utils.classification_utils import TrainValEval
from tumorClassify.components.data_train_and_validation import TrainValEval
from pathlib import Path

#paths = p.Paths()
#config = c.configuration()

config_params = ConfigurationManager()
data_training_validation_config = config_params.get_data_training_validation_config()

config = config_params.config
params = config_params.params

trainValEval = TrainValEval(config, params)

def make_df(predictions, class_labels=['normal', 'benign', 'malignant']):
    df = pd.DataFrame({'Class': class_labels, 'Prediction in %': predictions[0]})
    df['Prediction in %'] = pd.to_numeric(df['Prediction in %'], errors='coerce')
    df = df.sort_values(by='Prediction in %', ascending=False)
    df['Prediction in %'] = df['Prediction in %'].apply(lambda x: round(x*100,1))
    return df


def get_DETECTION_col2(path, upload=False):

    test_img = trainValEval.load_image(path=path, upload=upload)
    test_img = np.expand_dims(test_img, axis=0)
    pred = model.predict(test_img)

    target_class = np.argmax(pred, axis=1)[0]
    explainer = GradCAM()
    grad_cam_img = explainer.explain((test_img, test_img[0, ...]),
                                     model.get_layer(params.model_params.MODEL.lower()),
                                     layer_name=params.model_params.GRADCAM_LAYER,
                                     class_index=target_class,
                                     image_weight=0.70)
    df = make_df(pred)

    condition = df.iloc[0, 0]
    st.header(':violet[AI Model prediction:] ' + ':red[' + condition + ']')
    st.table(df)
    st.header(':red[AI model focus area]')
    st.image(grad_cam_img,
             caption='Grad CAM image',
             use_column_width=True
             )


st.set_page_config(
    page_title='AI Brain tumour Detection and Segmentation',
    layout='wide',
    initial_sidebar_state='expanded'
)
st.write("<style>div.Widget.stTitle {margin-top: -80px;}</style>", unsafe_allow_html=True)
st.title('AI Brain tumour Detection and Segmentation')
st.sidebar.header('Task & Config')

model_type = st.sidebar.radio("Select Task", ['Detection', 'Segmentation'])
thres_type = st.sidebar.radio("Sharpness Threshold (Seg. Model)", ['False', 'True'])

thres = None
if thres_type == 'True':
    thres = st.sidebar.slider('Sharpness', min_value=0.0, max_value=1.0, value=0.15)

if model_type == 'Detection':
    checkpoint_dir = Path("artifacts") / "models"
    model_checkpoint_path = checkpoint_dir / "model.h5"
    #model_path = paths.DETECTION_MODEL
    custom_objects = {'recall_c0': metrics.recall_c0, 'recall_c1': metrics.recall_c1, 'recall_c2': metrics.recall_c2,
                      'precision_c0': metrics.precision_c0, 'precision_c1': metrics.precision_c1,
                      'precision_c2': metrics.precision_c2}
    #model = tf.keras.models.load_model(str(model_checkpoint_path), custom_objects=custom_objects, safe_mode=False)
    try:
        model = tf.keras.models.load_model(str(model_checkpoint_path), custom_objects=custom_objects, safe_mode=False)
    except Exception as ex:
        st.error(f'Unable to load the pretrained model with weight at {model_checkpoint_path}')
        st.error(ex)


source_img = None

if model_type == 'Detection':
    col1, col2 = st.columns(2, gap='large')
    source_img = st.sidebar.file_uploader('Choose an image...', type=('jpg', 'jpeg', 'png'))
    with col1:
        try:
            if source_img is None:
                st.header(':blue[Example User Ultrasound Scan]')
                uploaded_image = PIL.Image.open(os.path.join('artifacts', 'default_images', 'benign (1).png'))
                st.image(uploaded_image,
                         caption='Example Ultrasound Scan',
                         use_column_width=True
                         )
            else:
                st.header(':blue[User Ultrasound Scan]')
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img,
                         caption='Uploaded Ultrasound Scan',
                         use_column_width=True
                         )
        except Exception as ex:
            st.error('Error occurred while opening the image.')
            st.error(ex)
    with col2:
        try:
            if source_img is None:
                get_DETECTION_col2(path=os.path.join('artifacts', 'default_images', 'benign (1).png'))
            else:
                get_DETECTION_col2(path=source_img, upload=True)

        except Exception as ex:
            st.error('Error occurred while opening the image.')
            st.error(ex)

