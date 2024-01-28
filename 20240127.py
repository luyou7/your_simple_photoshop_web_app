import streamlit as st
import cv2
import numpy as np


st.title("Your PhotoShop App")

uploaded_file = st.file_uploader("Upload your image", type=["jpg", "jpeg", "png"])

def save_image_as_png(image):
    _, buffer = cv2.imencode('.png', img=image)
    return buffer.tobytes()


if uploaded_file is not None:
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)

    st.image(image, caption="Before (Original)", use_column_width=True, channels="BGR")


    if st.checkbox('Brightness/Contrast'):
        brightness = st.slider("Brightness", -100, 100, 0, key="brightness")
        contrast = st.slider("Contrast", -100, 100, 0, key="contrast")
        image = cv2.convertScaleAbs(image, alpha=(contrast/127.0 + 1), beta=brightness)

    if st.checkbox('Black-White Effect'):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    if st.checkbox('Negative Effect'):
        image = 255 - image

    if st.checkbox('Blur (Gaussian)'):
        kernel = st.slider('Kernel', min_value=1, max_value=35, step=2, value=15, key='blur_kernel')
        sigma = st.slider('Sigma', min_value=0, max_value=15, step=1, value=7, key='blur_sigma')
        image = cv2.GaussianBlur(image, (kernel, kernel), sigmaX=sigma, sigmaY=sigma)

    
    if st.checkbox('Sharpen'):
        ksize = st.slider('ksize', min_value=1, max_value=25, step=2, value=1, key='sharpen_ksize')
        sigma = st.slider('sigma', min_value=5, max_value=15, step=5, value=5, key='sharpen_sigma')
        b_image = cv2.GaussianBlur(image, ksize=(ksize, ksize), sigmaX=sigma, sigmaY=sigma)
        effect = st.radio('Effect', options=['Strong', 'Normal', 'Low'], key='sharpen_effect')
        if effect=='Strong':
            image = cv2.addWeighted(src1=image, alpha=2.5, src2=b_image, beta=-1.5, gamma=0)
        elif effect=='Normal':
            image = cv2.addWeighted(src1=image, alpha=2, src2=b_image, beta=-1, gamma=0)
        elif effect=='Low':
            image = cv2.addWeighted(src1=image, alpha=1.5, src2=b_image, beta=-0.5, gamma=0)
        
    if st.checkbox('Sobel'):
        dx = st.slider('dx', min_value=0, max_value=1, step=1, value=1, key='sobel_dx')
        dy = st.slider('dy', min_value=0, max_value=1, step=1, value=1, key='sobel_dy')
        ksize = st.slider('ksize', min_value=1, max_value=35, step=2, value=15, key='sobel_ksize')
        scale = st.slider('scale', min_value=1, max_value=15, step=2, value=7, key='sobel_scale')
        image = cv2.Sobel(image, ddepth=-1, dx=dx, dy=dy, ksize=ksize, scale=scale)

    if st.checkbox('Laplacian'):
        ksize = st.slider('ksize', min_value=1, max_value=15, step=2, value=1, key='lap_ksize')
        scale = st.slider('scale', min_value=0, max_value=15, step=3, value=5, key='lap_scale')
        image = cv2.Laplacian(image, ddepth=-1, ksize=ksize, scale=scale)


    if st.checkbox('Canny'):
        t1 = st.slider('Low Threshold', min_value=0, max_value=250, step=1, value=128, key='canny_t1')
        t2 = st.slider('Hihg Threshold', min_value=t1+1, max_value=255, step=1, value=255, key='canny_t2')
        image = cv2.Canny(image, threshold1=t1, threshold2=t2)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if st.checkbox('Erode'):
        kernel = st.slider('Kernel', min_value=1, max_value=35, step=2, value=15, key='erode_kernel')
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel, kernel))
        image = cv2.erode(image, kernel=kernel)
        

    if st.checkbox('Dilate'):
        kernel = st.slider('Kernel', min_value=1, max_value=35, step=2, value=15, key='dilate_kernel')
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel, kernel))
        image = cv2.dilate(image, kernel=kernel)

    if st.checkbox('Mosaic Effect'):
        level = st.slider('Level', min_value=1, max_value=35, step=2, value=15, key='mosaic_level')
        h = int(image.shape[0]/level)                                                                                   
        w = int(image.shape[1]/level)                                                                                   
        image = cv2.resize(image, (w,h), interpolation=cv2.INTER_LINEAR)                                               
        image = cv2.resize(image, (image.shape[1],image.shape[0]), interpolation=cv2.INTER_NEAREST)                  


    st.image(image, caption="After", use_column_width=True, channels="BGR")


    download_button = st.download_button(
        label='Download your image.',
        data=save_image_as_png(image=image),
        file_name='image.png',
        key='download'
    )