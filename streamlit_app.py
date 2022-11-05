import cv2
import numpy as np
from PIL import Image
import streamlit as st
st.title("backbone Robot skin analysis")
def calculateBack(bytes_data): 
    #Open a simple image
    #img=cv2.imread("6_A_hgr2B_id05_1.jpg")
    original_image = Image.open(bytes_data)
    img = np.array(original_image)
    img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    #converting from gbr to hsv color space
    img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #skin color range for hsv color space 
    HSV_mask = cv2.inRange(img_HSV, (0, 15, 0), (17,170,255)) 
    HSV_mask = cv2.morphologyEx(HSV_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

    #converting from gbr to YCbCr color space
    img_YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    #skin color range for hsv color space 
    YCrCb_mask = cv2.inRange(img_YCrCb, (0, 135, 85), (255,180,135)) 
    YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

    #merge skin detection (YCbCr and hsv)
    global_mask=cv2.bitwise_and(YCrCb_mask,HSV_mask)
    global_mask=cv2.medianBlur(global_mask,3)
    global_mask = cv2.morphologyEx(global_mask, cv2.MORPH_OPEN, np.ones((4,4), np.uint8))


    HSV_result = cv2.bitwise_not(HSV_mask)
    YCrCb_result = cv2.bitwise_not(YCrCb_mask)
    global_result=cv2.bitwise_not(global_mask)

    #show results
    # cv2.imshow("1_HSV.jpg",HSV_result)
    # cv2.imshow("2_YCbCr.jpg",YCrCb_result)
    # cv2.imshow("3_global_result.jpg",global_result)
    # cv2.imshow("Image.jpg",img)
    #cv2.imwrite("1_HSV.jpg",HSV_result)
    #cv2.imwrite("2_YCbCr.jpg",YCrCb_result)
    #cv2.imwrite("3_global_result.jpg",global_result)

    st.image([img, global_result]  )
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()  

if __name__ == '__main__':
    img_file_buffer = st.camera_input("Take a picture")
    if img_file_buffer is not None:
        # To read image file buffer with OpenCV:
        bytes_data = img_file_buffer.getvalue()
        calculateBack(img_file_buffer)
        # Check the type of cv2_img:
        # Should output: <class 'numpy.ndarray'>
        #st.write(type(cv2_img))

        # Check the shape of cv2_img:
        # Should output shape: (height, width, channels)
        #st.write(cv2_img.shape)

    uploaded_file = st.file_uploader("Upload Your Image", type=['jpg', 'png', 'jpeg'])
    if uploaded_file is not None:
        # To read file as bytes: 
        calculateBack(uploaded_file)
