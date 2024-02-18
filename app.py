import streamlit as st
from skimage.metrics import structural_similarity
import imutils
import cv2
import numpy as np

def image_comparison(img1, img2):
    # Convert images to grayscale
    gray_orig = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray_mod = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Resize images to a common size
    common_size = (min(gray_orig.shape[1], gray_mod.shape[1]), min(gray_orig.shape[0], gray_mod.shape[0]))
    gray_orig = cv2.resize(gray_orig, common_size)
    gray_mod = cv2.resize(gray_mod, common_size)

    img1 = cv2.resize(img1, common_size)
    img2 = cv2.resize(img2, common_size)

    # Compute structural similarity
    (score, diff) = structural_similarity(gray_orig, gray_mod, full=True)
    diff = (diff * 255).astype("uint8")

    thresh = cv2.threshold(diff, 60, 255, cv2.THRESH_BINARY_INV)[1]
    
    # Find contours
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    # Set a minimum contour area to filter out small differences
    min_contour_area = 500
    
    # Draw bounding boxes on first and second images, filtering small contours
    for c in cnts:
        contour_area = cv2.contourArea(c)
        if contour_area > min_contour_area:
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Save images and show the final frame
    if score < 1:
        cv2.imwrite('Comparison1.png', img1)
        cv2.imwrite('Comparison2.png', img2)
        cv2.imwrite('diff.png', diff)

    final_frame = np.hstack((img1, img2))

    return final_frame, score

def main():
    st.title("Image Comparison App")

    uploaded_file1 = st.file_uploader("Upload Image 1", type=["jpg", "png", "jpeg"])
    uploaded_file2 = st.file_uploader("Upload Image 2", type=["jpg", "png", "jpeg"])

    if uploaded_file1 is not None and uploaded_file2 is not None:
        image1 = cv2.imdecode(np.fromstring(uploaded_file1.read(), np.uint8), 1)
        image2 = cv2.imdecode(np.fromstring(uploaded_file2.read(), np.uint8), 1)

        final_image, similarity_score = image_comparison(image1, image2)

        st.image(final_image, channels="BGR")

        if similarity_score < 1:
            st.write(f"Structural Similarity Index: {similarity_score:.2f}")
        else:
            st.write("No differences found.")

if __name__ == "__main__":
    main()
