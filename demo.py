import streamlit as st
import cv2
import numpy as np
from skimage import io
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

def load_and_resize_image(uploaded_file, target_size):
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
    resized_image = cv2.resize(image_rgb, target_size)
    return resized_image

# Initialize session state variables
if 'step' not in st.session_state:
    st.session_state.step = 1

def calculate_psnr(original_image, degraded_image):
    # Ensure both images have the same size
    if original_image.shape != degraded_image.shape:
        st.error("Both images must have the same dimensions for PSNR calculation.")
        return None

    # Calculate MSE (Mean Squared Error)
    mse = np.mean((original_image - degraded_image) ** 2)

    # Calculate PSNR
    if mse == 0:
        psnr = float('inf')  # Perfectly identical images
    else:
        max_pixel_value = 255  # Maximum pixel value for 8-bit images
        psnr = 10 * np.log10((max_pixel_value ** 2) / mse)

    return psnr

# Set the common target size for resizing
target_size = (1024, 768)  # Adjust to your desired dimensions

# Sobel filter function to detect edges for Y channel (luminance)
def sobel_edges(ycbcr_image):
    sobelx = cv2.Sobel(ycbcr_image[:, :, 0], cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(ycbcr_image[:, :, 0], cv2.CV_64F, 0, 1, ksize=3)
    edges = np.sqrt(sobelx ** 2 + sobely ** 2)
    return edges / np.max(edges)  # Normalizing to [0, 1] range

# Calculate local difference
def local_difference(edge_image1, edge_image2):
    diff = np.abs(edge_image1 - edge_image2)
    return diff / np.max(diff)  # Normalizing to [0, 1] range

# Local distance calculation
def local_distance(diff_image, block_size=8):
    M, N = diff_image.shape
    padded_diff_image = np.pad(diff_image, ((0, block_size), (0, block_size)), 'constant')
    local_dist = np.zeros_like(diff_image)

    for i in range(0, M, block_size):
        for j in range(0, N, block_size):
            block = padded_diff_image[i:i+block_size, j:j+block_size]
            min_dim = min(block_size, block.shape[0], block.shape[1])
            weights = 1 / (np.arange(1, min_dim+1)**2)
            weight_matrix = np.outer(weights, weights)
            weight_matrix = weight_matrix[:block.shape[0], :block.shape[1]]

            weighted_block = block * weight_matrix
            local_dist[i:i+block.shape[0], j:j+block.shape[1]] = np.sum(weighted_block)
    return local_dist

# Combine color components (MDC)
def combine_color_components(Y_diff, Cb_diff, Cr_diff, delta_param=0.35):
    return Y_diff + delta_param * (Cb_diff + Cr_diff)

# Streamlit app title
st.title('Image Quality Assessment App - MDC demo')

# Upload two images: original and color-degraded
original_file = st.file_uploader("Upload the original image", type=["jpg", "jpeg", "png"], key="original")
degraded_file = st.file_uploader("Upload the color-degraded image", type=["jpg", "jpeg", "png"], key="degraded")


if original_file and degraded_file:
    original_image = load_and_resize_image(original_file, target_size)
    degraded_image = load_and_resize_image(degraded_file, target_size)

    # Step 1: Convert to YCbCr and show images
    if st.session_state.step == 1:
        col1, col2 = st.columns(2)
        with col1:
            st.image([original_image], caption=['Original Image'], use_column_width=True)
        with col2:
            st.image([degraded_image], caption=['Color-Degraded Image'], use_column_width=True)
        if st.button('Proceed to Edge Detection'):
            st.session_state.step = 2

    # Step 2: Apply Sobel filter for edge detection
    if st.session_state.step == 2:
        col1, col2 = st.columns(2)
        st.session_state.original_ycbcr = cv2.cvtColor(original_image, cv2.COLOR_RGB2YCrCb)
        st.session_state.degraded_ycbcr = cv2.cvtColor(degraded_image, cv2.COLOR_RGB2YCrCb)

        # Compute edges and store in session state
        st.session_state.original_edges = sobel_edges(st.session_state.original_ycbcr)
        st.session_state.degraded_edges = sobel_edges(st.session_state.degraded_ycbcr)
        with col1:
            st.image([st.session_state.original_edges], caption=['Edges in Original Image'], use_column_width=True, clamp=True)
        with col2:
            st.image([st.session_state.degraded_edges], caption=['Edges in Color-Degraded Image'], use_column_width=True, clamp=True)
        if st.button('Proceed to Local Difference'):
            st.session_state.step = 3

    # Step 3: Calculate and show local difference
    if st.session_state.step == 3:
        col1, col2 = st.columns(2)
        st.session_state.local_diff = local_difference(st.session_state.original_edges, st.session_state.degraded_edges)
        with col1:
            st.image(st.session_state.local_diff, caption='Local Difference', use_column_width=True, clamp=True)
        if st.button('Proceed to Local Distance'):
            st.session_state.step = 4

    # Step 4: Calculate and show local distance
    if st.session_state.step == 4:
        col1, col2 = st.columns(2)
        st.session_state.local_dist = local_distance(st.session_state.local_diff)
        with col1:
            st.image(st.session_state.local_dist, caption='Local Distance', use_column_width=True, clamp=True)
        if st.button('Proceed to MDC Calculation'):
            st.session_state.step = 5

    # Step 5: Calculate and show MDC (Color Difference Metric)
    if st.session_state.step == 5:
        # Calculate local difference (if not already calculated)
        if 'local_diff' not in st.session_state:
            local_diff = local_difference(st.session_state.original_edges, st.session_state.degraded_edges)
            st.session_state.local_diff = local_diff
        else:
            local_diff = st.session_state.local_diff

        # Calculate local distance (if not already calculated)
        if 'local_dist' not in st.session_state:
            local_dist = local_distance(local_diff)
            st.session_state.local_dist = local_dist
        else:
            local_dist = st.session_state.local_dist

        # Extract Cb and Cr channels from the YCbCr images
        original_cb = st.session_state.original_ycbcr[:, :, 1]
        original_cr = st.session_state.original_ycbcr[:, :, 2]
        degraded_cb = st.session_state.degraded_ycbcr[:, :, 1]
        degraded_cr = st.session_state.degraded_ycbcr[:, :, 2]
        # Calculate the differences for Cb and Cr channels
        cb_diff = np.abs(original_cb - degraded_cb) / 255  # Normalizing to [0, 1] range
        cr_diff = np.abs(original_cr - degraded_cr) / 255  # Normalizing to [0, 1] range

        # Calculate MDC for each pixel
        mdc_value = combine_color_components(local_dist, cb_diff, cr_diff)
        st.image(mdc_value, caption='Color Difference Metric (MDC)', use_column_width=True, clamp=True)
        fig, ax = plt.subplots()
        cax = ax.matshow(mdc_value, cmap='hot')
        fig.colorbar(cax)
        ax.set_title('Local Difference')
        st.pyplot(fig)

        # Calculate the mean MDC value
        mean_mdc = np.mean(mdc_value)


        # Display the mean MDC value
        st.write("Mean MDC (Color Difference Metric):", mean_mdc)

        # Calculate SSIM
        original_image_gray = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
        degraded_image_gray = cv2.cvtColor(degraded_image, cv2.COLOR_RGB2GRAY)
        ssim_score = ssim(original_image_gray, degraded_image_gray)
        st.write(f'SSIM Score: {ssim_score}')

        # Calculate PSNR
        psnr_score = calculate_psnr(original_image, degraded_image)

        # Display PSNR score
        if psnr_score is not None:
            st.write(f"PSNR Score: {psnr_score:.2f} dB")

        if st.button('Restart'):
            st.session_state.step = 1   

