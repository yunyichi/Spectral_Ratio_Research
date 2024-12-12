import cv2
import numpy as np

# Constants
GAMMA_VALUE = 2.2  # Fixed gamma for linearization
gamma = 2.2  # Initial gamma correction value from slider

def linearize_image(image, gamma_value):
    """Undo the gamma correction to get a linearized image."""
    image_normalized = image / 255.0
    image_normalized = np.clip(image_normalized, 1e-6, 1.0) # avoid zero values
    linear_image = np.power(image_normalized, gamma_value)  # Undo gamma correction
    linear_image = np.clip(linear_image, 1e-6, 1.0)
    print("checking point 2308, 1668", image[700, 700] )
    print("checking point 2308, 1668", linear_image[700, 700] )
    return linear_image

def apply_gamma_correction(image, gamma_value):
    """Apply gamma correction to the linearized image."""
    corrected_image = np.power(image, 1 / gamma_value)  # Apply gamma correction
    return (corrected_image * 255).astype(np.uint8)

def calculate_spectral_ratio(dark_point, lit_point):
    """Calculate the normalized spectral ratio (NSR) for each channel."""
    SR_R = dark_point[2] / (lit_point[2] - dark_point[2]) if lit_point[2] != dark_point[2] else 1.0
    SR_G = dark_point[1] / (lit_point[1] - dark_point[1]) if lit_point[1] != dark_point[1] else 1.0
    SR_B = dark_point[0] / (lit_point[0] - dark_point[0]) if lit_point[0] != dark_point[0] else 1.0

    print("SR: ", SR_R, SR_G, SR_B)

    L = np.sqrt(SR_R**2 + SR_G**2 + SR_B**2)  # Length to normalize the spectral ratios

    NSR_R = SR_R / L
    NSR_G = SR_G / L
    NSR_B = SR_B / L

    return (NSR_B, NSR_G, NSR_R, SR_B, SR_G, SR_R, L)  # Return in BGR order

def spectral_ratio_adjustment(linear_image, corrected_image, dark_point, lit_point):
    """Apply SR-guided gamma adjustment based on spectral ratio."""
    # calculate spectral ratio and normailzed spectral ratio
    NSR_B, NSR_G, NSR_R, SR_B, SR_G, SR_R, L = calculate_spectral_ratio(dark_point, lit_point)
    norm_spectral_ratio = np.array([NSR_B, NSR_G, NSR_R])
    spectral_ratio = np.array([SR_B, SR_G, SR_R])

    # normalize input images to [0, 1] range
    I_prime = corrected_image.astype(np.float32) / 255.0
    I_prime = np.clip(I_prime, 1/255.0, 255)
    I_prime = np.clip(I_prime, 1e-6, 1.0)

    linear_image = np.clip(linear_image, 1/255.0, 1.0)
    linear_image = np.clip(linear_image, 1e-6, 1.0)
    lin_vis = (linear_image*255).astype(np.uint8)
    cv2.imshow("linvis", lin_vis)

    # convert to grayscale channel
    channel_prime = (I_prime[:,:,0] + I_prime[:,:,1] + I_prime[:,:,2])/3.0
    channel = (linear_image[:,:,0] + linear_image[:,:,1] + linear_image[:,:,2])/3.0

    # clip channels to remove outliers
    clipped_channel = np.clip(channel*1.5, 1/255.0, 1.0)
    clipped_channel_prime = np.clip(channel_prime, 1/255.0, np.max(channel) )

    # calculate the guiding factor 'g'
    print("L", L)
    g = (1 + L) / (clipped_channel_prime/clipped_channel) - L
    g = np.clip(g, 0, 1)
    
    g_vis = (g*255).astype(np.uint8)
    cv2.imshow("g", g_vis)

    # update the image with spectral ratio adjustment
    print("shape:", linear_image.shape)
    V = np.copy(linear_image)
    
    V[:,:,0] = (V[:,:,0] / (1 + g[:,:]/spectral_ratio[0]))
    V[:,:,1] = (V[:,:,1] / (1 + g[:,:]/spectral_ratio[1]))
    V[:,:,2] = (V[:,:,2] / (1 + g[:,:]/spectral_ratio[2]))

    print("spectral ratio", spectral_ratio)
    print("g thing", g[700, 700] )
    print("lin thing", linear_image[700, 700] )
    print("V thing", V[700, 700])

    Vvis = (255.0 * V / np.max(V)).astype(np.uint8)
    cv2.imshow("V-dark", Vvis)
    cv2.imshow("lin-sanity", linear_image)

    # undo spectral ratio normalization and calculate final adjusted image
    V[:,:,0] /= spectral_ratio[0]
    V[:,:,1] /= spectral_ratio[1]
    V[:,:,2] /= spectral_ratio[2]
    
    Vvis = (255.0 * V / np.max(V)).astype(np.uint8)
    cv2.imshow("V-lit", Vvis)
    
    # compute the 'effect factor' K
    VLen = (V[:,:,0] + V[:,:,1] + V[:,:,2])/3
    K = (channel_prime - channel) / VLen

    # pro-rate the effect from 50 to 150
    # anything less than 0.2 assume is shadow R(A)
    # anything more than 0.7 assume is fully lit R(A + D)
    # gamma is (I - .2) / 0.5
    # If pixel is RA -> S = A/D so RA/S = RD
    # if pixel is RA + RD, new pixel should be RA + R(D + dI) - RA + RD = R(dI)
    # if pixel is RA + gRD = RA + gR(A/S) = R(A + gA/S) = R(A(1 + g/S)
    # image pixel = R(A(1 + g/S)), we know g, S, so RA = pixel / (1 + g/S) and RD = RA/S
    # g = (1 + ||S||) / (||P'||/||P||) - ||S||
    
    # apply K to adjust the image
    sr_guided_gamma = np.copy(linear_image)
    sr_guided_gamma[:,:,0] += np.multiply(K, V[:,:,0])
    sr_guided_gamma[:,:,1] += np.multiply(K, V[:,:,1])
    sr_guided_gamma[:,:,2] += np.multiply(K, V[:,:,2])

    # Clip to valid pixel range [0, 255]
    sr_guided_gamma = np.clip(sr_guided_gamma * 255, 0, 255).astype(np.uint8)

    return sr_guided_gamma


def add_label(image, text):
    """Add label text to the top of the image."""
    label_height = 80
    labeled_image = np.full((image.shape[0] + label_height, image.shape[1], 3), 255, dtype=np.uint8)
    labeled_image[label_height:, :] = image
    cv2.putText(labeled_image, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    return labeled_image

def on_gamma_correction_trackbar(val):
    global gamma
    gamma = val / 100.0  # Convert slider value to float

    # Step 2: Apply gamma correction to the fixed linearized image
    corrected_image = apply_gamma_correction(linear_image, gamma)

    # Step 3: Apply SR-guided gamma correction based on linearized image
    final_image = spectral_ratio_adjustment(linear_image, corrected_image, dark_point, lit_point)

    # Add labels to each image
    labeled_linear = add_label((linear_image * 255).astype(np.uint8), "Linear Image")
    labeled_corrected = add_label(corrected_image, "Gamma-Corrected Image")
    labeled_final = add_label(final_image, "SR-Guided Gamma Image")

    # Concatenate all labeled images horizontally
    img_combined = cv2.hconcat([labeled_linear, labeled_corrected, labeled_final])

    # Display the concatenated image
    cv2.imshow("Gamma Correction", img_combined)

# Main processing
image_path = 'dokania_srijan_021.tif'  # Replace with your image path
image = cv2.imread(image_path)

if image is None:
    print(f"Error: Unable to load image at {image_path}")
else:
    # Predefined points
    lit_point = np.array([69.0, 109.0, 155.0])  # Example lit point (BGR format)
    dark_point = np.array([11.0, 10.0, 11.0])  # Example dark point (BGR format)

    # linearize the lit and dark points
    lit_point = np.clip(lit_point / 255.0, 1e-6, 1.0)    # Avoid zero values
    dark_point = np.clip(dark_point / 255.0, 1e-6, 1.0)  # Avoid zero values

    dark_point = np.power(dark_point, 2.2)
    lit_point = np.power(lit_point, 2.2)
    dark_point = np.clip(dark_point, 1e-6, None)
    lit_point = np.clip(lit_point, 1e-6, None)

    # Step 1: Linearize the original image once
    linear_image = linearize_image(image, GAMMA_VALUE)

    # Create window and slider
    cv2.namedWindow('Gamma Correction')
    cv2.createTrackbar('Gamma', 'Gamma Correction', int(gamma * 100), 300, on_gamma_correction_trackbar)

    # Initial display
    on_gamma_correction_trackbar(int(gamma * 100))

    # Wait for Esc key to exit
    while True:
        if cv2.waitKey(1) == 27:  # Esc key to exit
            break

cv2.destroyAllWindows()
