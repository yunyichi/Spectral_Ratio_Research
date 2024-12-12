import cv2

# Global variables to hold the points
dark_point = None
lit_point = None

def click_event(event, x, y, flags, param):
    global dark_point, lit_point

    if event == cv2.EVENT_LBUTTONDOWN:
        # If lit point hasn't been selected yet
        if lit_point is None:
            lit_point = image[y, x]
            print("Lit point selected at:", (x, y), "Color (BGR):", lit_point)
            cv2.putText(image, 'Lit Point', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imshow("Image", image)  # Show the updated image
            print("Now click on a dark point.")
        # If dark point hasn't been selected yet
        elif dark_point is None:
            dark_point = image[y, x]
            print("Dark point selected at:", (x, y), "Color (BGR):", dark_point)
            print("Selection complete.")

            # Optionally, you can display both points on the image
            cv2.putText(image, 'Dark Point', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.imshow("Image", image)  # Show the updated image

def main(image_path):
    global image

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return

    # Create window for the image
    cv2.imshow("Image", image)
    cv2.setMouseCallback("Image", click_event)

    print("Click on a lit point in the image.")
    
    # Wait for user to select both points
    while dark_point is None or lit_point is None:
        if cv2.waitKey(1) & 0xFF == 27:  # Esc key to exit
            break

    # Optionally print out the selected points after both are chosen
    if dark_point is not None and lit_point is not None:
        print("Lit Point Color (BGR):", lit_point)
        print("Dark Point Color (BGR):", dark_point)

    # Clean up
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = 'dokania_srijan_021.tif'  # Replace with your image path
    main(image_path)
