import customtkinter as tk
from customtkinter import filedialog
from PIL import Image, ImageTk 
import cv2
import numpy as np
import io

original_image = None  # Variable to store the original image
reference_image = None 
zoom_level = 100

# Function to save the modified image
def save_image():
    global zoom_level

    if cv2_image is not None:
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Images", "*.png"), ("All Files", "*.*")])
        if file_path:
            # Get the current image from the image_canvas
            image_canvas = canvas_image.postscript(colormode="color")

            # Convert the PostScript to a PIL image
            pil_image = Image.open(io.BytesIO(image_canvas.encode("utf-8")))
            # Save the image in the desired format
            pil_image.save(file_path)
            print("Image saved successfully!")

# Function to load an image
def load_image():
    global cv2_image, original_image
    file_path = filedialog.askopenfilename()
    if file_path:
        cv2_image = cv2.imread(file_path)
        original_image = cv2_image.copy()  # Make a copy of the original image
        show_image(cv2_image)

# Function to load the reference image
def load_reference_image():
    global reference_image

    file_path = filedialog.askopenfilename()
    if file_path:
        reference_image = cv2.imread(file_path)
        #show_image(reference_image)

# Function to zoom in on the image
def zoom_image():
    global zoom_level
    zoom_input = tk.CTkInputDialog(text="Enter the zoom level (in %):", title="Zoom")
    new_zoom = zoom_input.get_input()
    if new_zoom is not None:
        try:
            new_zoom = int(new_zoom)
            if new_zoom > 0:
                zoom_level = new_zoom
                show_image(original_image)  # Reload the original image with the new zoom
            else:
                print("Invalid zoom value. Zoom should be a positive number.")
        except ValueError:
            print("Invalid zoom value. Zoom should be a positive integer.")

# Function to display the loaded image in the interface
def show_image(image):
    global zoom_level
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = image.shape
    new_width = int(width * zoom_level / 100)
    new_height = int(height * zoom_level / 100)
    resized_image = cv2.resize(image, (new_width, new_height))
    pil_image = Image.fromarray(resized_image)
    tk_image = ImageTk.PhotoImage(pil_image)
    # Calculate the x and y coordinates to center the image on the image_canvas
    x = (canvas_image.winfo_width() - new_width) // 2
    y = (canvas_image.winfo_height() - new_height) // 2

    # Clear the canvas 
    canvas_image.delete("all")
    # Set the canvas size based on the size of the resized image
    canvas_image.config(width=new_width, height=new_height)
    # Create the centered image
    canvas_image.create_image(x, y, anchor=tk.NW, image=tk_image)
    canvas_image.image = tk_image

# Function to apply a blur filter to the image
def apply_blur():
    if cv2_image is not None:
        global original_image
        original_image = cv2_image.copy()  # Save the original image before applying the filter
        blurred_image = cv2.GaussianBlur(cv2_image, (21, 21), 0)
        show_image(blurred_image)

# Function to apply a median filter, remove salt and pepper noise
def median_filter():
    if cv2_image is not None:
        global original_image
        original_image = cv2_image.copy()  # Save the original image before applying the filter
        blurred_image = cv2.medianBlur(cv2_image, 3)
        show_image(blurred_image)

# Function to apply an enhancement filter to the image
def apply_enhancement():
    if cv2_image is not None:
        global original_image
        original_image = cv2_image.copy()  # Save the original image before applying the filter
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)  # Laplacian
        enhanced_image = cv2.filter2D(cv2_image, -1, kernel)
        show_image(enhanced_image)

# Function to apply a Sobel enhancement filter
def apply_sobel_enhancement():
    if cv2_image is not None:
        global original_image
        original_image = cv2_image.copy()  # Save the original image before applying the filter
        # Sobel filters for calculating partial derivatives
        sobX = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        sobY = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

        Gx = cv2.filter2D(cv2_image, cv2.CV_64F, sobX)  # Gradient in the X direction (rows)
        Gy = cv2.filter2D(cv2_image, cv2.CV_64F, sobY)  # Gradient in the Y direction (columns)

        magnitude = np.sqrt(Gx**2 + Gy**2)  # Magnitude of the gradient vector

        sharpened_image = cv2_image + 0.4 * magnitude
        sharpened_image[sharpened_image > 255] = 255
        sharpened_image = sharpened_image.astype(np.uint8)

        magnitude[magnitude > 255] = 255
        magnitude = magnitude.astype(np.uint8)

        show_image(sharpened_image)

# Function to apply an image gamma filter
def gamma_filter():
    if cv2_image is not None:
        global original_image
        original_image = cv2_image.copy()  # Save the original image before applying the filter
        gamma = (tk.CTkInputDialog(text="Enter the gamma value", title="Gamma")).get_input()
        gamma = int(gamma)
        img = cv2_image
        c = 255.0 / (255.0**gamma)
        gamma_image = c * (img.astype(np.float64))**gamma
        gamma_image = gamma_image.astype(np.uint8)
        show_image(gamma_image)

# Function to apply image equalization
def apply_equalization():
    if cv2_image is not None:
        global original_image
        original_image = cv2_image.copy()  # Save the original image before applying the filter
        img = cv2_image.copy()
        R = img.shape[0]
        C = img.shape[1]

        # Calculate the normalized histogram (pr)
        hist = cv2.calcHist([img], [0], None, [256], [0, 256]) 
        pr = hist / (R * C)

        # Cumulative distribution function (
        cdf = pr.cumsum()
        sk = 255 * cdf
        sk = np.round(sk)

        # Create the output image
        img_out = np.zeros(img.shape, dtype=np.uint8)
        for i in range(256):
            img_out[img == i] = sk[i]
        show_image(img_out)

# Function to apply equalization based on a reference image
def apply_equalization_by_reference():
    global original_image, reference_image
    load_reference_image()
    if cv2_image is not None and reference_image is not None:
        original_image = cv2_image.copy()  # Save the original image before applying the filter
        input_img = cv2_image.copy()  # Input image       
        reference_img = reference_image.copy()

        chans_input = cv2.split(input_img)  # Split color channels
        chans_ref = cv2.split(reference_img)

        # Iterate over the channels of the input image and calculate the histogram
        pr = np.zeros((256, 3))
        for chan, n in zip(chans_input, np.arange(3)):
            pr[:, n] = cv2.calcHist([chan], [0], None, [256], [0, 256]).ravel()

        # Iterate over the channels of the reference image and calculate the histogram
        pz = np.zeros((256, 3))
        for chan, n in zip(chans_ref, np.arange(3)):
            pz[:, n] = cv2.calcHist([chan], [0], None, [256], [0, 256]).ravel()
        
        # Calculate the CDFs for the input image
        cdf_input = np.zeros((256, 3))
        for i in range(3):
            cdf_input[:, i] = np.cumsum(pr[:, i])  # Reference
        
        # Calculate the CDFs for the reference image
        cdf_ref = np.zeros((256, 3))
        for i in range(3):
            cdf_ref[:, i] = np.cumsum(pz[:, i])  # Reference
    
    img_out = np.zeros(input_img.shape)  # Output image

    for c in range(3):  # Iterate over color planes
        for i in range(256):  # Iterate over the CDF of each plane of the image
            diff = np.absolute(cdf_ref[:, c] - cdf_input[i, c])
            index = diff.argmin()
            img_out[input_img[:, :, c] == i, c] = index

    img_out = img_out.astype(np.uint8)
    
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    reference_img = cv2.cvtColor(reference_img, cv2.COLOR_BGR2RGB)
    img_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)
    show_image(img_out)

# Function to restore the original image
def restore_original():
    global cv2_image, original_image
    if original_image is not None:
        cv2_image = original_image.copy()  # Restore the original image
        show_image(cv2_image)

brightness_factor = 1.0  # Initialize the brightness factor

# Function to adjust and display brightness
def adjust_and_show_brightness():
    global brightness_factor, cv2_image, original_image

    brightness_input = tk.CTkInputDialog(text="Enter the brightness multiplication factor (0~2):", title="Brightness")
    new_brightness_factor = brightness_input.get_input()

    if new_brightness_factor is not None:
        try:
            new_brightness_factor = float(new_brightness_factor)
            if 0 <= new_brightness_factor <= 2:
                brightness_factor = new_brightness_factor
                new_image = adjust_brightness(original_image, brightness_factor)
                show_image(new_image)
            else:
                print("Brightness factor out of valid range (0~2).")
        except ValueError:
            print("Invalid brightness factor value. Use a decimal number between 0 and 2.")

# Function to adjust the brightness of the image
def adjust_brightness(image, factor):
    if image.ndim == 2:
        # Grayscale image
        new_image = image.astype(np.float64)
        new_image = new_image * factor
        new_image[new_image > 255] = 255
        new_image[new_image < 0] = 0
        new_image = new_image.astype(np.uint8)
    elif image.ndim == 3:
        # Color image (3 channels: Red, Green, Blue)
        new_image = image.astype(np.float64)
        new_image = new_image * factor
        new_image[new_image > 255] = 255
        new_image[new_image < 0] = 0
        new_image = new_image.astype(np.uint8)
    else:
        raise ValueError("Image with unsupported number of dimensions.")

    return new_image

# Function to create a negative of the image
def create_negative(image):
    if image.ndim == 2:
        # Grayscale image
        negative_image = 255 - image
    elif image.ndim == 3:
        # Color image (3 channels: Red, Green, Blue)
        negative_image = 255 - image
    else:
        raise ValueError("Image with unsupported number of dimensions.")

    return negative_image

# Function to apply negative effect and display
def apply_negative_and_show():
    global cv2_image, original_image

    if cv2_image is not None:
        # Save the original image before applying the effect
        original_image = cv2_image.copy()

        # Apply the negative effect
        negative_image = create_negative(original_image)

        # Display the image with the effect in the interface
        show_image(negative_image)


# Function for Canny edge detection
def canny_edge():
    if cv2_image is not None:
        global original_image
        threshold1_input = tk.CTkInputDialog(text="Enter the first threshold value.", title="Threshold 1")
        threshold1 = int(threshold1_input.get_input())
        threshold2_input = tk.CTkInputDialog(text="Enter the second threshold value.", title="Threshold 2")
        threshold2 = int(threshold2_input.get_input())
        original_image = cv2_image.copy()  # Save the original image before applying the filter
        img = cv2_image
        
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Extract edges
        img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)
        canny_img = cv2.Canny(img_gray, threshold1, threshold2)

        show_image(canny_img)

# Function for Hough Transform
def hough_transform():
    if cv2_image is not None:
        global original_image
        original_image = cv2_image.copy()  # Save the original image before applying the filter
        img = cv2_image
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Create a simple dialog for the user to input Hough Transform thresholds and parameters
        threshold1_input = tk.CTkInputDialog(text="Enter the first Hough Transform threshold:", title="Hough Transform Threshold 1")
        #threshold1_input.wait_window()
        threshold1 = threshold1_input.get_input()

        threshold2_input = tk.CTkInputDialog(text="Enter the second Hough Transform threshold:", title="Hough Transform Threshold 2")
        #threshold2_input.wait_window()
        threshold2 = threshold2_input.get_input()

        rho_input = tk.CTkInputDialog(text="Enter the rho value:", title="Hough Transform Rho")
        #rho_input.wait_window()
        rho = rho_input.get_input()

        threshold_input = tk.CTkInputDialog(text="Enter the accumulator threshold value:", title="Hough Transform Accumulator Threshold")
        #threshold_input.wait_window()
        threshold = threshold_input.get_input()

        min_line_length_input = tk.CTkInputDialog(text="Enter the minimum line length:", title="Hough Transform Min Line Length")
        #min_line_length_input.wait_window()
        min_line_length = min_line_length_input.get_input()

        max_line_gap_input = tk.CTkInputDialog(text="Enter the maximum gap between lines:", title="Hough Transform Max Line Gap")
        #max_line_gap_input.wait_window()
        max_line_gap = max_line_gap_input.get_input()

        if all(param is not None for param in [threshold1, threshold2, rho, threshold, min_line_length, max_line_gap]):
            try:
                threshold1 = int(threshold1)
                threshold2 = int(threshold2)
                rho = float(rho)
                threshold = int(threshold)
                min_line_length = int(min_line_length)
                max_line_gap = int(max_line_gap)

                # Edge Detection
                blur = cv2.GaussianBlur(img_gray, (11, 11), 0)
                edges = cv2.Canny(blur, threshold1, threshold2, None, 3)

                cdstP = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

                linesP = cv2.HoughLinesP(edges, rho, np.pi / 180, threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)

                # Draw lines
                if linesP is not None:
                    for i in range(0, len(linesP)):
                        l = linesP[i][0]
                        cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv2.LINE_AA)

                show_image(cdstP)
            except ValueError:
                print("Invalid input values. Please enter valid numerical values.")



# Default arguments for buttons
btn_default_args = {'bg_color': '#190061',
                    'fg_color': '#3500D3'}

def create_buttons(master, default_args, text, command=None):
    button = tk.CTkButton(master, **default_args, text=text, command=command)
    return button

def create_buttons_menu():
    # Frame for buttons
    buttons_frame = tk.CTkFrame(master=window, width=160, height=320, fg_color="#190061")
    buttons_frame.pack(side="left", padx=10, pady=10)

    global btn_default_args

    # Button to load the image
    btn_load = create_buttons(buttons_frame, btn_default_args, "Load Image", command=load_image)
    btn_load.pack(pady=10)

    # Button to save the image
    btn_save = create_buttons(buttons_frame, btn_default_args, "Save Image", command=save_image)
    btn_save.pack(pady=10)

    # Button to restore the original image
    btn_restore = create_buttons(buttons_frame, btn_default_args, "Restore Original", command=restore_original)
    btn_restore.pack(pady=10)

    # Button to zoom in the image
    btn_zoom = create_buttons(buttons_frame, btn_default_args, "Zoom", command=zoom_image)
    btn_zoom.pack(padx=10, pady=10)

    # Button to select filters
    btn_restore = create_buttons(buttons_frame, btn_default_args, "Select Effects", command=effects_window)
    btn_restore.pack(pady=10)

def effects_window():
    new_window = tk.CTkToplevel(window)
    new_window.title("Effects")
    global btn_default_args
    # Button to apply blur
    btn_blur = create_buttons(new_window, btn_default_args, "Apply Blur (Gaussian)", command=apply_blur)
    btn_blur.pack(pady=10)

    # Button to apply enhancement
    btn_enhance = create_buttons(new_window, btn_default_args, "Apply Enhancement (Laplacian)", command=apply_enhancement)
    btn_enhance.pack(pady=10)

    # Button to apply enhancement (Sobel)
    btn_enhance = create_buttons(new_window, btn_default_args, "Apply Enhancement (Sobel)", command=apply_sobel_enhancement)
    btn_enhance.pack(pady=10)

    # Button to increase or decrease brightness
    btn_zoom = create_buttons(new_window, btn_default_args, "Brightness", command=adjust_and_show_brightness)
    btn_zoom.pack(side="top", padx=10, pady=10)

    # Button for negative image
    btn_zoom = create_buttons(new_window, btn_default_args, "Negative", command=apply_negative_and_show)
    btn_zoom.pack(side="top", padx=10, pady=10)

    # Button to apply median filter
    btn_zoom = create_buttons(new_window, btn_default_args, "Median", command=median_filter)
    btn_zoom.pack(side="top", padx=10, pady=10)

    # Button to apply equalization
    btn_zoom = create_buttons(new_window, btn_default_args, "Equalization", command=apply_equalization)
    btn_zoom.pack(side="top", padx=10, pady=10)

    # Button to apply specification
    btn_zoom = create_buttons(new_window, btn_default_args, "Specification", command=apply_equalization_by_reference)
    btn_zoom.pack(side="top", padx=10, pady=10)

    # Button to apply gamma filter
    btn_zoom = create_buttons(new_window, btn_default_args, "Gamma", command=gamma_filter)
    btn_zoom.pack(side="top", padx=10, pady=10)

    # Button for Canny edge detection
    btn_zoom = create_buttons(new_window, btn_default_args, "Canny", command=canny_edge)
    btn_zoom.pack(side="top", padx=10, pady=10)

    # Button for Hough Transform
    btn_zoom = create_buttons(new_window, btn_default_args, "Hough Transform", command=hough_transform)
    btn_zoom.pack(side="top", padx=10, pady=10)

if __name__ == "__main__":
    # Main window configuration
    tk.set_default_color_theme("dark-blue")
    window = tk.CTk()
    window.title("PicTrix - Image Editor")
    window.maxsize(width=1600, height=800)
    window.minsize(width=600, height=400)
    
    window.resizable(width=False, height=False)
    create_buttons_menu()
    
    # Display the loaded image on the Canvas
    canvas_image = tk.CTkCanvas(master=window, bg="#240090")
    canvas_image.pack(side="left", fill="both", expand=True, padx=10, pady=10)

    cv2_image = None

    # Function to resize the image when the window is resized
    def resize_canvas(event):
        if original_image is not None:
            show_image(original_image)

    # Bind the resize_canvas function to the window resize event
    window.bind("<Configure>", resize_canvas)

    window.mainloop()
