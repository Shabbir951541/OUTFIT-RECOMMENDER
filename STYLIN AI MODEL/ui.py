import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import requests
import io
import os

def upload_files():
    # Open the file dialog to select images
    file_paths = filedialog.askopenfilenames(title='Choose images', filetypes=[('Image files', '*.jpg *.jpeg *.png')])
    if not file_paths:
        return

    # Clear the previous results and images
    for widget in scrollable_frame.winfo_children():
        widget.destroy()

    # Display and upload each image
    for file_path in file_paths:
        image = Image.open(file_path)
        image.thumbnail((150, 150))
        img_display = ImageTk.PhotoImage(image)

        # Image label
        img_label = tk.Label(scrollable_frame, image=img_display)
        img_label.image = img_display
        img_label.pack()

        # Upload and predict
        try:
            response = make_prediction_request(file_path)
            if response.ok:
                predictions = response.json()
                print("Predictions:", predictions)  # Debugging line

                # Iterate through predictions and check each one
                for key, prediction in predictions.items():
                    if prediction in ['Topwear', 'Bottomwear', 'Shoes', 'Watches']:
                        directory = f'd:/test/{prediction}'
                        os.makedirs(directory, exist_ok=True)

                        # Save the image in the respective directory
                        image.save(os.path.join(directory, os.path.basename(file_path)))
                        break  # Exit the loop after the first match

                result_text = f"Predictions:\n{predictions}"
            else:
                result_text = "Failed to get prediction from server."

        except requests.ConnectionError:
            messagebox.showerror("Connection Error", "Could not connect to server.")
            return

        # Result label
        result_label = tk.Label(scrollable_frame, text=result_text)
        result_label.pack()

def make_prediction_request(file_path):
    # URL to the Flask app's prediction endpoint
    url = 'http://127.0.0.1:5000/predict'
    
    # Open file and send it in a POST request to the Flask server
    with open(file_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(url, files=files)
    return response

# Set up the main application window
root = tk.Tk()
root.title('Image Classifier GUI')

# Scrollable frame to hold the widgets
canvas = tk.Canvas(root)
scrollbar = tk.Scrollbar(root, orient='vertical', command=canvas.yview)
scrollable_frame = tk.Frame(canvas)
scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(
        scrollregion=canvas.bbox("all")
    )
)
canvas.create_window((0, 0), window=scrollable_frame, anchor='nw')
canvas.configure(yscrollcommand=scrollbar.set)

# Button to upload and predict images
upload_button = tk.Button(root, text='Upload and Predict Images', command=upload_files)
upload_button.pack()

import random
# ... [Your existing imports] ...
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Load the color classifier model
#color_classifier_path = 'D:/STYLIN AI MODEL/color_classifier_model.h5'
#color_classifier = load_model(color_classifier_path)

def classify_color(image_path):
    """ Classify the color of the clothing item using the Flask server. """
    response = make_prediction_request(image_path)
    if response.ok:
        predictions = response.json()
        print("Color Predictions:", predictions)  # Debugging line

        # Extract the color prediction
        # Assuming the server returns a dictionary where 'color' key holds the color prediction
        if 'color' in predictions:
            color_prediction = predictions['color']
            r, g, b = color_prediction_to_rgb(color_prediction)
            return r, g, b
        else:
            print("Color key not found in predictions")
            return None
    else:
        print("Failed to get color prediction from server.")
        return None


def color_prediction_to_rgb(color_prediction):
    # Convert the color prediction to RGB format
    # Placeholder logic - replace with your actual conversion logic
    r, g, b = 255, 255, 255  # Example: white color
    return r, g, b


import colorsys

def rgb_to_hsl(rgb):
    """ Convert RGB color to HSL color format. """
    r, g, b = rgb
    r /= 255.0
    g /= 255.0
    b /= 255.0
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return h * 360, l * 100, s * 100  # Convert fractions to degrees and percentages

def color_complements(color1, color2):
    """ Determine if two colors complement each other. """
    h1, _, _ = rgb_to_hsl(color1)
    h2, _, _ = rgb_to_hsl(color2)

    # Check if the hues are approximately opposite each other on the color wheel
    return abs(h1 - h2) > 150 and abs(h1 - h2) < 210

# Example usage
#print(color_complements((255, 0, 0), (0, 0, 255)))  # Red and Blue


def get_random_file_from_directory(directory, excluded_color=None):
    """ Returns a random file from the specified directory, considering color. """
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    if excluded_color is not None and np.any(excluded_color):
        files = [f for f in files if not color_complements(classify_color(os.path.join(directory, f)), excluded_color)]

    return random.choice(files) if files else None


def recommend_outfit():
    categories = ['Topwear', 'Bottomwear', 'Shoes', 'Watches']
    base_directory = 'd:/test/'

    outfit = {}
    last_color = None
    for category in categories:
        directory = os.path.join(base_directory, category)
        file = get_random_file_from_directory(directory, excluded_color=last_color)
        if file:
            image_path = os.path.join(directory, file)
            outfit_color = classify_color(image_path)
            print(f"Category: {category}, Outfit Color: {outfit_color}, Last Color: {last_color}")  # Debugging print
            if last_color is None or color_complements(outfit_color, last_color):
                outfit[category] = file
                last_color = outfit_color
            else:
                outfit[category] = 'No items available'
        else:
            print(f"No available items for {category}")  # Debugging print

    # Debugging: Print the final outfit recommendation
    for cat, item in outfit.items():
        print(f"{cat}: {item}")


    # Display the recommended outfit
    display_outfit_recommendation(outfit)


def classify_season(image_path):
    """ Classify the season of the clothing item using the Flask server. """
    response = make_prediction_request(image_path)
    if response.ok:
        predictions = response.json()
        print("Season Predictions:", predictions)  # Debugging line

        # Extract the season prediction
        # Assuming the server returns a dictionary where 'season' key holds the season prediction
        if 'prediction5' in predictions:
            season_prediction = predictions['prediction5']
            return season_prediction
        else:
            print("Season key not found in predictions")
            return None
    else:
        print("Failed to get season prediction from server.")
        return None
import requests
import datetime

def get_current_season():
    # OpenWeatherMap endpoint
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={42.304642}&lon={-83.067868}&appid={'d2edbfe440adb548ffff7943d2bb1a56'}&units=metric"

    # Make the request
    response = requests.get(url)
    if response.ok:
        weather_data = response.json()
        temp = weather_data['main']['temp']

        # Define temperature thresholds for different seasons (this is a rough approximation)
        if temp > 25:
            return 'Summer'
        elif 15 <= temp <= 25:
            return 'Fall'
        else:
            return 'Winter'
    else:
        print("Failed to retrieve weather data")
        return None

import os

import os
import random

def get_random_file_from_directory_weather(directory):
    """ Returns a list of all files from the specified directory. """
    # List all files in the directory
    return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

def recommend_weather_outfit():
    categories = ['Topwear', 'Bottomwear', 'Shoes', 'Watches']
    base_directory = 'd:/test/'

    #current_season = get_current_season() # Assuming this is set or retrieved from somewhere
    current_season=get_current_season()
    print(f"Current Season: {current_season}")

    outfit = {}
    for category in categories:
        directory = os.path.join(base_directory, category)
        files = get_random_file_from_directory_weather(directory)
        random.shuffle(files)  # Shuffle the files for random selection

        for file in files:
            image_path = os.path.join(directory, file)
            outfit_season = classify_season(image_path)
            print(f"Category: {category}, File: {file}, Outfit Season: {outfit_season}")

            if outfit_season == current_season:
                outfit[category] = file
                break
        else:  # This 'else' is associated with the for-loop (executes if no break occurs)
            print(f"No available items for {category} matching the season {current_season}")
            outfit[category] = 'No items suitable for current season'

    for cat, item in outfit.items():
        print(f"{cat}: {item}")

    display_weather_outfit_recommendation(outfit)






# Create frames for each category
topwear_frame = tk.Frame(root)
topwear_frame.pack()
topwear_label = tk.Label(topwear_frame, text="Topwear")
topwear_label.pack()

bottomwear_frame = tk.Frame(root)
bottomwear_frame.pack()
bottomwear_label = tk.Label(bottomwear_frame, text="Bottomwear")
bottomwear_label.pack()

shoes_frame = tk.Frame(root)
shoes_frame.pack()
shoes_label = tk.Label(shoes_frame, text="Shoes")
shoes_label.pack()

watches_frame = tk.Frame(root)
watches_frame.pack()
watches_label = tk.Label(watches_frame, text="Watches")
watches_label.pack()

# ... [Rest of your setup code] ...


# Global lists to keep track of dynamic labels
topwear_images = []
bottomwear_images = []
shoes_images = []
watches_images=[]

def clear_previous_recommendations():
    """ Clear previous recommendation images. """
    for widgets in [topwear_images, bottomwear_images, shoes_images, watches_images]:
        for widget in widgets:
            widget.destroy()
        widgets.clear()


def display_outfit_recommendation(outfit):
    # Clear previous recommendations
    clear_previous_recommendations()

    # Display new recommendations
    for category, file in outfit.items():
        image_path = os.path.join('d:/test/', category, file)
        if file != 'No items available' and os.path.isfile(image_path):
            image = Image.open(image_path)
            image.thumbnail((150, 150))
            img_display = ImageTk.PhotoImage(image)

            img_label = tk.Label(image=img_display)
            img_label.image = img_display  # Keep a reference

            if category == 'Topwear':
                img_label.pack(in_=topwear_frame)
                topwear_images.append(img_label)
            elif category == 'Bottomwear':
                img_label.pack(in_=bottomwear_frame)
                bottomwear_images.append(img_label)
            elif category == 'Shoes':
                img_label.pack(in_=shoes_frame)
                shoes_images.append(img_label)
            elif category == 'Watches':
                img_label.pack(in_=shoes_frame)
                shoes_images.append(img_label)

# ... [Rest of your imports and setup code] ...

# Assuming you have your category frames set up (topwear_frame, bottomwear_frame, etc.)

def display_weather_outfit_recommendation(outfit):
    # Clear previous recommendations
    clear_previous_recommendations()

    # Display new recommendations
    for category, file in outfit.items():
        image_path = os.path.join('d:/test/', category, file)
        if file != 'No items suitable for current season' and os.path.isfile(image_path):
            image = Image.open(image_path)
            image.thumbnail((150, 150))
            img_display = ImageTk.PhotoImage(image)

            img_label = tk.Label(image=img_display)
            img_label.image = img_display  # Keep a reference

            if category == 'Topwear':
                img_label.pack(in_=topwear_frame)
                topwear_images.append(img_label)
            elif category == 'Bottomwear':
                img_label.pack(in_=bottomwear_frame)
                bottomwear_images.append(img_label)
            elif category == 'Shoes':
                img_label.pack(in_=shoes_frame)
                shoes_images.append(img_label)
            elif category == 'Watches':
                img_label.pack(in_=watches_frame)
                watches_images.append(img_label)

# ... [Rest of your code] ...


# Add a button to recommend outfits
recommend_button = tk.Button(root, text='Recommend Outfit', command=recommend_outfit)
recommend_button.pack()

weather_button = tk.Button(root, text='Weather', command=recommend_weather_outfit)
weather_button.pack()



# Packing the canvas and scrollbar
canvas.pack(side='left', fill='both', expand=True)
scrollbar.pack(side='right', fill='y')

# Starting the GUI loop
root.mainloop()
