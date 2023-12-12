from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

def load_model_and_classes(model_path):
    # Load the model
    model = load_model(model_path)
    # Assuming class_indices are the same as when you trained your model
    #class_indices =  {"Blazers":0, "Kurtas":1, "Shirts":2, "Sweatshirts":3, "Tshirts":4}
    #class_indices= {'Jeans': 0, 'Shorts': 1, 'Track Pants': 2, 'Trousers': 3}
    #class_indices= {"Capris": 0, "Jeans": 1, "Leggings": 2, "Salwar and Dupatta": 3, "Shorts": 4, "Skirts": 5, "Stockings": 6, "Track Pants": 7, "Trousers": 8}
    class_indices= {'Jeans': 0, 'Leggings': 1, 'Salwar and Dupatta': 2, 'Shorts': 3, 'Skirts': 4, 'Track Pants': 5, 'Trousers': 6}
    return model, class_indices,
def load_model_and_classes2(model_path):
    # Load the model
    model = load_model(model_path)
    # Assuming class_indices are the same as when you trained your model
    #class_indices2 =  {"Blazers": 0, "Kurtas": 1, "Shirts": 2, "Sweatshirts": 3, "Tshirts": 4}
    class_indices2= {'Blazers': 0, 'Jackets': 1, 'Kurtas': 2, 'Kurtis': 3, 'Shirts': 4, 'Sweaters': 5, 'Sweatshirts': 6, 'Tops': 7, 'Tshirts': 8}
    #class_indices2={'Blazers': 0, 'Kurtas': 1, 'Kurtis': 2, 'Shirts': 3, 'Sweatshirts': 4, 'Tops': 5, 'Tshirts': 6}
    return model, class_indices2

def load_model_and_classes3(model_path):
    # Load the model
    model = load_model(model_path)
    # Assuming class_indices are the same as when you trained your model
    class_indices3 =  {"Bags": 0, "Bottomwear": 1, "Shoes": 2, "Topwear": 3, "Watches": 4}
    #class_indices2= {'Bottomwear': 0, 'Shoes': 1, 'Topwear': 2}
    return model, class_indices3

def load_model_and_classes4(model_path):
    # Load the model
    model = load_model(model_path)
    # Assuming class_indices are the same as when you trained your model
    class_indices4 =  {"Boys": 0, "Girls": 1, "Men": 2, "Unisex": 3, "Women": 4}
    #class_indices2= {'Bottomwear': 0, 'Shoes': 1, 'Topwear': 2}
    return model, class_indices4

def load_model_and_classes5(model_path):
    # Load the model
    model = load_model(model_path)
    # Assuming class_indices are the same as when you trained your model
    class_indices5 = {'Beige': 0, 'Black': 1, 'Blue': 2, 'Bronze': 3, 'Brown': 4, 'Burgundy': 5, 'Charcoal': 6, 'Coffee Brown': 7, 'Copper': 8, 'Cream': 9, 'Fluorescent Green': 10, 'Gold': 11, 'Green': 12, 'Grey': 13, 'Grey Melange': 14, 'Khaki': 15, 'Lavender': 16, 'Lime Green': 17, 'Magenta': 18, 'Maroon': 19, 'Mauve': 20, 'Metallic': 21, 'Multi': 22, 'Mushroom Brown': 23, 'Mustard': 24, 'Navy Blue': 25, 'Nude': 26, 'Off White': 27, 'Olive': 28, 'Orange': 29, 'Peach': 30, 'Pink': 31, 'Purple': 32, 'Red': 33, 'Rose': 34, 'Rust': 35, 'Sea Green': 36, 'Silver': 37, 'Skin': 38, 'Steel': 39, 'Tan': 40, 'Taupe': 41, 'Teal': 42, 'Turquoise Blue': 43, 'White': 44, 'Yellow': 45, 'nan': 46}
    #class_indices2= {'Bottomwear': 0, 'Shoes': 1, 'Topwear': 2}
    return model, class_indices5

def load_model_and_classes6(model_path):
    # Load the model
    model = load_model(model_path)
    # Assuming class_indices are the same as when you trained your model
    class_indices6 = {'Casual': 0, 'Ethnic': 1, 'Formal': 2, 'Sports': 3}
    #class_indices2= {'Bottomwear': 0, 'Shoes': 1, 'Topwear': 2}
    return model, class_indices6

def load_model_and_classes7(model_path):
    # Load the model
    model = load_model(model_path)
    # Assuming class_indices are the same as when you trained your model
    class_indices7 = {'Fall': 0, 'Summer': 1, 'Winter': 2}
    #class_indices2= {'Bottomwear': 0, 'Shoes': 1, 'Topwear': 2}
    return model, class_indices7

def make_prediction(model, img_path, class_indices):
    img = image.load_img(img_path, target_size=(60, 80))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    predicted_class = list(class_indices.keys())[predicted_class_index]
    return predicted_class
