from flask import request, jsonify
from app import app
from utils.models_utils import make_prediction, load_model_and_classes, load_model_and_classes2, load_model_and_classes3, load_model_and_classes4,load_model_and_classes5, load_model_and_classes6,load_model_and_classes7

# Load model and class indices
model, class_indices = load_model_and_classes('D:/STYLIN AI MODEL/bottomwear_classifier_model.h5')
model2, class_indices2 = load_model_and_classes2('D:/STYLIN AI MODEL/topwear_classifier_model.h5')
model3, class_indices3 = load_model_and_classes3('D:/STYLIN AI MODEL/primary_classifier_complex_model.h5')
model4, class_indices4 = load_model_and_classes4('D:/STYLIN AI MODEL/gender_classifier_complex_model.h5')
model5, class_indices5 = load_model_and_classes5('D:/STYLIN AI MODEL/color_classifier_model.h5')
model6, class_indices6 = load_model_and_classes6('D:/STYLIN AI MODEL/typecloth_classifier_model.h5')
model7, class_indices7 = load_model_and_classes7('D:/STYLIN AI MODEL/season_classifier_model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        # Save the file to a temporary file
        img_path = 'temp_image.jpg'
        file.save(img_path)

        # Make prediction
        predicted_class3= make_prediction(model3,img_path,class_indices3)
        predicted_class4=make_prediction(model4,img_path,class_indices4)
        predicted_class5=make_prediction(model5,img_path,class_indices5)
        predicted_class6=make_prediction(model6, img_path, class_indices6)
        predicted_class7=make_prediction(model7, img_path, class_indices7)
        if(predicted_class3=='Bottomwear'):
            predicted_class = make_prediction(model, img_path, class_indices)
            if(predicted_class4=='Men' and predicted_class=='Leggings'):
                predicted_class='Track Pants'
            if(predicted_class=='Jeans'):
                predicted_class7='Winter'
            return jsonify({'prediction':predicted_class3, 'prediction2':predicted_class4,'prediction1':predicted_class,'prediction3':predicted_class5, 'prediction4':predicted_class6,'prediction5':predicted_class7})
        elif(predicted_class3=='Topwear'):
            predicted_class2= make_prediction(model2, img_path, class_indices2)
            if(predicted_class2=='Sweatshirts' or predicted_class2=='Jackets'):
                predicted_class7='Winter'
            return jsonify({'prediction':predicted_class3, 'prediction2':predicted_class4,'prediction1':predicted_class2,'prediction3':predicted_class5, 'prediction4':predicted_class6,'prediction5':predicted_class7})
        else:
            return jsonify({'prediction':predicted_class3,'prediction1':predicted_class4,'prediction2':predicted_class5, 'prediction4':predicted_class6,'prediction5':predicted_class7})

        #return jsonify({'prediction': predicted_class, 'prediction2':predicted_class2, 'prediction3':predicted_class3 })
