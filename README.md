🩺 Skin Cancer Classification with CNN
This project is a Convolutional Neural Network (CNN) based skin cancer classification system built using TensorFlow/Keras and deployed with Streamlit.
It allows users to upload a skin lesion image and get a prediction of its category with confidence scores.
⚠ Note: The trained model file (cancer_model.h5) is not included in this repository due to size limits.
However, you can train it yourself using the provided Jupyter notebooks.

🧠 Dataset
This project is trained on the HAM10000 dataset — a large collection of multi-source dermatoscopic images of common pigmented skin lesions.
You can download it here:
HAM10000 Dataset (Kaggle)

⚙️ Setup Instructions
1️⃣ Clone the repository
git clone https://github.com/yourusername/skin-cancer-classification.git
cd skin-cancer-classification

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Train the model
Since the trained model is not provided, you need to run the notebook to train it.
jupyter notebook notebooks/train_model.ipynb
This will create a cancer_model.h5 file inside the model/ folder.

4️⃣ Run the Streamlit app
streamlit run app.py

🖼️ Usage
Upload an image of a skin lesion (.jpg, .png, .jpeg).
Click Predict.
View the predicted category and confidence score.

📌 Example Output
Uploaded Image:
Prediction: Melanoma
Confidence: 92.45%

📦 Requirements
See requirements.txt for the full list of dependencies.
Main packages:

TensorFlow / Keras
NumPy
Pillow
Streamlit
🚀 Future Improvements
Deploy as a web service (e.g., on Hugging Face Spaces).
Support multiple images at once.
Add Grad-CAM visualizations for interpretability.
