# 👗 Fashion-MNIST Clothing Classifier

A deep learning project that classifies clothing images into 10 categories using **transfer learning with VGG16** in PyTorch. A Streamlit web app lets you upload an image and see the predicted class, the confidence and a probability chart for all classes.

---

## 📸 Screenshots


| 

![Screenshot 1](Screenshot%202025-09-30%20214553.png)

 | 

![Screenshot 2](Screenshot%202025-10-09%20012225.png)

 | 

![Screenshot 3](Screenshot%202025-10-09%20012308.png)

 |

---

## 🚀 Features

- Upload a JPG or PNG image of a clothing item and get the predicted class with confidence
- Interactive Plotly bar chart with the probability of all 10 classes
- Option to invert colors for photos with light backgrounds
- Sidebar with instructions, options and an about section
- Training script with PyTorch `Dataset`, `DataLoader`, Adam optimizer and `CrossEntropyLoss`

---

## 📊 Dataset

**Fashion-MNIST:** grayscale images of 28x28 pixels in 10 clothing categories. The training script reads the data from a CSV file (`imagedata`) where the first column is the label and the remaining 784 columns are pixel values.

| Label | Class |
|---|---|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle Boot |

**Split:** 80% training and 20% testing (`random_state=42`).

---

## 🧠 Model & Training

- **Base model:** VGG16 pretrained on ImageNet, with the convolutional feature layers frozen
- **Custom classifier head:** `Linear(25088, 1024)` → ReLU → Dropout(0.3) → `Linear(1024, 512)` → ReLU → Dropout(0.3) → `Linear(512, 10)`
- **Preprocessing:** each 28x28 image is converted to 3 channels, resized to 256, center-cropped to 224 and normalised with ImageNet mean and standard deviation
- **Loss:** `CrossEntropyLoss`
- **Optimizer:** Adam (learning rate 0.0001), trained only on the classifier head
- **Epochs / batch size:** 10 epochs, batch size 32
- **Evaluation:** train and test accuracy are printed at the end of the training script

---

## 🧩 Key Concepts

**Transfer learning.** Instead of training a network from scratch, VGG16 (already trained on ImageNet) is reused as a feature extractor. Its convolutional layers are frozen and only the new classifier head is trained, which needs less data and compute.

**Training loop in PyTorch.** Data is loaded in batches with `DataLoader`. For each batch the model makes predictions, `CrossEntropyLoss` measures the error, Autograd computes the gradients and Adam updates the classifier weights.

**Data preprocessing for a pretrained model.** VGG16 expects 3-channel 224x224 images with ImageNet normalisation, so the grayscale Fashion-MNIST images are converted to match.

**Probability output.** A softmax over the 10 outputs gives a probability for every class. The app shows the top prediction with its confidence and all probabilities as a bar chart.

**Model serving with Streamlit.** The trained weights are loaded once with `st.cache_resource` so that predictions are fast and anyone can test the model from the browser.

---

## 🛠️ Tech Stack

Python, PyTorch, torchvision, scikit-learn, Pandas, NumPy, Matplotlib, Plotly, Pillow, Streamlit

---

## 📂 Project Structure

```
.
├── Fmnist_image_model.py        # Dataset class, VGG16 model and training/evaluation (Google Colab)
├── Streamlit_web_app.py         # Streamlit web app
├── requirements.txt             # Python dependencies
├── Screenshot ... .png          # App screenshots
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository

```
git clone https://github.com/HamzaWaseem2005/Fmnist_Cloth_Model.git
cd Fmnist_Cloth_Model
```

### 2. Create a virtual environment

```
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

### 4. Add the model weights

The trained weights file `vgg16_weights.pth` is not stored in this repository because of its size (VGG16 weights are large). To create it, run `Fmnist_image_model.py` in Google Colab with the dataset file uploaded. The script ends with:

```python
torch.save(vgg16.state_dict(), "vgg16_weights.pth")
```

Download the file and place it in the project root, next to `Streamlit_web_app.py`.

### 5. Run the app

```
streamlit run Streamlit_web_app.py
```

---

## 🖼️ How to Use the App

1. Upload a clear photo of a single clothing item (JPG or PNG).
2. The app converts the image to 28x28 grayscale, like the training data.
3. It shows the predicted class and the confidence.
4. Use the sidebar to show or hide the probability chart.
5. If the photo has a light background and the prediction looks wrong, tick **Invert colors**.

**Tip:** simple images of one item on a plain background work best.

---

## ⚠️ Limitations

- The model is trained on small 28x28 grayscale images. Real photos with busy backgrounds, shadows or colours can be predicted less accurately than Fashion-MNIST test images.
- Shirt, T-shirt/top, Pullover and Coat look similar at low resolution and are the classes most likely to be confused.
- The training script is written for Google Colab (it uses `google.colab.files`) and needs small edits to run locally.
- This is a learning project, not a production system.

## 🚀 Future Improvements

- Report test accuracy and a confusion matrix in this README
- Data augmentation so that real-world photos are handled better
- Fine-tune some VGG16 layers and compare with a custom CNN
- Host the weights on Hugging Face Hub and deploy the app on Streamlit Community Cloud

---

## 👤 Author

**Muhammad Hamza Waseem**
[GitHub](https://github.com/HamzaWaseem2005) · [LinkedIn](https://www.linkedin.com/in/muhammad-hamza-waseem-976535336)
