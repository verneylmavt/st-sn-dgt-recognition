# 🔢 Single Digit Recognition Model Collections

This repository contains machine learning models of Single Digit Recognition, designed to be deployed using ONNX and utilized in a Streamlit-based web application. The app provides an interactive interface for performing this task using neural network architectures. [Check here to see other ML tasks](https://github.com/verneylmavt/ml-model).

For more information about the training process, please check the `.ipynb` files in the `training` folder.

## 🎈 Demo App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://verneylogyt-sn-dgt-recognition.streamlit.app/)

![Demo GIF](https://github.com/verneylmavt/st-sn-dgt-recognition/blob/main/assets/demo.gif)

If you encounter message `This app has gone to sleep due to inactivity`, click `Yes, get this app back up!` button to wake the app back up.

<!-- [https://verneylogyt.streamlit.app/](https://verneylogyt.streamlit.app/) -->

## ⚙️ Running Locally

If the demo page is not working, you can fork or clone this repository and run the application locally by following these steps:

### Prerequisites

Ensure you have the following installed:

- Python 3.8 or later
- pip (Python Package Installer)

### Installation Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/verneylmavt/st-sn-dgt-recognition.git
   cd st-sn-dgt-recognition
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Acknowledgement

I acknowledge the use of the **MNIST** dataset provided by **Yann LeCun, Corinna Cortes, and Christopher J.C. Burges**. This dataset has been instrumental in conducting the research and developing this project.

- **Dataset Name**: MNIST (Modified National Institute of Standards and Technology) database
- **Source**: [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)
- **License**: [Specific licensing information may be required; please refer to the dataset's source for details.]
- **Description**: This dataset contains 70,000 grayscale images of handwritten digits (0-9), each 28x28 pixels in size. It is commonly used for training and testing in the field of machine learning and has become a benchmark for evaluating image processing systems.

I deeply appreciate the efforts of Yann LeCun, Corinna Cortes, and Christopher J.C. Burges in making this dataset available.
