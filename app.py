import os
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_extras.chart_container import chart_container
from streamlit_extras.mention import mention
from streamlit_extras.echo_expander import echo_expander
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import onnxruntime as ort
from torchvision import transforms

# ----------------------
# Configuration
# ----------------------

# ----------------------
# Model Information
# ----------------------
model_info = {
    "rnet-cnn": {
        "subheader": "Model: ResNet CNN",
        "pre_processing": """
Dataset = Modified National Institute of Standards and Technology (MNIST)
        """,
        "parameters": """
Batch Size = 128

Epochs = 15
Learning Rate = 0.001
Learning Rate Scheduler = StepLR
Loss Function = CrossEntropyLoss
Optimizer = AdamW
Weight Decay = 0.0001
        """,
        "model_code": """
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class Model(nn.Module):
    def __init__(self, num_classes=10):
        super(Model, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.layer2 = Block(32, 32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.layer4 = Block(64, 64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.layer6 = Block(128, 128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.pool1(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.pool2(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.pool3(out)
        out = self.global_avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc(out)
        return out
        """
    }
}

# ----------------------
# Loading Function
# ----------------------

@st.cache_resource
def load_model(model_name):
    try:
        model_path = os.path.join("models", str(model_name), "model.onnx")
        ort_session = ort.InferenceSession(model_path)
    except FileNotFoundError:
        st.error(f"Model file not found for {model_name}. Please ensure 'model-state.pth' exists in the model directory.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading the model for {model_name}: {e}")
        st.stop()
    return ort_session
        
@st.cache_data
def load_training_data():
    training_data = {
        "Epoch": list(range(1, 16)),
        "Train Loss": [
            0.2140, 0.0643, 0.0524, 0.0463, 0.0401, 
            0.0369, 0.0348, 0.0196, 0.0175, 0.0168, 
            0.0151, 0.0147, 0.0145, 0.0142, 0.0118
        ],
        "Train Accuracy": [
            0.9367, 0.9819, 0.9849, 0.9864, 0.9881, 
            0.9892, 0.9900, 0.9946, 0.9952, 0.9951, 
            0.9955, 0.9961, 0.9960, 0.9960, 0.9967
        ],
        "Validation Loss": [
            0.2009, 0.0444, 0.1140, 0.0608, 0.0431, 
            0.0355, 0.0497, 0.0199, 0.0177, 0.0152, 
            0.0188, 0.0176, 0.0180, 0.0164, 0.0184
        ],
        "Validation Accuracy": [
            0.9334, 0.9862, 0.9650, 0.9814, 0.9877, 
            0.9895, 0.9857, 0.9940, 0.9946, 0.9951, 
            0.9937, 0.9947, 0.9956, 0.9955, 0.9953
        ]
    }
    return pd.DataFrame(training_data)

# ----------------------
# Prediction Function
# ----------------------

def predict_digit(ort_session, pil_image):
    input_shape = ort_session.get_inputs()[0].shape
    _, _, target_height, target_width = input_shape

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((target_height, target_width)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    image = transform(pil_image)
    if image.mean() > 0.5:
        image = 1.0 - image
    image = image.unsqueeze(0).numpy().astype(np.float32)

    ort_inputs = {ort_session.get_inputs()[0].name: image}
    ort_outs = ort_session.run(None, ort_inputs)
    predicted = np.argmax(ort_outs[0], axis=1)[0]

    return predicted

# ----------------------
# Page UI
# ----------------------
def main():
    st.title("Single Digit Recognition")
    
    model_names = list(model_info.keys())
    model = st.selectbox("Select a Model", model_names)
    st.divider()
    
    net = load_model(model)
    training_data = load_training_data()
    
    st.subheader(model_info[model]["subheader"])
    
    with st.form(key="dgt_recognition_form"):
        # user_input = st.text_input("Enter Text Here:")
        st.write("Draw Digit Here:")
        canvas_result = st_canvas(
            fill_color="#eee",
            stroke_width=17,
            stroke_color="#000000",
            background_color="#FFFFFF",
            update_streamlit=True,
            height=125,
            width=125,
            drawing_mode='freedraw',
            key="canvas",
        )
        
        submit_button = st.form_submit_button(label="Recognize")
        
        if submit_button:
            if canvas_result.image_data is None:
                st.warning("Please draw a digit.")
            else:
                pil_image = Image.fromarray(canvas_result.image_data.astype("uint8"), "RGBA").convert("L")
                digit = predict_digit(net, pil_image)
                st.success(f"Predicted Digit: {digit}")
                canvas_result.image_data = None
    
    # st.divider()        
    st.feedback("thumbs")
    # st.warning("""Disclaimer: This model has been quantized for optimization.""")
    mention(
            label="GitHub Repo: verneylmavt/st-sn-dgt-recognition",
            icon="github",
            url="https://github.com/verneylmavt/st-sn-dgt-recognition"
        )
    mention(
            label="Other ML Tasks",
            icon="streamlit",
            url="https://verneylogyt.streamlit.app/"
        )
    st.divider()
    
    if "pre_processing" in model_info[model]:
        st.subheader("""Pre-Processing""")
        st.code(model_info[model]["pre_processing"], language="None")
    else: pass
    
    if "parameters" in model_info[model]:
        st.subheader("""Parameters""")
        st.code(model_info[model]["parameters"], language="None")
    else: pass
    
    st.subheader("""Model""")
    with echo_expander(code_location="below", label="Code"):
        import torch
        import torch.nn as nn
        
        
        class Block(nn.Module):
            def __init__(self, in_channels, out_channels, stride=1, downsample=None):
                super(Block, self).__init__()
                # First Convolutional Layer for Residual Block
                self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                                    stride=stride, padding=1, bias=False)
                # Batch Normalization Layer for First Convolution
                self.bn1 = nn.BatchNorm2d(out_channels)
                # Activation Layer for Non-Linear Transformation
                self.relu = nn.ReLU(inplace=True)
                # Second Convolutional Layer for Residual Block
                self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                                    stride=1, padding=1, bias=False)
                # Batch Normalization Layer for Second Convolution
                self.bn2 = nn.BatchNorm2d(out_channels)
                
                # Downsampling Layer for Adjusting Identity Shortcut
                self.downsample = downsample
            
            def forward(self, x):
                # Identity Shortcut Connection
                identity = x
                
                # First Convolutional Layer Transformation
                out = self.conv1(x)
                out = self.bn1(out)
                out = self.relu(out)
                # Second Convolutional Layer Transformation
                out = self.conv2(out)
                out = self.bn2(out)
                
                # Downsampling of Identity Shortcut (if Applicable)
                if self.downsample:
                    identity = self.downsample(x)
                # Addition of Identity Shortcut to Residual Block Output
                out += identity
                # Activation of Residual Block Output
                out = self.relu(out)
                return out
        
        
        class Model(nn.Module):
            def __init__(self, num_classes=10):
                super(Model, self).__init__()
                
                # First Convolutional Layer for Feature Extraction
                self.layer1 = nn.Sequential(
                    nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True)
                )
                # First Residual Block for Feature Refinement
                self.layer2 = Block(32, 32)
                # First Max Pooling Layer for Spatial Downsampling
                self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
                
                # Second Convolutional Layer for Feature Extraction
                self.layer3 = nn.Sequential(
                    nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True)
                )
                # Second Residual Block for Feature Refinement
                self.layer4 = Block(64, 64)
                # Second Max Pooling Layer for Spatial Downsampling
                self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
                
                # Third Convolutional Layer for Feature Extraction
                self.layer5 = nn.Sequential(
                    nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True)
                )
                # Third Residual Block for Feature Refinement
                self.layer6 = Block(128, 128)
                # Third Max Pooling Layer for Spatial Downsampling
                self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
                
                # Global Average Pooling Layer for Spatial Dimension Reduction
                self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
                # Dropout Layer for Regularization
                self.dropout = nn.Dropout(0.5)
                # Fully Connected Layer for Digit Classification
                self.fc = nn.Linear(128, num_classes)
            
            def forward(self, x):
                # First Convolutional Layer Transformation
                out = self.layer1(x)
                # First Residual Block Transformation
                out = self.layer2(out)
                # First Max Pooling Layer Transformation
                out = self.pool1(out)
                
                # Second Convolutional Layer Transformation
                out = self.layer3(out)
                # Second Residual Block Transformation
                out = self.layer4(out)
                # Second Max Pooling Layer Transformation
                out = self.pool2(out)
                
                # Third Convolutional Layer Transformation
                out = self.layer5(out)
                # Third Residual Block Transformation
                out = self.layer6(out)
                # Third Max Pooling Layer Transformation
                out = self.pool3(out)
                
                # Global Average Pooling Transformation
                out = self.global_avg_pool(out)
                # Flattening of Pooled Output
                out = out.view(out.size(0), -1)
                # Dropout Application to Flattened Output
                out = self.dropout(out)
                # Transformation of Flattened Features → Digit Scores
                out = self.fc(out)
                return out
    # st.code(model_info[model]["model_code"], language="python")
    
    if "forward_pass" in model_info[model]:
        st.subheader("Forward Pass")
        for key, value in model_info[model]["forward_pass"].items():
            st.caption(key)
            st.latex(value)
    else: pass
    
    st.subheader("""Training""")
    with chart_container(training_data):
        st.line_chart(training_data.set_index("Epoch"))

if __name__ == "__main__":
    main()