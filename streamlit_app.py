import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import requests
import io
from PIL import Image
import gdown

# Page configuration
st.set_page_config(
    page_title="MNIST Digit Generator",
    page_icon="🔢",
    layout="wide"
)

# Generator class (same as in training)
class Generator(nn.Module):
    def __init__(self, noise_dim, num_classes, img_size):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.num_classes = num_classes
        self.img_size = img_size
        
        # Label embedding
        self.label_embedding = nn.Embedding(num_classes, num_classes)
        
        # Input dimension: noise_dim + num_classes
        input_dim = noise_dim + num_classes
        
        self.model = nn.Sequential(
            # First layer: 128 -> 7*7*256
            nn.Linear(input_dim, 256 * 7 * 7),
            nn.BatchNorm1d(256 * 7 * 7),
            nn.ReLU(True),
            
            # Reshape to (batch_size, 256, 7, 7)
            nn.Unflatten(1, (256, 7, 7)),
            
            # Upsample to 14x14
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            # Upsample to 28x28
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            # Final layer
            nn.ConvTranspose2d(64, 1, 3, 1, 1),
            nn.Tanh()
        )
    
    def forward(self, noise, labels):
        # Embed labels
        label_embedding = self.label_embedding(labels)
        
        # Concatenate noise and label embedding
        gen_input = torch.cat([noise, label_embedding], dim=1)
        
        # Generate image
        img = self.model(gen_input)
        return img

@st.cache_resource
def load_model():
    """Load the trained generator model"""
    try:
        # Set device
        device = torch.device('cpu')  # Use CPU for deployment
        
        # Model configuration
        NOISE_DIM = 100
        NUM_CLASSES = 10
        IMAGE_SIZE = 28
        
        # Initialize generator
        generator = Generator(NOISE_DIM, NUM_CLASSES, IMAGE_SIZE)
        
        # Try to load from different sources
        model_loaded = False
        
        # Method 1: Try to load from local file
        try:
            checkpoint = torch.load('mnist_cgan_models.pth', map_location=device)
            generator.load_state_dict(checkpoint['generator_state_dict'])
            model_loaded = True
            st.success("Model loaded from local file!")
        except:
            pass
        
        # Method 2: Try to download from Google Drive (you'll need to upload your model here)
        if not model_loaded:
            try:
                # Replace with your actual Google Drive file ID
                file_id = "1DiiPY1flp8fyPB1xoRerYt_ILlBgjo1F"
                url = f"https://drive.google.com/uc?id={file_id}"
                
                # Download model
                gdown.download(url, 'downloaded_model.pth', quiet=False)
                checkpoint = torch.load('downloaded_model.pth', map_location=device)
                generator.load_state_dict(checkpoint['generator_state_dict'])
                model_loaded = True
                st.success("Model loaded from Google Drive!")
            except:
                pass
        
        if not model_loaded:
            st.error("Could not load model. Please check model file.")
            return None, None, None, None
        
        generator.eval()
        return generator, device, NOISE_DIM, NUM_CLASSES
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None, None

def generate_digit_images(generator, device, noise_dim, digit, num_images=5):
    """Generate multiple images for a specific digit"""
    if generator is None:
        return []
    
    images = []
    
    with torch.no_grad():
        for _ in range(num_images):
            # Generate random noise
            noise = torch.randn(1, noise_dim, device=device)
            label = torch.tensor([digit], device=device)
            
            # Generate image
            fake_image = generator(noise, label)
            
            # Convert to numpy and denormalize
            img = fake_image.cpu().squeeze().numpy()
            img = (img + 1) / 2  # Denormalize from [-1, 1] to [0, 1]
            img = np.clip(img, 0, 1)  # Ensure values are in [0, 1]
            
            images.append(img)
    
    return images

def display_images(images, digit):
    """Display generated images in a grid"""
    if not images:
        st.error("No images to display")
        return
    
    cols = st.columns(5)
    
    for i, img in enumerate(images):
        with cols[i]:
            # Convert to PIL Image for better display
            img_pil = Image.fromarray((img * 255).astype(np.uint8))
            img_pil = img_pil.resize((112, 112), Image.NEAREST)  # Upscale for better visibility
            
            st.image(img_pil, caption=f"Generated {digit} #{i+1}", use_column_width=True)

# Main app
def main():
    st.title("🔢 MNIST Handwritten Digit Generator")
    st.markdown("---")
    
    st.markdown("""
    This app uses a **Conditional GAN** trained on the MNIST dataset to generate handwritten digits.
    Select a digit below and click generate to see 5 unique variations!
    """)
    
    # Load model
    with st.spinner("Loading trained model..."):
        generator, device, noise_dim, num_classes = load_model()
    
    if generator is None:
        st.stop()
    
    # Digit selection
    st.markdown("### Select a digit to generate:")
    
    # Create digit selection buttons
    cols = st.columns(10)
    selected_digit = None
    
    for i in range(10):
        with cols[i]:
            if st.button(str(i), key=f"digit_{i}", use_container_width=True):
                selected_digit = i
    
    # Alternative: Selectbox for digit selection
    st.markdown("**Or use the dropdown:**")
    dropdown_digit = st.selectbox("Choose digit:", list(range(10)), key="dropdown")
    
    # Use button selection if available, otherwise use dropdown
    if selected_digit is not None:
        digit_to_generate = selected_digit
    else:
        digit_to_generate = dropdown_digit
    
    st.markdown(f"**Selected digit: {digit_to_generate}**")
    
    # Generate button
    if st.button("🎨 Generate 5 Images", type="primary", use_container_width=True):
        with st.spinner(f"Generating 5 images of digit {digit_to_generate}..."):
            # Generate images
            images = generate_digit_images(
                generator, device, noise_dim, digit_to_generate, num_images=5
            )
            
            if images:
                st.markdown(f"### Generated Images for Digit {digit_to_generate}")
                display_images(images, digit_to_generate)
                
                st.markdown("---")
                st.markdown("**Note:** Each generation produces unique variations of the selected digit!")
            else:
                st.error("Failed to generate images. Please try again.")
    
    # Information section
    st.markdown("---")
    with st.expander("ℹ️ About this model"):
        st.markdown("""
        **Model Architecture:** Conditional Generative Adversarial Network (cGAN)
        
        **Training Details:**
        - Dataset: MNIST (28×28 grayscale handwritten digits)
        - Framework: PyTorch
        - Training Environment: Google Colab with T4 GPU
        - Training from scratch (no pre-trained weights)
        
        **How it works:**
        1. The generator creates images from random noise + digit label
        2. The discriminator learns to distinguish real vs generated images
        3. Through adversarial training, the generator learns to create realistic digits
        
        **Diversity:** Each generation uses different random noise, creating unique variations!
        """)

if __name__ == "__main__":
    main()