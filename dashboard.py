import streamlit as st
import numpy as np
import tensorflow as tf

# Define the number of latent variables
num_latent_vars = 16

with st.sidebar:
    sliders = []
    for i in range(num_latent_vars):
        slider = st.slider(f'F{i+1}', -5, 5, 1)
        sliders.append(slider)

# Load models and set cache to prevent reloading
@st.cache(allow_output_mutation=True)
def load_models():
    encoder = tf.keras.models.load_model('VAE_512_encoder', compile=False)
    decoder = tf.keras.models.load_model('VAE_512_decoder', compile=False)
    return encoder, decoder

# Load data and set cache to prevent reloading
@st.cache(allow_output_mutation=True)
def load_data():
    data_np = np.load('512.npy', allow_pickle=True)
    return data_np

# Reconstruct data
def reconstruct_data(data_np, encoder, decoder, sliders):
    # Create dummy latent variable layer
    latent_layer = np.zeros((1, num_latent_vars))
    
    # Assign slider values to the latent layer
    latent_layer[0] = np.array(sliders)
    
    # Concatenate latent variables with data
    latent_layer = np.concatenate((data_np[:, 512:, 0], latent_layer), axis=1)
    
    # Reconstruct data
    reconstructed_data = decoder.predict(latent_layer.astype('float32'))
    return reconstructed_data

# Main code
def main():
    st.write('## VAE Making sense of sensor data')

    # Load models
    encoder, decoder = load_models()

    # Load data
    data_np = load_data()

    epochLength = 512

    if st.button('Reconstruct Data'):
        with st.spinner("Reconstructing Data..."):
            file = np.random.randint(data_np.shape[0])
            st.line_chart(data_np[file, :epochLength, :])
            reconstructed_data = reconstruct_data(data_np, encoder, decoder, sliders)
            st.line_chart(reconstructed_data[0, :epochLength, :])


if __name__ == '__main__':
    main()
