This was for my science fair at Baton Rouge Magnet High School. You can use this project for whatever you want.

## Dependencies

This project requires the following Python libraries to be installed on your Raspberry Pi:

### Core Libraries
- **OpenCV** (`cv2`) - Computer vision and image processing
- **NumPy** - Numerical operations and array handling
- **Pynput** - Keyboard input simulation (for testing GPIO functionality)
- **Picamera2** - Raspberry Pi camera interface
- **AI Edge LiteRT** (`ai_edge_litert`) - Google's on-device ML inference runtime (formerly TensorFlow Lite)
- **Piper TTS** - Text-to-speech synthesis for audio feedback

### Installation Commands
```bash
# Install core dependencies
pip install opencv-python numpy pynput picamera2 ai-edge-litert

# Install Piper TTS (requires separate installation)
# Download from: https://github.com/OHF-Voice/piper1-gpl
```
### Hardware Requirements
- Raspberry Pi 5 (tested) or compatible model
- Raspberry Pi Camera Module (I've used IMX708)
- USB audio output device (for TTS playback)
- GPIO buttons (pins 17 and 26)
- 15000mAh 3A Power Bank or
- MINIMUM 3A power source

