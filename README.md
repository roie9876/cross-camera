# Cross-Camera Tracking

This project demonstrates live cross-camera tracking using object detection and re-identification. The application uses YOLO for object detection and ResNet50 for feature extraction.

## Features

- Live video stream processing
- Object detection and re-identification
- Real-world location estimation
- Streamlit-based UI

## Requirements

- Python 3.11
- OpenCV
- PyTorch
- torchvision
- ultralytics
- scipy
- Pillow
- streamlit
- geopy

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/cross-camera.git
    cd cross-camera
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Run the Streamlit application:
    ```sh
    streamlit run cross-camera.py
    ```

2. Upload two video files and start the live stream.

## License

This project is licensed under the MIT License.
```