import cv2

def main():
    # Attempt to open the built-in FaceTime camera on macOS.
    # If this doesn't work, try cap = cv2.VideoCapture(0) without CAP_AVFOUNDATION.
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

    if not cap.isOpened():
        print("Cannot open camera. Make sure you have granted camera permissions.")
        return

    # Define video writer settings
    # MJPG is widely supported; you can also try 'mp4v' for MP4 containers.
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    fps = 20  # Adjust as needed
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Output filename and VideoWriter initialization
    out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

    print("Recording... Press 'q' to stop.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from camera. Exiting...")
            break

        # Show live feed in a window
        cv2.imshow('Mac Camera', frame)

        # Write frame to the output file
        out.write(frame)

        # Press 'q' in the window to stop recording
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release camera and file writer resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Video saved to 'output.mp4'")

if __name__ == "__main__":
    main()