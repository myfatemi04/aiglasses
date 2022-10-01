import cv2
import dlib
import numpy as np

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
cap = cv2.VideoCapture(0)

px_per_foot = 180

while True:
    _, image = cap.read()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 1)
    # Detect landmarks for each face
    for rect in rects:
        # Get the landmark points
        shape = predictor(gray, rect)
        # Convert it to the NumPy Array
        shape_np = np.zeros((68, 2), dtype="int")
        for i in range(0, 68):
            shape_np[i] = (shape.part(i).x, shape.part(i).y)
        shape = shape_np

        current_dist_px = np.sum((shape[39] - shape[42]) ** 2) ** 0.5

        scale_from_1ft = px_per_foot / current_dist_px

        print(f"Estimated distance: {scale_from_1ft:.2f} ft")

        # Display the landmarks
        for i, (x, y) in enumerate(shape):
            # Draw the circle to mark the keypoint
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

    cv2.imshow('Landmark Detection', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
