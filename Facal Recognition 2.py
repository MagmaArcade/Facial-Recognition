from matplotlib.widgets import Button
from tkinter import Tk, Label, Entry, Button as TkButton, messagebox
import cv2
import face_recognition
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import roc_curve, auc
import csv
import datetime

# Function to draw boxes and labels on faces
def draw_face_boxes(image, face_locations, face_ids, liveness_scores):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)
    draw = ImageDraw.Draw(pil_image)
    font = ImageFont.truetype("arial.ttf", 16)  # Change font and size as needed

    for (top, right, bottom, left), student_id, liveness_score in zip(face_locations, face_ids, liveness_scores):
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))
        text = f"{student_id} ({'Live' if liveness_score else 'Spoof'})"
        text_bbox = font.getbbox(text)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5), text, fill=(255, 255, 255, 255), font=font)

    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

# Function for liveness detection
def detect_liveness(face_image):
    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    texture_analysis_result = cv2.Laplacian(gray, cv2.CV_64F).var()
    threshold = 130  # Adjust as needed
    is_live = texture_analysis_result > threshold
    return is_live

# Function to evaluate the face verification system using ROC curve and AUC
def evaluate_system(true_labels, similarity_scores):
    fpr, tpr, thresholds = roc_curve(true_labels, similarity_scores)
    roc_auc = auc(fpr, tpr)
    print("False Positive Rates: ", fpr)
    print("True Positive Rates: ", tpr)
    print("Thresholds: ", thresholds)
    print("AUC: ", roc_auc)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()

# Load sample pictures and learn how to recognize them
def load_known_faces(image_folder):
    known_face_encodings = []
    known_face_ids = []
    for file_name in os.listdir(image_folder):
        if file_name.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image_path = os.path.join(image_folder, file_name)
            image = face_recognition.load_image_file(image_path)
            face_encoding = face_recognition.face_encodings(image)[0]
            student_id = os.path.splitext(file_name)[0]
            known_face_encodings.append(face_encoding)
            known_face_ids.append(student_id)
    print('Learned encoding for', len(known_face_encodings), 'images.')
    return known_face_encodings, known_face_ids

# Function to validate student information against CSV file
def validate_student_info(firstname, lastname, student_id, dob):
    with open('data/files/students.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if (row[0].lower() == firstname.lower() and row[1].lower() == lastname.lower() and row[2] == student_id and row[3] == dob):
                return True
    return False

# Function to write to attendance CSV file
def write_to_attendance_csv(student_id):
    attendance_csv_file = "data/files/attendance.csv"
    file_exists = os.path.exists(attendance_csv_file)
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(attendance_csv_file, mode='a', newline='') as file:
        fieldnames = ['Student ID', 'Time']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({'Student ID': student_id, 'Time': current_time})

# Function to handle unknown face detected
def unknown_face_detected():
    print("Unknown face detected!")
    root = Tk()
    root.title("Enter Details")
    Label(root, text="First Name:").grid(row=0, sticky="W")
    firstname_entry = Entry(root)
    firstname_entry.grid(row=0, column=1)
    Label(root, text="Last Name:").grid(row=1, sticky="W")
    lastname_entry = Entry(root)
    lastname_entry.grid(row=1, column=1)
    Label(root, text="Student ID:").grid(row=2, sticky="W")
    id_entry = Entry(root)
    id_entry.grid(row=2, column=1)
    Label(root, text="Date of Birth:").grid(row=3, sticky="W")
    dob_entry = Entry(root)
    dob_entry.grid(row=3, column=1)
    def submit_clicked():
        firstname = firstname_entry.get()
        lastname = lastname_entry.get()
        student_id = id_entry.get()
        dob = dob_entry.get()
        print("First Name:", firstname)
        print("Last Name:", lastname)
        print("Student ID:", student_id)
        print("DOB:", dob)
        if validate_student_info(firstname, lastname, student_id, dob):
            messagebox.showinfo("Notification", "Student information validated.")
            ret, frame = video_capture.read()
            if ret:
                cv2.imwrite(os.path.join(train_data_folder, f"{student_id}.jpg"), frame)
                print("Picture saved as", f"{student_id}.jpg")
                messagebox.showinfo("Notification", "Updating Student Database, Please Wait.")
                write_to_attendance_csv(student_id)
        else:
            messagebox.showerror("Error", "Invalid student information.")
        root.destroy()
    submit_button = TkButton(root, text="Submit", command=submit_clicked)
    submit_button.grid(row=4, columnspan=2)

def train_model(train_data_folder, val_data_folder):
    # Load training data
    train_encodings, train_ids = load_known_faces(train_data_folder)
    # Load validation data
    val_encodings, val_ids = load_known_faces(val_data_folder)

    # Placeholder for validation performance check
    best_model = None
    best_accuracy = 0

    for epoch in range(10):  # Placeholder for actual training iterations
        print(f"Epoch {epoch + 1}/10")
        
        # Training logic goes here (placeholder)
        
        # Validation logic
        correct_predictions = 0
        for val_encoding, val_id in zip(val_encodings, val_ids):
            matches = face_recognition.compare_faces(train_encodings, val_encoding)
            face_distances = face_recognition.face_distance(train_encodings, val_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                predicted_id = train_ids[best_match_index]
                if predicted_id == val_id:
                    correct_predictions += 1
        
        val_accuracy = correct_predictions / len(val_ids)
        print(f"Validation accuracy: {val_accuracy * 100:.2f}%")

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_model = (train_encodings, train_ids)

    return best_model

def test_model(test_data_folder, model):
    test_encodings, test_ids = load_known_faces(test_data_folder)
    train_encodings, train_ids = model

    correct_predictions = 0
    for test_encoding, test_id in zip(test_encodings, test_ids):
        matches = face_recognition.compare_faces(train_encodings, test_encoding)
        face_distances = face_recognition.face_distance(train_encodings, test_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            predicted_id = train_ids[best_match_index]
            if predicted_id == test_id:
                correct_predictions += 1

    test_accuracy = correct_predictions / len(test_ids)
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")

def verify_model(verification_data_folder, model):
    verification_encodings, verification_ids = load_known_faces(verification_data_folder)
    train_encodings, train_ids = model

    correct_predictions = 0
    for verification_encoding, verification_id in zip(verification_encodings, verification_ids):
        matches = face_recognition.compare_faces(train_encodings, verification_encoding)
        face_distances = face_recognition.face_distance(train_encodings, verification_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            predicted_id = train_ids[best_match_index]
            if predicted_id == verification_id:
                correct_predictions += 1

    verification_accuracy = correct_predictions / len(verification_ids)
    print(f"Verification accuracy: {verification_accuracy * 100:.2f}%")

def main():
    global video_capture, known_face_encodings, known_face_ids, true_labels, similarity_scores, train_data_folder

    # Define paths to folders containing face images
    train_data_folder = "classification_data/train_data"
    val_data_folder = "classification_data/val_data"
    test_data_folder = "classification_data/test_data"
    verification_data_folder = "verification_data"

    # Train the model
    print("Training the model...")
    best_model = train_model(train_data_folder, val_data_folder)

    # Test the model
    print("Testing the model...")
    test_model(test_data_folder, best_model)

    # Verify the model
    print("Verifying the model...")
    verify_model(verification_data_folder, best_model)

    # Load known faces and their encodings
    known_face_encodings, known_face_ids = best_model

    # Initialize webcam
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("Error: Webcam not found or unable to access.")
        return

    # Add a short delay to ensure the webcam is ready
    time.sleep(2)

    true_labels = []
    similarity_scores = []

    plt.ion()  # Turn on interactive mode for matplotlib
    fig, ax = plt.subplots()
    im = ax.imshow(np.zeros((480, 640, 3), dtype=np.uint8))
    ax.axis('off')

    # Create button for marking attendance
    mark_attendance_button_ax = plt.axes([0.4, 0.05, 0.2, 0.075])
    mark_attendance_button = Button(mark_attendance_button_ax, 'Mark Attendance')
    mark_attendance_button.on_clicked(mark_attendance)

    # Create button for quitting the program
    quit_button_ax = plt.axes([0.7, 0.05, 0.2, 0.075])
    quit_button = Button(quit_button_ax, 'Quit')
    quit_button.on_clicked(quit_program)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Error: Frame not captured.")
            break

        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        face_ids = []
        liveness_scores = []

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                student_id = known_face_ids[best_match_index]
            else:
                student_id = "Unknown"

            top, right, bottom, left = face_location
            face_image = frame[top:bottom, left:right]
            is_live = detect_liveness(face_image)
            face_ids.append(student_id)
            liveness_scores.append(is_live)
            true_labels.append(1 if student_id != "Unknown" else 0)
            similarity_scores.append(1 - face_distances[best_match_index])

        print("True labels:", true_labels)
        print("Similarity scores:", similarity_scores)

        output_frame = draw_face_boxes(frame, face_locations, face_ids, liveness_scores)
        output_frame_rgb = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
        im.set_data(output_frame_rgb)
        plt.draw()
        plt.pause(0.001)

    video_capture.release()
    plt.ioff()
    plt.close()

    evaluate_system(true_labels, similarity_scores)

if __name__ == "__main__":
    main()
