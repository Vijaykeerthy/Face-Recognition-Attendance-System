import cv2
import os
from flask import Flask, request, render_template
from datetime import date, datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib

# VARIABLE
MESSAGE = "WELCOME! Instruction: To register your attendance, kindly click on 'a' on the keyboard"

# Defining Flask App
app = Flask(__name__)

# Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

# Initializing VideoCapture object to access the WebCam
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(1)  # Trying an external webcam

# If the external webcam fails, fallback to the default webcam
if not cap.isOpened():
    cap = cv2.VideoCapture(0)

# If these directories don't exist, create them
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')

# Check and create attendance file if it doesn't exist
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Time')

# Function to get the total number of registered users
def totalreg():
    return len(os.listdir('static/faces'))

# Function to extract faces from an image
def extract_faces(img):
    if img is not None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.3, 5)
        return face_points
    else:
        return []

# Function to identify face using the pre-trained ML model
def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)

# Function to train the model on all faces available in faces folder
def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')

# Extract attendance information from today's attendance file
def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    l = len(df)
    return names, rolls, times, l

# Add attendance for a specific user
def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")

    # Construct the file path for today's attendance CSV
    attendance_file = f'Attendance/Attendance-{datetoday}.csv'
    
    try:
        # Read the current attendance CSV
        df = pd.read_csv(attendance_file)

        # Check if the user has already marked attendance
        if str(userid) not in list(df['Roll']):
            # Append new attendance entry
            with open(attendance_file, 'a') as f:
                f.write(f'\n{username},{userid},{current_time}')
            print(f"Attendance marked for {username} at {current_time}.")
            return True  # Attendance successfully marked
        else:
            print(f"{username} has already marked attendance for the day.")
            return False  # Attendance already marked
    except FileNotFoundError:
        print("Attendance file not found. Please check if it has been created.")
        return False  # Attendance file not found
    except Exception as e:
        print(f"An error occurred: {e}")
        return False  # Some error occurred

# Main page route
@app.route('/')
def home():
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess=MESSAGE)

# Route for starting attendance capture
@app.route('/start', methods=['GET'])
def start():
    ATTENDANCE_MARKED = False
    attendance_success = False
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        names, rolls, times, l = extract_attendance()
        MESSAGE = 'No face model found. Please register first.'
        return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess=MESSAGE)
    
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))[0]
            cv2.putText(frame, f'{identified_person}', (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2)
            if cv2.waitKey(1) == ord('a'):
                attendance_success = add_attendance(identified_person)
                if attendance_success:
                    current_time_ = datetime.now().strftime("%H:%M:%S")
                    print(f"Attendance marked for {identified_person} at {current_time_}")
                else:
                    print(f"Failed to mark attendance for {identified_person}.")
                ATTENDANCE_MARKED = True
                break
        
        if ATTENDANCE_MARKED:
            break
        
        cv2.imshow('Attendance Check - Press "q" to exit', frame)
        
        if cv2.waitKey(1) == ord('q'):
            cap.release()
            cv2.destroyAllWindows()  # Check for 'q' key press to exit
            break
    cap.release()
    cv2.destroyAllWindows()
    cap.release()
    cv2.destroyAllWindows()
    names, rolls, times, l = extract_attendance()
    MESSAGE = 'Attendance taken successfully' if attendance_success else 'Attendance not marked!'
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess=MESSAGE)

# Route to add new user
@app.route('/add', methods=['GET', 'POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = 'static/faces/' + newusername + '_' + str(newuserid)
    
    # Check if the user folder exists, if not, create it
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    
    cap = cv2.VideoCapture(0)
    i, j = 0, 0
    while True:
        ret, frame = cap.read()
        faces = extract_faces(frame)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
            cv2.putText(frame, f'Images Captured: {i}/50', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
            if i < 50:  # Capture 50 images
                name = newusername + '_' + str(i) + '.jpg'
                cv2.imwrite(userimagefolder + '/' + name, frame[y:y+h, x:x+w])
                i += 1
            j += 1
        if i >= 50 or j >= 500:  # Stop capturing after 50 images or after 500 iterations
            break
        
        cv2.imshow('Adding New User - Press "q" to exit', frame)
        if cv2.waitKey(1) == ord('q'):
            cap.release()
            cv2.destroyAllWindows()  # Check for 'q' key press to exit
            break
    
    # Train the model with the new user's images
    train_model()
    
    # Add user details to the attendance CSV file
    with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
        f.write(f'\n{newusername},{newuserid},')
    
    cap.release()
    cv2.destroyAllWindows()  # Release the video capture

    # Update the home page with the new user info
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess=MESSAGE)

if __name__ == '__main__':
    app.run(debug=True)
