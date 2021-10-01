# Patentado Erick 2015

# Imports
import cv2
from tkinter import *
from PIL import Image, ImageTk
import os
import imutils
import numpy as np
import pyttsx3


class LimitedList:

    def __init__(self, limit):
        self.limit = limit
        self.list = [None for index in range(0, self.limit)]
        self.current_index = 0

    def add(self, value):
        self.check_index()
        self.list[self.current_index] = value
        self.current_index += 1

    def check_index(self):
        if self.current_index == self.limit:
            self.current_index = 0
            self.list = [None for index in range(0, self.limit)]

    def check_value(self, value):
        if self.list.count(value) == self.limit:
            return True
        else:
            return False


class FaceRecognizer:
    def __init__(self):
        # creating Tkinter Main Window
        self.root = Tk()
        self.root.geometry("1000x680+200+10")
        self.root.title("Guarderia - Amiguitos del Rey")
        self.root.resizable(width=False, height=False)

        # insert background image
        self.background = PhotoImage(file="Image/Mis Amiguitos.png")
        self.main_label = Label(self.root, image=self.background).place(x=0, y=0, relwidth=1, relheight=1)

        # button exit
        button = Button(self.root, text="Refresh", command=self.exit)
        button.pack()
        button.place(x=730, y=220)

        # button recognition
        self.text = Entry(self.root, width=30, bg='White')
        self.text.pack(pady=10)
        self.text.place(x=700, y=260)

        # creating buttons

        fondo_boton = "#C7D0D8"
        button_capture = Button(self.root, text="Add new face", command=self.face_capture, bg=fondo_boton,
                                relief="flat", cursor="hand2",width=13,height=2,font=("Calisto MT", 12, "bold"))
        button_capture.place(x=720, y=310)
        button_capture.pack()

        # button train
        button_train = Button(self.root, text="Train", bg=fondo_boton, relief="flat", command=self.train_recognizer,
                              cursor="hand2", width=13, height=2, font=("Calisto MT", 12, "bold"))
        button_train.place(x=720, y=420)
        button_train.pack()

        # button recognize
        button_recognition = Button(self.root, text="Recognize faces", command=self.recognize_faces, bg=fondo_boton,
                                    relief="flat", cursor="hand2", width=13, height=2, font=("Calisto MT", 12, "bold"))
        button_recognition.place(x=720, y=530)
        button_recognition.pack()

        video = Label(self.root)
        video.pack()

        # Getting video from webcam
        self.vid = cv2.VideoCapture(0)

        # Loop which display video on the label
        while (True):
            ret, frame = self.vid.read()  # Reads the video
            if ret == False: break
            frame = imutils.resize(frame, width=640)
            # Converting the video for Tkinter
            # cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)

            # cv2.namedWindow("video")
            # cv2.moveWindow("video", 150, 150)
            # cv2.imshow("video", frame)

            # img = Image.fromarray(cv2image)
            # imgtk = ImageTk.PhotoImage(image=img)

            # print(img)
            # Setting the image on the label
            # video.config(image=imgtk)

            self.root.update()  # Updates the Tkinter window

            k = cv2.waitKey(1)
            if k == 27:
                break
            self.root.update()
        self.root.mainloop()
        self.vid.release()
        cv2.destroyAllWindows()

    def face_capture(self):
        etiq_de_video = Label(self.root, bg="red")
        etiq_de_video.place(x=45, y=200)
        personName = self.text.get()
        dataPath = '/home/mars/Projects/python/face-detector/marco/ReconociminetoFacial/Data'
        personPath = dataPath + '/' + personName

        if not os.path.exists(personPath):
            print('Carpeta creada: ', personPath)
            os.makedirs(personPath)

        cap = cv2.VideoCapture(0)
        # cap = cv2.VideoCapture('Video.mp4')

        faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        count = 0

        while True:
            ret, frame = cap.read()
            if ret == False: break
            etiq_de_video.place(x=45, y=200)

            frame = imutils.resize(frame, width=640)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            auxFrame = frame.copy()

            faces = faceClassif.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                rostro = auxFrame[y:y + h, x:x + w]
                rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(personPath + '/rotro_{}.jpg'.format(count), rostro)
                count = count + 1
            # press ESC to end this

            # tkinter display
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
            image = ImageTk.PhotoImage(image=img)
            etiq_de_video.configure(image=image)
            etiq_de_video.image = image

            k = cv2.waitKey(1)
            if k == 27 or count >= 300:
                break

            self.root.update()  # Updates the Tkinter window

    def exit(self):
        self.vid.release()
        cv2.destroyAllWindows()

    def train_recognizer(self):
        dataPath = '/home/mars/Projects/python/face-detector/marco/ReconociminetoFacial/Data'
        peopleList = os.listdir(dataPath)
        print('Lista de personas: ', peopleList)

        labels = []
        facesData = []
        label = 0

        for nameDir in peopleList:
            personPath = dataPath + '/' + nameDir
            print('Leyendo las imágenes')

            for fileName in os.listdir(personPath):
                print('Rostros: ', nameDir + '/' + fileName)
                labels.append(label)
                facesData.append(cv2.imread(personPath + '/' + fileName, 0))
            # image = cv2.imread(personPath+'/'+fileName,0)
            # cv2.imshow('image',image)
            # cv2.waitKey(10)
            label = label + 1

        # print('labels= ',labels)
        # print('Número de etiquetas 0: ',np.count_nonzero(np.array(labels)==0))
        # print('Número de etiquetas 1: ',np.count_nonzero(np.array(labels)==1))

        # Métodos para entrenar el reconocedor
        # face_recognizer = cv2.face.EigenFaceRecognizer_create()
        # face_recognizer = cv2.face.FisherFaceRecognizer_create()
        face_recognizer = cv2.face.LBPHFaceRecognizer_create()

        # Entrenando el reconocedor de rostros
        print("Entrenando...")
        face_recognizer.train(facesData, np.array(labels))

        # Almacenando el modelo obtenido
        # face_recognizer.write('modeloEigenFace.xml')
        # face_recognizer.write('modeloFisherFace.xml')
        face_recognizer.write('modeloLBPHFace.xml')
        print("Modelo almacenado...")

    def recognize_faces(self):
        dataPath = '/home/mars/Projects/python/face-detector/marco/ReconociminetoFacial/Data'
        imagePaths = os.listdir(dataPath)
        print('imagePaths=', imagePaths)

        # face_recognizer = cv2.face.EigenFaceRecognizer_create()
        # face_recognizer = cv2.face.FisherFaceRecognizer_create()
        face_recognizer = cv2.face.LBPHFaceRecognizer_create()

        # Leyendo el modelo
        # face_recognizer.read('modeloEigenFace.xml')
        # face_recognizer.read('modeloFisherFace.xml')
        face_recognizer.read('modeloLBPHFace.xml')

        cap = cv2.VideoCapture(0)
        # cap = cv2.VideoCapture('Video.mp4')

        faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        identified_people = LimitedList(10)

        while True:
            ret, frame = cap.read()
            if ret == False: break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            auxFrame = gray.copy()

            faces = faceClassif.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                rostro = auxFrame[y:y + h, x:x + w]
                rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
                result = face_recognizer.predict(rostro)

                cv2.putText(frame, '{}'.format(result), (x, y - 5), 1, 1.3, (255, 255, 0), 1, cv2.LINE_AA)

                '''
                # EigenFaces
                if result[1] < 5700:
                    cv2.putText(frame,'{}'.format(imagePaths[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
                    cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
                else:
                    cv2.putText(frame,'Desconocido',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
                    cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
    
                # FisherFace
                if result[1] < 500:
                    cv2.putText(frame,'{}'.format(imagePaths[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
                    cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
                else:
                    cv2.putText(frame,'Desconocido',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
                    cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
                  '''
                # LBPHFace
                if result[1] < 70:
                    cv2.putText(frame, '{}'.format(imagePaths[result[0]]), (x, y - 25), 2, 1.1, (0, 255, 0), 1,
                                cv2.LINE_AA)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    identified_people.add(imagePaths[result[0]])
                    print(identified_people)
                    try:
                        if identified_people.check_value(imagePaths[result[0]]):
                            engine = pyttsx3.init()
                            engine.setProperty("voice", "spanish-latin-am")
                            engine.say("Papa de " + imagePaths[result[0]])
                            engine.runAndWait()
                    except IndexError:
                        pass
                else:
                    cv2.putText(frame, 'Desconocido', (x, y - 20), 2, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

            cv2.imshow('frame', frame)
            k = cv2.waitKey(1)
            if k == 27:
                break


face = FaceRecognizer()
