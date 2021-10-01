import tkinter as tk
from PIL import Image,ImageTk
import cv2
import imutils
print("libreria Leida...")


ventana = tk.Tk()
ventana.geometry("1000x680+200+10")
ventana.title("Terrones Digital")
ventana.resizable(width=False,height=False)
fondo=tk.PhotoImage(file="Image/Mis Amiguitos.png")
fondo1= tk.Label(ventana,image=fondo).place(x=0,y=0,relwidth=1,relheight=1)

#Funciones
video=None

def video_stream():
    global video
    video=cv2.VideoCapture(0)
    iniciar()

def iniciar():
    global video
    ret,frame=video.read()
    if ret == True:
        etiq_de_video.place(x=45,y=200)
        frame=imutils.resize(frame,width=570)
        frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        img=Image.fromarray(frame)
        image=ImageTk.PhotoImage(image=img)
        etiq_de_video.configure(image=image)
        etiq_de_video.image=image
        etiq_de_video.after(10,iniciar)

    else:
        etiq_de_video.image=""
        video.release()

def quitar():
    global video
    etiq_de_video.place_forget()
    video.release()



#Colores
fondo_boton="#C7D0D8"






#Botones
boton=tk.Button(ventana,text="Registrar",bg=fondo_boton,relief="flat",
                cursor="hand2",command=video_stream,width=13,height=2,font=("Calisto MT",12,"bold"))
boton.place(x=720, y=310)

boton2=tk.Button(ventana,text="Entrenar",bg=fondo_boton,relief="flat",
                cursor="hand2",command=quitar,width=13,height=2,font=("Calisto MT",12,"bold"))
boton2.place(x=720, y=420)


boton3=tk.Button(ventana,text="Reconocer",bg=fondo_boton,relief="flat",
                cursor="hand2",width=13,height=2,font=("Calisto MT",12,"bold"))
boton3.place(x=720, y=530)


#Etiqueta

etiq_de_video=tk.Label(ventana,bg="red")
etiq_de_video.place(x=45,y=200)


ventana.mainloop()