import cv2
#changer le numero pour une camera sur l<ordinateur
#cap = cv2.VideoCapture(1)

#changer l'adresse par l'adresse de la cameraa ip
cap = cv2.VideoCapture('http://10.1.3.142:4747/mjpegfeed?640x480')
# Defenir le codec pour la video
#choisir le type en argument de videowriter
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('video.avi', fourcc, 20.0, (640, 480))
n=0
while (cap.isOpened()):
    ret, frame = cap.read()
    #a chaque boucle, une image au nom name+n est enregistree
    name = "capture/name"+str(n)+".jpg"
    n+=1
    cv2.imwrite(name, frame)
    if ret == True:
        frame = cv2.flip(frame, 0)
        #on ajoute limage courante dans la video
        out.write(frame)
        #on l'affiche en temps reel
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# on detruit tout a la fin
cap.release()
out.release()
cv2.destroyAllWindows()
