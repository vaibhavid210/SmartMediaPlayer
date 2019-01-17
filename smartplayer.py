import vlc, easygui
import cv2



face_cascade = cv2.CascadeClassifier('/home/vaibhavi/opencv/opencv-3.2.0/data/haarcascades/haarcascade_frontalface_default.xml')

media = easygui.fileopenbox(title="Choose media to open")
player = vlc.MediaPlayer(media)




while True:
    choice = easygui.buttonbox(title="Smart Media Player",msg="Press Play to start",choices=["Play","Pause","Stop","New","Smart"])
    print(choice)
    if choice == "Play":
        player.play()
    elif choice == "Pause":
        player.pause()
    elif choice == "Stop":
        player.stop()
    elif choice == "New":
        media = easygui.fileopenbox(title="Choose media to open")
        player = vlc.MediaPlayer(media)
  
    elif choice == "Smart":
      cap = cv2.VideoCapture(0)
      player_actiqve = 0
      player_paused = 0

      player.play()
      player_active = 1 
      player_paused = 0
      
 

      while True:
	ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	for (x,y,w,h) in faces:
             cv2.rectangle(frame, (x, y), (x+w, y+h), (0 , 0, 255), 2) 

    	if len(faces) < 1 and (player_paused == 0 and player_active == 1):
      		player.pause()
      		player_active = 0
    
      		player_paused = 1
      		print ('Video Paused')


    	elif len(faces)>0 and (player_active == 0 and player_paused == 1):
      		player.play()
      		player_active = 1 
	        player_paused = 0
	        print ('Video Playing')
      
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame,"press q to stop",(0,130), font, 1, (255,255,155))
        cv2.imshow('frame', frame)
	
	#cv2.putText(frame,"press q to stop")

    	if cv2.waitKey(1) & 0xFF == ord('q'):
        	player.stop()
		break
      cap.release()
      cv2.destroyAllWindows()
    else:
      break
