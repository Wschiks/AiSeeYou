import cv2
import pygame
import mediapipe as mp
import time

# Initialiseer pygame mixer
pygame.mixer.init()

def waarschuwing_geven():
    pygame.mixer.music.load("wegKijk.mp3")  # Zorg dat dit bestand in dezelfde map staat
    pygame.mixer.music.play()

def compliment_geven():
    pygame.mixer.music.load("mooieOgen.mp3")  # Zorg dat dit bestand in dezelfde map staat
    pygame.mixer.music.play()

# Mediapipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=False)  # Minder zwaar voor performance
mp_drawing = mp.solutions.drawing_utils

# Webcam starten
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Timers
laatste_waarschuwing = 0
laatste_compliment = time.time()
waarschuwing_interval = 5        # seconden
compliment_interval = 300      # 5 minuten
gezicht_continu_aanwezig = True  # Volgen of gezicht non-stop gezien wordt

print("Druk op Q om te stoppen.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Spiegel beeld en converteer naar RGB
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Verwerk met Mediapipe
    results = face_mesh.process(rgb_frame)

    # Detecteer gezicht
    gezicht_aanwezig = results.multi_face_landmarks is not None

    huidige_tijd = time.time()

    # Wegkijkgedrag
    if not gezicht_aanwezig:
        if huidige_tijd - laatste_waarschuwing > waarschuwing_interval:
            waarschuwing_geven()
            laatste_waarschuwing = huidige_tijd
        gezicht_continu_aanwezig = False  # Onderbreek streak

    else:
        # Alleen als gezicht onafgebroken aanwezig is sinds laatste compliment
        if gezicht_continu_aanwezig and huidige_tijd - laatste_compliment > compliment_interval:
            compliment_geven()
            laatste_compliment = huidige_tijd
        elif not gezicht_continu_aanwezig:
            # Reset streak bij eerste keer weer detectie
            gezicht_continu_aanwezig = True
            laatste_compliment = huidige_tijd

        # Teken gezichtsmasker
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)

    cv2.imshow('Oogscanner', frame)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
