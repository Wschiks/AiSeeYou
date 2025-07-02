import cv2
import pygame
import mediapipe as mp
import time
from plyer import notification
import numpy as np

# Initialiseer pygame mixer
pygame.mixer.init()

def disco_scherm():
    disco_venster = 'doom'
    cv2.namedWindow(disco_venster, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(disco_venster, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    pygame.mixer.music.load("doom.mp3")
    pygame.mixer.music.play()
    start_tijd = time.time()
    disco_duur = 10

    while time.time() - start_tijd < disco_duur:
        rood_scherm = np.zeros((500, 800, 3), dtype=np.uint8)
        rood_scherm[:, :, 2] = 255  # Alleen rood kanaal op max

        elapsed = time.time() - start_tijd
        aftel_waarde = max(0, int(disco_duur - elapsed))  # zorgt dat het niet negatief wordt

        # Tekst in 2 regels
        tekst1 = "JE KEEK WEG NU"
        tekst2 = f"HEB JE Timeout {aftel_waarde}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2
        kleur = (255, 255, 255)  # wit
        dikte = 3

        positie1 = (50, 220)
        positie2 = (50, 280)

        cv2.putText(rood_scherm, tekst1, positie1, font, font_scale, kleur, dikte, cv2.LINE_AA)
        cv2.putText(rood_scherm, tekst2, positie2, font, font_scale, kleur, dikte, cv2.LINE_AA)

        cv2.imshow(disco_venster, rood_scherm)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    pygame.mixer.music.stop()
    cv2.destroyWindow(disco_venster)


def waarschuwing_geven():
    pygame.mixer.music.load("wegKijk.mp3")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.wait(100)  # wacht eventjes, zodat CPU niet 100% draait


def compliment_geven():
    pygame.mixer.music.load("mooieOgen.mp3")  # Zorg dat dit bestand in dezelfde map staat
    pygame.mixer.music.play()

def toon_notificatie(titel, bericht):
    notification.notify(
        title=titel,
        message=bericht,
        timeout=5,  # seconden
        app_name='Oogscanner',
        app_icon='job.ico'  # Of zet hier een .ico bestand voor een eigen icoontje
    )
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
            disco_scherm()
            toon_notificatie("Opletten!", "JE KEEK WEG!")
            laatste_waarschuwing = huidige_tijd
        gezicht_continu_aanwezig = False  # Onderbreek streak

    else:
        # Alleen als gezicht onafgebroken aanwezig is sinds laatste compliment
        if gezicht_continu_aanwezig and huidige_tijd - laatste_compliment > compliment_interval:
            compliment_geven()
            toon_notificatie("Goed bezig!", "Je hebt al 5 minuten goed gekeken 👀")
            laatste_compliment = huidige_tijd
        elif not gezicht_continu_aanwezig:
            # Reset streak bij eerste keer weer detectie
            gezicht_continu_aanwezig = True
            laatste_compliment = huidige_tijd

        # Teken gezichtsmasker
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
