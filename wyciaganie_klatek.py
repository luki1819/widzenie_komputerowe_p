import os
import cv2

# Ścieżka zapisu
folder_zapisu = r'klatki z filmu'


if not os.path.exists(folder_zapisu):
    os.makedirs(folder_zapisu)

# Przetwarzanie wideo
count = 0
frame_interval = 100
cap = cv2.VideoCapture("trees.mp4")

if not cap.isOpened():
    print("Błąd: Nie udało się otworzyć pliku wideo.")
else:
    print("Plik wideo otwarty pomyślnie.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Koniec wideo lub błąd w odczycie klatki.")
        break

    if count % frame_interval == 0:
        time_in_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        time_in_seconds = time_in_ms / 1000
        hours = int(time_in_seconds // 3600)
        minutes = int((time_in_seconds % 3600) // 60)
        seconds = int(time_in_seconds % 60)
        milliseconds = int(time_in_ms % 1000)
        timestamp = f"{hours:02d}_{minutes:02d}_{seconds:02d}_{milliseconds:03d}"

        file_path = os.path.join(folder_zapisu, f"frame_{timestamp}.jpg")

        if frame is not None:
            success = cv2.imwrite(file_path, frame)
            if success:
                print(f"Klatka {count} zapisana jako {file_path}")
            else:
                print(f"Błąd: Nie udało się zapisać klatki {count}.")
    count += 1

cap.release()
cv2.destroyAllWindows()
