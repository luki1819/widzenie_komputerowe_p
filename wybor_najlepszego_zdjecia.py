import cv2
import os
import numpy as np


def ostrosc(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()  # wariancja pikseli im większa tym lepiej

def calculate_contrast(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray.std()


def calculate_snr(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    signal = np.mean(gray)
    noise = np.std(gray)
    if noise == 0:  # uniknięcie dzielenia przez 0
        return float('inf') # im więcej tym mniej szumów
    snr = signal / noise
    return snr

def lisc_powierzchnia(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    green_area = cv2.countNonZero(mask)
    total_area = mask.size
    return green_area / total_area


def normalize_column(data, column_index):

    # Pobranie wartości z określonej kolumny
    column = [row[column_index] for row in data]
    min_val = min(column)
    max_val = max(column)

    if max_val == min_val:
        # Obsługa przypadku, gdy wszystkie wartości w kolumnie są takie same
        return [0.5 for _ in column]

    # Normalizacja wartości
    normalized_column = [(value - min_val) / (max_val - min_val) for value in column]
    return normalized_column

def normalize_results(results):
    #(1 = sharp, 2 = area, 3 = kontrast, 4 = snr)
    num_columns = len(results[0])
    normalized_data = [list(row) for row in results]
    for column_index in range(1, num_columns-1):
        normalized_column = normalize_column(results, column_index)
        for i, value in enumerate(normalized_column):
            normalized_data[i][column_index] = value
    return [tuple(row) for row in normalized_data]

def apply_canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    v = np.median(gray)
    sigma = 0.33
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))

    # Zastosowanie filtru Canny'ego
    edges = cv2.Canny(gray, lower , upper)
    return edges


folder_path = r"klatki_posortowane\norway maple"
output_path = "best_image.jpg"


results = []

for filename in os.listdir(folder_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        file_path = os.path.join(folder_path, filename)
        image = cv2.imread(file_path)
        if image is not None:

            # Obliczam miary jakości
            sharp = ostrosc(image)
            area = lisc_powierzchnia(image)
            kontrast = calculate_contrast(image)
            snr = calculate_snr(image)

            results.append((file_path, sharp, area, kontrast, snr))

waga_ostrosc = 1
waga_powierzchnia = 0
waga_kontrast = 10
waga_snr = 1

results = [
    (
        file_path,
        sharp,
        area,
        kontrast,
        snr,
    )
    for file_path, sharp, area, kontrast, snr in results
]
normalized_results = normalize_results(results)
for i, result in enumerate(normalized_results):
    weighted_score = (result[1] * waga_ostrosc +
                      result[2] * waga_powierzchnia +
                      result[3] * waga_kontrast +
                      result[4] * waga_snr)

    normalized_results[i] = result + (weighted_score,)

normalized_results.sort(key=lambda x: x[5], reverse=True)
best_image_path = normalized_results[0][0] if results else None

if best_image_path:
    print(f"Najlepsze zdjęcie: {best_image_path}")
    # Opcjonalnie zapisz wybrane zdjęcie
    best_image = cv2.imread(best_image_path)
    cv2.imwrite(output_path, best_image)

    edges = apply_canny(best_image)

    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    image_with_edges = cv2.addWeighted(best_image, 0.8, edges_colored, 1, 0)

    cv2.namedWindow("Best Image with Edges", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Best Image with Edges", best_image.shape[1], best_image.shape[0])
    cv2.imshow("Best Image with Edges", image_with_edges)

    cv2.waitKey(0)  # Czekaj na naciśnięcie dowolnego klawisza
    cv2.destroyAllWindows()
else:
    print("Brak zdjęć do analizy!")

