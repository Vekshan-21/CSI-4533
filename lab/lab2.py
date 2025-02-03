import cv2
import numpy as np
import os

def detect_person(image):
    """Détecte la personne dans l'image à l'aide de HOG+SVM."""
    if image is None:
        print("Erreur : image non chargée.")
        return None
    
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    rects, _ = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)
    if len(rects) > 0:
        x, y, w, h = rects[0]  # Prendre la première détection
        return image[y:y+h, x:x+w]
    return None

def compute_histogram(image, bins=(8, 8, 8)):
    """Calcule l'histogramme normalisé d'une image."""
    if image is None:
        return None
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def compare_histograms(hist1, hist2, method=cv2.HISTCMP_CORREL):
    """Compare deux histogrammes avec la méthode spécifiée."""
    if hist1 is None or hist2 is None:
        return 0
    return cv2.compareHist(hist1, hist2, method)

def process_images(image_folder, reference_image_path):
    """Parcourt les images du dossier et compare leurs histogrammes avec celui de la référence."""
    if not os.path.exists(reference_image_path):
        print(f"Erreur : L'image de référence '{reference_image_path}' est introuvable.")
        return []
    
    reference_image = cv2.imread(reference_image_path)
    if reference_image is None:
        print(f"Erreur : impossible de charger l'image de référence {reference_image_path}")
        return []
    
    reference_person = detect_person(reference_image)
    if reference_person is None:
        print("Impossible de détecter la personne dans l'image de référence.")
        return []
    
    reference_hist = compute_histogram(reference_person)
    matches = []
    
    if not os.path.exists(image_folder):
        print(f"Erreur : Le dossier '{image_folder}' n'existe pas.")
        return []
    
    image_list = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if not image_list:
        print(f"Aucune image trouvée dans '{image_folder}'.")
        return []
    
    for image_name in image_list:
        image_path = os.path.join(image_folder, image_name)
        print(f"Lecture de : {image_path}")
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"Erreur : impossible de charger l'image {image_path}")
            continue
        
        detected_person = detect_person(image)
        if detected_person is not None:
            image_hist = compute_histogram(detected_person)
            similarity = compare_histograms(reference_hist, image_hist)
            
            if similarity > 0.8:  # Seuil ajustable
                matches.append(image_path)
                print(f"Correspondance trouvée : {image_path} (sim = {similarity:.2f})")
    
    return matches

# Exemple d'utilisation
image_folder_path = "C:\\Users\\Owner\\Downloads\\2022-12-15_all_masks\\2022-12-15_lunch_3_masks\\masks\\2022-12-15\\lunch\\3\\cam1"
target_image_path = "C:\\Users\\Owner\\Downloads\\2022-12-15_all_masks\\suspect.png"  # Image cible

matching_images = process_images(image_folder_path, target_image_path)
