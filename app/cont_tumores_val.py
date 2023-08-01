import torch
# Mi libreria:
from processLIDC3 import Patient
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import random
import argparse
import os
import cv2
from datetime import datetime



def contar_tumores_imagen(mask_b, iou_threshold=0.3):
    _, binary_image_b = cv2.threshold(mask_b, 100, 255, cv2.THRESH_BINARY)
    binary_image_b = cv2.convertScaleAbs(binary_image_b)
    contours_b, _ = cv2.findContours(binary_image_b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return len(contours_b)


def get_confusion_matrix2(patient_list):
    n_tumors = 0
    for patient_id in tqdm(patient_list):
        print('patient_id', patient_id)
        patient = Patient(patient_id)
        patient.scale()
        images, mask = patient.get_tensors(scaled = True)
        n_slices = mask.shape[0]
        mask = mask.cpu().detach().numpy()
        mask = mask.astype('int8')*255

        for i in range(n_slices):
            
            if np.all(mask[i,0] == 0):
                continue
            print(f'{patient_id}  slices {i}/{n_slices} Hay tumor.')

            n_tumor = contar_tumores_imagen(mask[i,0])
            print('\t +', n_tumor)
            n_tumors+= n_tumor
        print(n_tumors)
            # print('confusion_matriz_total', confusion_matrix_total)
    return n_tumors





if __name__ == "__main__":
    random.seed(123)

    
    print('Buscando los pacientes...', flush= True)
    patients = os.listdir('/home/abel/lidc-dataset/TCIA_LIDC-IDRI_20200921/LIDC-IDRI/')
    archivo = open('./failed_patients.txt', 'r')  # Reemplaza 'nombre_archivo.txt' por el nombre de tu archivo
    failed_patients = []

    for linea in archivo:
        linea = linea.strip()  # Elimina los espacios en blanco al principio y al final de la línea
        failed_patients.append(linea)
    archivo.close()
    patients = [pat for pat in patients if not pat=='LICENSE' and pat not in failed_patients]
    n_val = int(len(patients) * 0.1)
    if True:
        # Seleciona aleatoriamente el 30% de los patients
        patients_list = random.sample(patients, n_val)
        print(patients_list)
    else:
        val_patients = random.sample(patients, n_val)
        patients_list = [nombre for nombre in patients if nombre not in val_patients]
        
    print('Cargando el modelo...', flush=True)
    get_confusion_matrix2(patients_list)


    
    