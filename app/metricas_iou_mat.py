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

def calculate_iou(contour_a, contour_b):

    # Rellena el contorno con un valor blanco (255) en la imagen en blanco
    mascara_a = cv2.drawContours(np.zeros((512,512)), [contour_a], -1, (255), thickness=cv2.FILLED)
    mascara_b = cv2.drawContours(np.zeros((512,512)), [contour_b], -1, (255), thickness=cv2.FILLED)
    mascara_c = mascara_a*mascara_b
    interseccion = np.count_nonzero(mascara_c > 0)
    area_a = np.count_nonzero(mascara_a > 0)
    area_b = np.count_nonzero(mascara_b > 0)
    union = area_a + area_b - interseccion
    if  union == 0:
        if interseccion != 0:
            print('no deberias ver este mensaje: union=0 pero interseccion!=0 ¿?¿?¿?')
        iou = 999999999999
    else:
        iou = interseccion / union
    # print(f'area_a: {area_a}, area_b: {area_b}, interseccion, {interseccion}')
    # print('iou', iou)
    # fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    # axs[0].imshow(mascara_a, cmap='gray')
    # axs[1].imshow(mascara_b, cmap='gray')
    # axs[2].imshow(mascara_c, cmap='gray')
    # plt.show()

    return iou


def calculate_confusion_matrix(mask_a, mask_b, iou_threshold=0.3):
    _, binary_image_b = cv2.threshold(mask_b, 100, 255, cv2.THRESH_BINARY)
    binary_image_b = cv2.convertScaleAbs(binary_image_b)
    _, binary_image_a = cv2.threshold(mask_a, 100, 255, cv2.THRESH_BINARY)
    binary_image_a = cv2.convertScaleAbs(binary_image_a)
    contours_a, _ = cv2.findContours(binary_image_a, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_b, _ = cv2.findContours(binary_image_b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    true_positive = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0
    # print(contours_a)
    if len(contours_a)==0:
        false_negative += len(contours_b)
        return {
        'TP': true_positive,
        'TN': true_negative,
        'FP': false_positive,
        'FN': false_negative
    }
    if len(contours_b)==0:
        false_positive += len(contours_a)
        return {
        'TP': true_positive,
        'TN': true_negative,
        'FP': false_positive,
        'FN': false_negative
    }
    
    for contour_a in contours_a:
        iou_max = 0
        for contour_b in contours_b:
            iou = calculate_iou(contour_a, contour_b)
            if iou > iou_max:
                iou_max = iou

        if iou_max >= iou_threshold:
            true_positive += 1
            # print('sumamos 1 a true_positive')
        else:
            false_positive += 1
            # print('sumamos 1 a false_positive')

    #     print({
    #     'TP': true_positive,
    #     'TN': true_negative,
    #     'FP': false_positive,
    #     'FN': false_negative
    # })

    for contour_b in contours_b:
        iou_max = 0
        for contour_a in contours_a:
            iou = calculate_iou(contour_b, contour_a)
            if iou > iou_max:
                iou_max = iou

        if iou_max < iou_threshold:
            false_negative += 1
            # print('sumamos una a false_negative')
    #     print({
    #     'TP': true_positive,
    #     'TN': true_negative,
    #     'FP': false_positive,
    #     'FN': false_negative
    # })

    true_negative = np.count_nonzero(cv2.bitwise_not(mask_a) & cv2.bitwise_not(mask_b))

    confusion_matrix = {
        'TP': true_positive,
        'TN': 0, #true_negative,
        'FP': false_positive,
        'FN': false_negative
    }
    # print(confusion_matrix)
    if true_positive+false_negative != len(contours_b):
        print('WARNING: true_positive+false_negative != len(contours_b)',true_positive, false_negative, len(contours_b))
    return confusion_matrix


def sumar_diccionarios(dic1, dic2):
    for key, value in dic2.items():
        if key in dic1:
            dic1[key] += value
        else:
            dic1[key] = value
    return dic1


def get_confusion_matrix2(patient_list, threshold=0.4):
    confusion_matrix_total = {
        'TP': 0,
        'TN': 0,
        'FP': 0,
        'FN': 0
    }
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
            prediccion = patient.predict(model, slices=(i,), scaled=True, gpu = True)
            prediccion = np.where(prediccion >= threshold, 1, 0)[0,0,:,:]
            prediccion = prediccion.astype('int8')*255
            confusion_matrix_ = calculate_confusion_matrix(prediccion, mask[i,0], iou_threshold=0.3)
            confusion_matrix_total = sumar_diccionarios(confusion_matrix_total, confusion_matrix_)
            # print('confusion_matriz_total', confusion_matrix_total)
    return confusion_matrix_total


def plot_confusion_matrix(confusion_dict, save= './',show=False, threshold=999):
    labels = list(confusion_dict.keys())
    confusion_matrix = [[confusion_dict['TP'], confusion_dict['FP']],
                        [confusion_dict['FN'], confusion_dict['TN']]]

    # plt.figure(figsize=(6, 4))
    fig, ax = plt.subplots()

    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Reds", xticklabels=('label: nodulo', 'label: no_nodulo'), yticklabels=('pred: nodulo', 'pred: no_nodulo'))
    ax.set_xlabel("Label")
    ax.set_ylabel("Prediccion")
    ax.set_title(f"Confusion Matrix: IoU. threshold= {threshold}, iou_threshold = 0.1")
    if show:
        # Mostrar la figura
        plt.show()
    if save is not None:
        fecha_actual = datetime.now()
        # Formatear la fecha en el formato deseado (por ejemplo, "año_mes_dia_hora_minuto_segundo")
        fecha = fecha_actual.strftime("%Y-%m-%d_%H-%M-%S")
        path = save+f'confusion_matrix_iou_{fecha}.png'
        plt.savefig(path, dpi=300)
        print(f'figura guardada {path}')



if __name__ == "__main__":
    random.seed(123)
    parser = argparse.ArgumentParser()
    # Agregar los argumentos necesarios
    parser.add_argument('--val', action='store_true', default = True)
    parser.add_argument('--model', type=str, default='./default_model.pt')
    parser.add_argument('--save', type=str, default='./')
    parser.add_argument('--path2dataset', type=str, default='../../manifest-1675801116903/LIDC-IDRI/')
    parser.add_argument('--valsplit', type=float, default=0.1)
    parser.add_argument('--threshold', type=float, default=0.5)
    # parser.add_argument('--batch', type=float, default=5)
    args = parser.parse_args()
    
    print('Buscando los pacientes...', flush= True)
    patients = os.listdir(args.path2dataset)
    archivo = open('./failed_patients.txt', 'r')  # Reemplaza 'nombre_archivo.txt' por el nombre de tu archivo
    failed_patients = []

    for linea in archivo:
        linea = linea.strip()  # Elimina los espacios en blanco al principio y al final de la línea
        failed_patients.append(linea)
    archivo.close()
    patients = [pat for pat in patients if not pat=='LICENSE' and pat not in failed_patients]
    n_val = int(len(patients) * args.valsplit)
    if args.val:
        # Seleciona aleatoriamente el 30% de los patients
        patients_list = random.sample(patients, n_val)
    else:
        val_patients = random.sample(patients, n_val)
        patients_list = [nombre for nombre in patients if nombre not in val_patients]
        
    print('Cargando el modelo...', flush=True)
    model = torch.jit.load(args.model)
    model.to('cuda')
    model.eval()
    matriz = get_confusion_matrix2(patients_list, threshold=args.threshold)
    print(matriz)
    # cm = get_confusion_matrix_list(patients_list, model, args.threshold, args.batch)
    plot_confusion_matrix(matriz, save=args.save, show=False, threshold = args.threshold)
    
    
    