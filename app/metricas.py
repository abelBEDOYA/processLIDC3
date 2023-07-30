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
from datetime import datetime
import time

def get_confusion_matrix(id_patient, model, threshold = 0.5, batch = 10):
    cm = np.zeros((2,2))
    batch = int(batch)
    patient = Patient(id_patient)
    patient.scale()
    images, mask = patient.get_tensors(scaled = True)
    mask = mask.cpu().detach().numpy()[:,0,:,:]
    n_slices = mask.shape[0]
    # slices = (0, batch-1)
    prediccion = patient.predict(model, slices=(0,), scaled=True, gpu = True)
    # prediccion = np.where(prediccion >= threshold, 1, 0)[:,0,:,:]
    # masks_slices = mask[0:batch-1]
    # print(masks_slices.shape)
    # prediccion = np.array([])
    mask_label = mask[0]
    for i in tqdm(range(1,n_slices)):
        if np.all(mask[i] == 0):
            print('batch_saltado')
            continue

        # print(i+batch, n_slices)
        pred = patient.predict(model, slices=(i,), scaled=True, gpu = True)
        # print(pred.shape)
        pred_bin = np.where(pred >= threshold, 1, 0)[:,0,:,:]
        prediccion = np.concatenate((prediccion, pred_bin), axis=0)
        # print(f'{i}', prediccion.shape)
        mask_label = np.concatenate((mask_label, mask[i]), axis=0)
        print(mask_label.shape, prediccion.shape)

    # label = mask[slices[0]: slices[-1]+1].flatten()
    label = mask_label.flatten()
    prediccion = prediccion.flatten()
    
    print(label.shape, prediccion.shape)
    cm_ = confusion_matrix(label, prediccion, labels=(0,1))
    cm = cm + np.array(cm_)
    
    return cm


def get_confusion_matrix_list(id_patient, model, threshold = 0.5, batch = 10):
    cm = np.zeros((2,2))
    if isinstance(id_patient,str):
        print('haciendo inferencia del paciente {}'.format(id_patient))
        cm = get_confusion_matrix(id_patient, model, threshold =threshold, batch = batch)
        return cm
    else:
        print('haciendo inferencia del primer paciente...')
        cm = get_confusion_matrix(id_patient[0], model, threshold =threshold, batch = batch)
        for id in tqdm(id_patient[1:]):
            cm = cm + get_confusion_matrix(id, model, threshold =threshold, batch = batch)
            print(cm)
        return cm

def plotNsave(cm, save = None, show = True):
    fig, ax = plt.subplots()

    # Crear mapa de calor utilizando seaborn
    sns.heatmap(np.int32(cm), annot=True, fmt="d", cmap="Reds", cbar=False, square=True)  # , xticklabels=labels, yticklabels=labels)

    # Añadir etiquetas a los ejes
    ax.set_xlabel("Etiqueta Predicha")
    ax.set_ylabel("Etiqueta Verdadera")

    # Añadir título
    ax.set_title("Confusion Matrix, area")
    if show:
        # Mostrar la figura
        plt.show()
    if save is not None:
        fecha_actual = datetime.now()
        # Formatear la fecha en el formato deseado (por ejemplo, "año_mes_dia_hora_minuto_segundo")
        fecha = fecha_actual.strftime("%Y-%m-%d_%H-%M-%S")
        path = save+f'confusion_matrix_{fecha}.png'
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
    parser.add_argument('--batch', type=float, default=5)
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
    cm = get_confusion_matrix_list(patients_list, model, args.threshold, args.batch)
    plotNsave(cm, args.save, show=False)
    
    
    
