import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
import random
import argparse
import time
from collections import deque
# Mi libreria:
from processLIDC3 import Patient
import datetime
import cv2
from unet import UNet
import csv
random.seed(123)


def train_val_split(patients_list, val_split):
    """TOma la lsita de pacientes list(str) y hace la separacion
    en train y validation segun la proporcion indicada en val_split.
    Args:
        patients_list (list(str)): lista on los id de los pacientes
        val_split (float): proporcion destinada a validation
    Returns:
        train_patients, val_val_patients (list, list): lista de
            nombres de pacientes para train y validation
        """
    n_val = int(len(patients_list) * val_split)

    # Seleciona aleatoriamente el 30% de los patients
    val_patients = random.sample(patients_list, n_val)

    # Crea una lista con los patient que no fueron seleccionados
    train_patients = [nombre for nombre in patients_list if nombre not in val_patients]
    return train_patients, val_patients


def get_val_loss(model, val_patients, batch_size=4, loss_type = 1):
    if len(val_patients)==0:
        return 0

    loss_batch = np.array([])
    batch_loss_history = np.array([])
    loss_patient = np.array([])
    print('Realizando validacion...')
    tqdm_val_patients = tqdm(val_patients,leave=False, position=0)
    for id_pat in tqdm_val_patients:
        time.sleep(1)
        
        tqdm_val_patients.set_description('{}. {}. Progreso val.:'.format(get_tiempo(),id_pat))
        
        # Cargamos datos de un paciente:
        patient = Patient(id_pat)
        
        # Escalamos:
        patient.scale()
        
        # Obtenemos los tensores:
        imgs, mask = patient.get_tensors(scaled=True)
        if torch.cuda.is_available():
            device = torch.device('cuda')
            imgs, mask = imgs.to(device), mask.to(device)
        # Preparamos tensores para recorrerlos:
        # primera = 2
        # ultima = 10
        # dataset = TensorDataset(imgs[primera:ultima], mask[primera:ultima])
        dataset = TensorDataset(imgs, mask)
        

        train_loader = DataLoader(dataset,batch_size=batch_size, shuffle=True)
        loss_batch = np.array([])
        for batch_idx, (data, target) in enumerate(train_loader):
            if torch.mean(target)==0:
                # print('es 0')
                continue
            # # Forward pass
            output = model(data)
            # Calcular pérdida
            loss, _, _ = loss_function(output, target, loss_type = loss_type)
            # print('loss', loss, 'torch.mean(target):', torch.mean(target))
            
            loss_batch = np.append(loss_batch, loss.item())
            batch_loss_history = np.append(batch_loss_history, loss.item())
        if len(loss_batch)==0:
                print('Paciente sin tumor (es posible, pacient=988 por ejemplo). Paciente saltado')
                continue
        loss_patient = np.append(loss_patient, np.mean(np.array(loss_batch)))
        # print('id_patient', id_pat, 'loss_patient', loss_patient)
    val_mean_loss = np.mean(loss_patient)
    return val_mean_loss


def plot(data, show=False, path_save=None, name_plot='loss_plot', loss_type=1):
    epoch_loss_history = data['epoch_loss_history']
    batch_loss_history = data['batch_loss_history']
    patient_loss_history = data['patient_loss_history']
    epoch_val_loss_history = data['epoch_val_loss_history']
    iou_history = data['iou_history']
    wbce_history =data['wbce_history']
    n_epochs = len(epoch_loss_history)
    if loss_type==3:
        plt.plot(np.linspace(1, n_epochs, np.array(patient_loss_history).shape[0]), np.array(patient_loss_history), '.', alpha=0.2, label='Train Patient Loss')
        plt.plot(np.linspace(1, n_epochs, n_epochs), np.array(epoch_loss_history), label='Train Epoch Loss')
        plt.plot(np.linspace(1, n_epochs, n_epochs), np.array(epoch_val_loss_history), label='Val. Epoch Loss')
        plt.ylabel('loss')
    else:
        #plt.plot(np.linspace(1, n_epochs, np.array(batch_loss_history).shape[0]), np.log(np.array(batch_loss_history)), label='Train Batch Loss')
        plt.plot(np.linspace(1, n_epochs, np.array(patient_loss_history).shape[0]), np.array(patient_loss_history), '.', alpha=0.2,label='Train Patient Loss')
        plt.plot(np.linspace(1, n_epochs, n_epochs), np.array(epoch_loss_history), label='Train Epoch Loss')
        plt.plot(np.linspace(1, n_epochs, n_epochs), np.array(epoch_val_loss_history), label='Val. Epoch Loss')
        plt.plot(np.linspace(1, n_epochs, n_epochs), np.array(wbce_history), label='wbce_history')
        plt.plot(np.linspace(1, n_epochs, n_epochs), np.array(iou_history), label='iou_history')
        plt.yscale("log")
        plt.ylabel('log(loss)')
    plt.title(f'Loss: Type {loss_type}')
    plt.xlabel('Epoch')
    plt.legend(loc = 'best', frameon=True)
    if path_save is not None:
        plt.savefig(path_save+'{}.png'.format(name_plot), dpi=300)
        print(f'Loss plots guardados en {path_save}')
    if show:
        plt.show()
    plt.close('all')


def save_model(model, path='./', model_name='model', extension = '.pt'):
    if extension in ['.pt', '.pth']:
        pass
    else:
        extension = '.pt'
        
    if path[-1]=='/':
        pass
    else:
        path = path+'/'
        
    # Guardar el modelo
    if extension == '.pt':
        torch.save(model, path+model_name+'_nojit.pt')
        model_scripted = torch.jit.script(model) # Export to TorchScript
        model_scripted.save(path+model_name+'.pt') # Save
    else:
        torch.save(model.state_dict(), path+model_name+'.pth')
    print('Modelo {}{}.pth guardado.'.format(path, model_name))


def get_tiempo(con_fecha=False):
    fecha_hora_actual = datetime.datetime.now()
    # Obtener partes individuales de la fecha y hora
    anio = fecha_hora_actual.year
    mes = fecha_hora_actual.month
    dia = fecha_hora_actual.day
    hora = fecha_hora_actual.hour
    minuto = fecha_hora_actual.minute
    segundo = fecha_hora_actual.second
    if con_fecha:
        tiempo = '{}-{}-{}. {}:{}:{}'.format(anio, mes, dia, hora, minuto, round(segundo))
    else:
        tiempo = '{}:{}:{}'.format(hora, minuto, round(segundo))
    return tiempo


def loss_function(output, target, loss_type = 1):
    if loss_type == 1:
        # Definir función de pérdida
        loss_fn = nn.BCELoss(reduction='none')
        loss_ = loss_fn(output, target)
        return loss_
    elif loss_type == 2:
        # print(output.shape, flush=True)
        # print(target.shape, flush=True)
        
        # Loss de mascara nodulos:
        output_nodulo = output[:,0,:,:]
        target_nodulo = target[:,0,:,:]


        ## IoU
        intersection = torch.sum(output_nodulo * target_nodulo)
        union = torch.sum(output_nodulo) + torch.sum(target_nodulo) - intersection
        iou = intersection / (union + 1e-7)  # small constant to avoid division by zero
        if union == 0:
            iou=1
        loss_iou = 1 - iou
        ## BCE
        weights = target_nodulo*20+1
        loss = F.binary_cross_entropy(output_nodulo, target_nodulo, reduction='none')
        weighted_loss = loss * weights
        iou_loss1 = 4000*loss_iou
        wbce_loss1 = torch.sum(weighted_loss)
        loss_total_0 = iou_loss1 + wbce_loss1
        
        # Loss de mascara sana:
        ## Solo BCE
        output_sana = output[:,1,:,:]
        target_sana = target[:,1,:,:]
        weights = (-1*target_nodulo+1)*20+1
        loss = F.binary_cross_entropy(output_sana, target_sana, reduction='none')
        weighted_loss = loss * weights
        loss_total_1 = 1000*loss_iou + torch.sum(weighted_loss)
        
        
        loss_total = loss_total_0 + loss_total_1
        
        
        return loss_total, iou_loss1, wbce_loss1
    elif loss_type == 3:
        
        intersection = torch.sum(output * target)
        union = torch.sum(output) + torch.sum(target) - intersection
        iou = intersection / (union + 1e-7)  # small constant to avoid division by zero
        loss_iou = 1 - iou
        # print(loss_iou)
        return loss_iou
    elif loss_type == 4:
        weights = target*20+1
        loss = F.binary_cross_entropy(output, target, reduction='none')
        weighted_loss = loss * weights
        return torch.sum(weighted_loss)
    else:
        print('Indica una loss function que sea 1, 2 o 3. Has indicado loss = {}'.format(loss_type))

def save_patients_train_val_csv(train_list, val_list, folder_path):
    nombre_archivo_csv = f"{folder_path}pacientes_train_val.csv"

    # Combinar las dos listas en una lista de tuplas (cada tupla representa una fila en el CSV)
    filas = list(zip(train_list, val_list))

    # Escribir la información en el archivo CSV
    with open(nombre_archivo_csv, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["train_list", "val_list"])  # Escribir encabezados de columna
        writer.writerows(filas)

def train(model, n_epochs:int =4, 
          batch_size: int = 4, 
          val_split: float = 0.2,
          path2dataset: str = '../../manifest-1675801116903/LIDC-IDRI/',
          path2savefiles: str = './',
          plot_metrics: bool = False,
          save_plots: bool = False,
          save_epochs = None,
          model_extension = '.pt',
          failed_patients: list = [],
          loss_type: int = 1,
          verbose: bool = False):
    """Ejecuta el entrenamiento

    Args:
        model (_type_): modelo a entrenar
        epochs (int, optional): numero de epocas. Defaults to 4.
        batch_size (int, optional): batch de imagenes (no pacientes) a evaluar antes de haer backprop. Defaults to 4.
        val_split (float, optional): porcentaje del dataset a validation. Defaults to 0.2.
        path2dataset: str = '../../manifest-1675801116903/LIDC-IDRI/',
        plot: bool = False,
        save_plots: bool = False)
    """
    patients = os.listdir(path2dataset)
    patients = [pat for pat in patients if not pat=='LICENSE' and pat not in failed_patients]

    train_patients, val_patients = train_val_split(patients, val_split)
    train_patients, val_patients = ['LIDC-IDRI-0001'], ['LIDC-IDRI-0002']
    save_patients_train_val_csv(train_patients, val_patients, path2savefiles)
    # Definir optimizador
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


    loss_batch = np.array([])
    batch_loss_history = np.array([])

    loss_patient = np.array([])
    patient_loss_history = np.array([])

    epoch_loss_history = np.array([])
    epoch_loss_history = np.array([])
    epoch_val_loss_history = np.array([])
    tiempos_paciente = deque([6,6,6,6,6], maxlen=5)
    print('Inicio de entrenamiento: {}'.format(get_tiempo(con_fecha=True)))
    iou_history = np.array([])
    wbce_history = np.array([])
    for epoch in range(n_epochs):
        print(f'Epoch: {epoch+1}/{n_epochs}. {get_tiempo(con_fecha=True)}')
        loss_patient = np.array([])
        random.shuffle(train_patients)
        len_train_patients = len(train_patients)
        tqdm_train_patients = tqdm(train_patients,leave=False, position=0)
        iou_epoch = np.array([])
        wbce_epoch = np.array([])
        for i, id_pat in enumerate(tqdm_train_patients):
            inicio = time.time()
            time.sleep(1)
            
            tqdm_train_patients.set_description('Epoch: {}/{}. {}. Rate {} s/p. {}/{}. {}. Progreso de la epoca:'.format(epoch+1, n_epochs, get_tiempo(), 
                                                                                                           round(sum(tiempos_paciente)/5, 2),
                                                                                                           i,
                                                                                                           len_train_patients,
                                                                                                           id_pat))
            t1 = time.time()
            # Cargamos datos de un paciente:
            patient = Patient(id_pat)
            # Escalamos:
            patient.scale()
            t2 = time.time()
            # Obtenemos los tensores:
            imgs, mask = patient.get_tensors(scaled=True)
            t3 = time.time()
            dataset = TensorDataset(imgs, mask)

            train_loader = DataLoader(dataset,batch_size=batch_size, shuffle=True)
            loss_batch = np.array([])
            t4 = time.time()
            if verbose:
                print('Conseguir paciente: ',t2-t1, 's')
                print('Conseguir tesores de paciente: ',t3 -t2 , 's')
                print('Preparar data loader: ',t4-t3, 's')
            for batch_idx, (data, target) in enumerate(train_loader):
                t6 = time.time()
                if torch.all(target[:,0,:,:] == 0):
                    if random.random() > 0.10:
                        # print('\t es 0')
                        continue
                    # else:
                    #     print('batch sin tumor pero lo cogemos')
                # else:
                #     print('batch con tumor')
                t6_ = time.time()
                if torch.cuda.is_available():
                    device = torch.device('cuda')
                    data, target = data.to(device), target.to(device)

                t7 = time.time()
                # print(torch.mean(target))
                # # Forward pass
                output = model(data)
                t8 = time.time()
                # Calcular pérdida
                loss,iou, wbce = loss_function(output, target, loss_type=loss_type)
                iou_epoch = np.append(iou_epoch, iou.cpu().detach().numpy())
                wbce_epoch = np.append(wbce_epoch, wbce.cpu().detach().numpy())
                t9 = time.time()
                # print('\t loss', loss, 'torch.mean(target):', torch.mean(target))
                # # # Calcular gradientes y actualizar parámetros
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                t10 = time.time()
                loss_batch = np.append(loss_batch, loss.item())
                batch_loss_history = np.append(batch_loss_history, loss.item())
                if verbose:
                    print('Comprobar ceros en batch: ',t7-t6, 's')
                    print('Mover batch a la grafica: ', t6_-t6)
                    print('Predict: ',t8-t7, 's')
                    print('Loss calculation: ',t9-t8, 's')
                    print('Back propagation: ',t10-t9, 's')
                    print('el batch tarda: ', t10-t6)
                    print('-----------')
            print('Realizacion media iou...')
            print(np.mean(iou_epoch))
            print('---------')
            print('realizadon media de wbce: ')
            print(np.mean(wbce_epoch))
            
            del data
            del target
            del dataset
            del patient
            if len(loss_batch)==0:
                print('Paciente sin tumor (es posible, pacient=988 por ejemplo). Paciente saltado')
                continue
            loss_patient = np.append(loss_patient, np.mean(np.array(loss_batch)))
            # print('id_patient', id_pat, 'loss_patient', loss_patient)
            patient_loss_history = np.append(patient_loss_history, loss_patient)
            tiempo_paciente = time.time()-inicio
            tiempos_paciente.append(tiempo_paciente)
        iou_history = np.append(iou_history, np.mean(iou_epoch))
        wbce_history = np.append(wbce_history, np.mean(wbce_epoch))
        epoch_loss_history = np.append(epoch_loss_history, np.mean(np.array(loss_patient)))
        print('Fin epoca {}: {}'.format(epoch+1, get_tiempo()))

        # # Calculemos el loss del val:
        val_loss = get_val_loss(model, val_patients, batch_size, loss_type = loss_type)
        epoch_val_loss_history = np.append(epoch_val_loss_history, val_loss)

        if save_epochs is not None:
            if epoch//save_epochs == epoch/save_epochs and epoch>1:
                save_model(model, path2savefiles, model_name= 'model-epoch{}'.format(epoch))
                if save_plots:
                    data_dict ={
                        'epoch_loss_history': epoch_loss_history,
                        'batch_loss_history': batch_loss_history,
                        'patient_loss_history': patient_loss_history,
                        'epoch_val_loss_history': epoch_val_loss_history,
                        'iou_history': iou_history,
                        'wbce_history': wbce_history
                        }
                    plot(data_dict, show=plot_metrics, path_save=path2savefiles, name_plot= 'loss_epoch_{}'.format(epoch+1), loss_type=loss_type)

        print('Train Epoch: {}\t Train Loss: {:.6f}. Val Loss: {:.6f}'.format(
            epoch+1, epoch_loss_history[-1], epoch_val_loss_history[-1]))
        print('-----------------------------------')
    print('Fin de entrenamiento: {}'.format(get_tiempo()))
    data_dict ={
                'epoch_loss_history': epoch_loss_history,
                'batch_loss_history': batch_loss_history,
                'patient_loss_history': patient_loss_history,
                'epoch_val_loss_history': epoch_val_loss_history,
                'iou_history': iou_history,
                'wbce_history': wbce_history
                }
    save_model(model, path2savefiles, model_name='finalmodel', extension=model_extension)
    if save_plots:
        plot(data_dict, show=plot_metrics, path_save=path2savefiles, loss_type=loss_type)
    else:
        plot(data_dict, show=plot_metrics, loss_type=loss_type)
    

def checks_alright(args):
    if not args.n_epochs > 0 and not isinstance(args.n_epochs, int):
        raise ValueError("n_epochs no es entero y >0")
    if not args.batch_size > 0 and not isinstance(args.batch_size, int):
        raise ValueError("batch_size no es entero y >0")
    
    # print(args.val_split > 0 and not args.val_split < 1 and isinstance(args.val_split, float))
    if not args.val_split > 0 and not args.val_split < 1 and not isinstance(args.val_split, float):
        print('fallooooo')
        raise ValueError("val_split no es float y >0 y >1")
    
    try:
        if not os.path.exists(args.path2dataset):
            raise ValueError("La ruta al dataset, path2dataset, esta mal")
    except:
        raise ValueError("La ruta al dataset, path2dataset, esta mal")
    try:
        if not os.path.exists(args.path2savefiles):
            raise ValueError("La ruta a donde se guardaran los archivos, path2savefiles, esta mal")
    except:
        raise ValueError("La ruta a donde se guardaran los archivos, path2savefiles, esta mal")
    
    if not isinstance(args.plot_metrics, bool):
        raise ValueError("save_plots no es booleano")
    
    if not isinstance(args.plot_metrics, bool):
        raise ValueError("save_plots no es booleano")
    
    if not isinstance(args.save_epochs, int):
        if  args.save_epochs > 0 and not args.save_epochs < args.n_epochs:
            raise ValueError("save_epochs no es entero y >0 y menos que n_epochs")



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    # Agregar los argumentos necesarios
    parser.add_argument('--n_epochs', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--val_split', type=float, default=0.2)
    parser.add_argument('--path2dataset', type=str, default='../../manifest-1675801116903/LIDC-IDRI/')
    parser.add_argument('--path2savefiles', type=str, default='./')
    parser.add_argument('--plot_metrics', action='store_true', default = False)
    parser.add_argument('--save_plots', action='store_true', default = True)
    parser.add_argument('--save_epochs', type=int, default=None)
    parser.add_argument('--model_extension', type=str, default='.pt')
    parser.add_argument('--loss_type', type=int, default=1)
    # Obtener los argumentos proporcionados por el usuario
    args = parser.parse_args()
    checks_alright(args)

    archivo = open('./failed_patients.txt', 'r')  # Reemplaza 'nombre_archivo.txt' por el nombre de tu archivo

    failed_patients = []

    for linea in archivo:
        linea = linea.strip()  # Elimina los espacios en blanco al principio y al final de la línea
        failed_patients.append(linea)
    archivo.close()
    print('Descargando el modelo...')
    # # Descargamos el modelo preentrenado:
    # model = UNet(in_channels=1, 
    #                  out_channels=1, 
    #                  init_features=32) # , dropout_rate=0.2)
    model_entrenado = torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana', pretrained=True, scale=0.5)
    model = UNet(n_channels=3, n_classes=2)  # , init_features=32) # , dropout_rate=0.2)
    # # Cargar los pesos del modelo entrenado en el modelo aleatorio
    model.load_state_dict(model_entrenado.state_dict())
    if torch.cuda.is_available():
        print('moviendo a la grafica...')
        try:
            device = torch.device('cuda')
            model = model.to(device)
            print('INFO: Modelo funcionando en la GPU')
        except:
            print('WARNING: Hay fallo al trasladar a la grafica, el enrteno ira muy lento.'\
                  'Soluciones: \n 1. Reinicia el ordenador \n 2. Prueba NVIDIA-SMI')
    
    
    # Llamar a la función train con los argumentos
    train(model, n_epochs=args.n_epochs, batch_size=args.batch_size, val_split=args.val_split,
        path2dataset=args.path2dataset, path2savefiles=args.path2savefiles,
        plot_metrics=args.plot_metrics, save_plots=args.save_plots,
        save_epochs=args.save_epochs, failed_patients=failed_patients,
        model_extension=args.model_extension, loss_type=args.loss_type)
        
        
        

