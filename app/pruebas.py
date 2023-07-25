from unet import UNet
import torch
from processLIDC3 import Patient
import matplotlib.pyplot as plt
import torch.nn as nn
from torchviz import make_dot

if __name__ =='__main__':
    model_entrenado = torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana', pretrained=True, scale=0.5)
    model = UNet(n_channels=3, n_classes=2)  # , init_features=32) # , dropout_rate=0.2)
    # # Cargar los pesos del modelo entrenado en el modelo aleatorio
    model.load_state_dict(model_entrenado.state_dict())
        
    
    patient = Patient('LIDC-IDRI-0002')
    
    patient.scale()
    
    images, mask = patient.get_tensors(scaled=True)
    # print(mask.shape)
    print(images.shape)
    print(mask.shape)
    # plt.imshow(mask[179, 0])
    # plt.show()
    # plt.imshow(mask[179,1])
    # plt.show()
    pred = model(images[175:181,:,:])
    prediccion = pred.cpu().detach().numpy()[0,:,:,:]
    if not torch.all(pred > 0):
            print('OJOOOOOO que hay valores negativos!!')
    if torch.any(pred >1):
        print('OJOOOOOO que hay valores por encima de uno!!')
    # print(prediccion.shape)
    plt.imshow(prediccion[0])
    plt.show()
    plt.imshow(prediccion[1])
    plt.show()
    # print(prediccion.shape)
    
    # graph = make_dot(pred, params=dict(model.named_parameters()))

    # # Guarda la imagen
    # graph.format = 'png'  # También puedes guardarla en otros formatos como 'svg'
    # graph.render("unet_github")
    
    