import cv2 as cv 

from collections import OrderedDict

from model.lightning_wrappers import DeepFontWrapper
from model.deepfont import DeepFont, DeepFontAutoencoder
import torch 
import torch.nn.functional as F 
from dataset.transformations import IPtrans
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABELS2ID = {'unbold': 0, 'bold': 1, 'italic': 2, 'bold_italic': 3}
ID2LABELS = {0: 'unbold', 1: 'bold', 2: 'italic', 3: 'bold_italic'}

def predict(model: DeepFontWrapper, img): 
    img_input = IPtrans(img)
    img_input.unsqueeze_(0)
    logit = model.forward(img_input.cuda())
    pred = F.softmax(logit, dim=1)

    return ID2LABELS[pred.argmax().item()]

def load_model(checkpoint_path, n_classes, device): 
    df = DeepFont(autoencoder= DeepFontAutoencoder(),  num_classes= n_classes)
    checkpoint = torch.load(checkpoint_path)
    
    
    df_state_dict = OrderedDict()

    for k, v in checkpoint['state_dict'].items():
        name = k[6:] 
        df_state_dict[name]=v

    df.load_state_dict(df_state_dict)
    return DeepFontWrapper(model= df, num_classes= n_classes).to(device)




if __name__=="__main__": 
    img1 = cv.imread("v5.png", cv.IMREAD_GRAYSCALE)
    img2 = cv.imread("1.jpg", cv.IMREAD_GRAYSCALE)

    #Loading model  
    
    model = load_model("output_df/last.ckpt", n_classes=4, device=device)
    
    infer_b = time.time()
    pred1 = predict(model, img=img1)
    pred2 = predict(model, img=img2)
    print(f"Inference time: {time.time() - infer_b}")
    cv.imshow(pred1, img1)
    cv.imshow(pred2, img2)
    cv.waitKey(0)
    