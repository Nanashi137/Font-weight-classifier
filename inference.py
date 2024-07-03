import cv2 as cv 
import numpy as np

from collections import OrderedDict

from model.lightning_wrappers import DeepFontWrapper
from model.deepfont import DeepFont, DeepFontAutoencoder
import torch 
import torch.nn.functional as F 
from dataset.transformations import IPtrans, inference_input
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


def ensemble_predict(model, img):
    all_soft_preds = []
    squeeze_ratio=np.random.uniform(low=1.5, high=3.5)
    for _ in range(2):
        patches = [inference_input(img, squeezing_ratio=squeeze_ratio) for _ in range(5)]
        # return patches
        inputs = torch.tensor(np.asarray(patches))

        preds = model(inputs.cuda())
        soft_preds = F.softmax(preds, dim=1)
        all_soft_preds.append(soft_preds)

    probs = torch.cat(all_soft_preds).mean(0)
    return ID2LABELS[probs.argmax().item()], torch.round(probs, decimals=3).tolist()



if __name__=="__main__": 
    img1 = cv.imread("v5.png", cv.IMREAD_GRAYSCALE)

    #Loading model  
    model = load_model("output_df/r.ckpt", n_classes=4, device=device)
    
    infer_b = time.time()
    pred1, score = ensemble_predict(model, img=img1)
    print(f"Inference time: {time.time() - infer_b}")
    print(pred1)
    print(np.argmax(np.array(score)))
    # cv.imshow("img", img1)
    # cv.waitKey(0)
    