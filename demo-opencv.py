import torch
import torch.utils.data
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision.transforms.functional import rgb_to_grayscale

from utils import AttnLabelConverter
from model import Model

from dataset import NormalizePAD
from PIL import Image,ImageDraw,ImageFont
import math
import os
from time import time

from implements import *

# OCR result Threshold
global thresh
thresh   = 0.5

# Image convert to Tensor
def ConvertToTensor(s_size, src):
    imgH = s_size[0]
    imgW = s_size[1]

    input_channel = 3 if src.mode == 'RGB' else 1

    transform     = NormalizePAD((input_channel, imgH, imgW))

    w, h          = src.size
    ratio         = w / float(h)

    if math.ceil(imgH * ratio) > imgW:
        resized_w = imgW
    else:
        resized_w = math.ceil(imgH * ratio)

    # tensor reshape
    resized_image = src.resize((resized_w, imgH), Image.BICUBIC)

    # img to Tensor
    tmp           = transform(resized_image)

    img_tensor    = torch.cat([tmp.unsqueeze(0)], 0)

    # rgb tensor convert to grayscale tensor
    img_tensor    = rgb_to_grayscale(img_tensor)

    return img_tensor

# OCR Recognition
def Recognition(opt,img):
    # static w,h
    s_size = [opt.imgH, opt.imgW]

    # result Text
    text   = []

    if img:
        # convert image to tensor
        src               = ConvertToTensor(s_size, img)

        batch_size        = src.size(0)
        image             = src.to(device)

        # For max length prediction
        length_for_pred   = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
        text_for_pred     = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

        preds             = model(image, text_for_pred, is_train=False)

        # select max probabilty (greedy decoding) then decode index to character
        _, preds_index    = preds.max(2)
        preds_str         = converter.decode(preds_index, length_for_pred)

        preds_prob        = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)

        pred_EOS          = preds_str[0].find('[s]')

        # prune after "end of sentence" token ([s])
        pred              = preds_str[0][:pred_EOS]
        pred_max_prob     = preds_max_prob[0][:pred_EOS]

        # calculate confidence score (= multiply of pred_max_prob)
        # confidence score type is tensor
        confidence_score  = pred_max_prob.cumprod(dim=0)[-1]

        if confidence_score >= thresh:
            text.append(pred)
        else:
            text.append('Missing OCR')
        
        text.append(confidence_score) # text = [predict text, confidence score]

    else:
        text.append('Missing Plate') # text  = [Missing Plate] 

    return text

# Get Coordinate of Number Plate
def GetCoordinate(cor):
    pts      = []
    x_coor   = cor[0][0]
    y_coor   = cor[0][1]

    for i in range(4):
        pts.append([int(x_coor[i]),int(y_coor[i])])

    pts      = np.array(pts, np.int32)
    pts      = pts.reshape((-1,1,2))
    return pts

if __name__ is '__main__':
    global device,converter

    """ wpod net model load """
    home                = os.getcwd()
    wpod_net_path       = os.path.join(home, "weights/wpod-net.json")
    load_model(wpod_net_path)

    """ argpaser & device configure"""
    device              = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt                 = CreateParser()

    """ gpu configure """
    cudnn.benchmark     = True
    cudnn.deterministic = True
    opt.num_gpu         = torch.cuda.device_count()

    """ model configuration """
    converter           = AttnLabelConverter(opt.character)
    opt.num_class       = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3

    model    = Model(opt)
    model    = torch.nn.DataParallel(model).to(device)

    # load model
    model.load_state_dict(torch.load(opt.saved_model, map_location=device))

    # index for Test
    idx      = 0

    # get image list
    dir_path = ''  # Input Your Image Directory Path
    img_list = os.listdir(dir_path)

    # predict
    model.eval()           # model convert to predict model
    with torch.no_grad():  # Don't Use gradient of model
        while True:
            # Time Check
            t        = time()
            # get image
            img_path = os.path.join(dir_path, img_list[idx])

            # Image preprocessing
            dst      = preprocess_image(img_path)

            # get plate image, get coordinate plate
            img,cor  = Detect_plate(dst)
            print('Detect Number Plate Time : ', time() - t)

            # OCR Time Check
            t1       = time()

            #### 435,100 img resize
            if img:
                img  = img.resize((435,100))

            # get ocr result
            result   = Recognition(opt,img)
            print('OCR Recognition Time : ', time() - t1)
            print('Total Process Time : ',time() - t)

            # cv2 image to PIL image, Required Draw text
            dst      = Image.fromarray((dst * 255).astype(np.uint8))
            # pillow font & draw Object
            font     = ImageFont.truetype('fonts/gulim.ttc',size=30)
            draw     = ImageDraw.Draw(dst)
            
            # draw ocr
            draw.text((30,280),result[0],(255,0,0),font=font) # predict text or Missing str

            if len(result) == 2: # confidence score 
                draw.text((30,300),str(round(float(result[1]),4)),(255,0,0),font=font)
            
            # ---------------- Image Format Convert ---------------
            # PIL image to cv2 image and RGB Fotmat to BGR Format
            dst      = np.array(dst)
            dst      = cv2.cvtColor(dst,cv2.COLOR_RGB2BGR)

            # plate image cv2 convert for Test
            if img:
                img  = np.array(img)
                img  = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
                cv2.imshow('plate',img)
            # -----------------------------------------------------

            # draw plate
            if cor:
                pts  = GetCoordinate(cor)
                cv2.polylines(dst,[pts],True,color=(255,0,255),thickness=1)
                del pts

            cv2.imshow('car image', dst)

            del cor,draw,font,img

            # cv2 Key Event
            if cv2.waitKey()   == ord('n'):
                if idx         == len(img_list):
                    pass

                idx += 1

            elif cv2.waitKey() == ord('p'):
                if idx         == 0:
                    pass

                idx -= 1

            elif cv2.waitKey() == 27:
                break
