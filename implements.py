import cv2
import numpy as np
from local_utils import detect_lp
from keras.models import model_from_json
import argparse
from PIL import Image


def load_model(path):
    try:
        path = path.split('.')[0]

        with open('%s.json' % path, 'r') as json_file:
            model_json = json_file.read()

        global model
        model = model_from_json(model_json, custom_objects={})

        model.load_weights('%s.h5' % path)

        print("Loading model successfully...")

        return model
    except Exception as e:
        print(e)


def preprocess_image(image_path,resize=False):
    #img = cv2.imread(image_path)
    img = np.fromfile(image_path,np.uint8)
    img = cv2.imdecode(img,cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255
    if resize:
        img = cv2.resize(img, (224,224))
    return img


def get_plate(vehicle, Dmax=608, Dmin=256):
    ratio     = float(max(vehicle.shape[:2])) / min(vehicle.shape[:2])
    side      = int(ratio * Dmin)
    bound_dim = min(side, Dmax)

    _ , LpImg, _, cor = detect_lp(model, vehicle, bound_dim, lp_threshold=0.5)  

    if not LpImg:
        print('Number Plate Detected Failed.')

    return LpImg,cor

def Detect_plate(img):
    
    Lpimg = None
    dst   = None
    cor   = None

    try:
        Lpimg,cor = get_plate(img)
        print('Number Plate Detected Successed.')
    except Exception as e:
        print('Number Plate Detected Failed.')
        print(e)

    # convert cv2 image to pillow Image
    if Lpimg:
        dst = Image.fromarray((Lpimg[0]*255).astype(np.uint8))

    return dst,cor


def CreateParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder',     type=str, help='path to image_folder which contains text images')
    parser.add_argument('--workers',          type=int, help='number of data loading workers', default=0)
    parser.add_argument('--batch_size',       type=int, default=128, help='input batch size')
    parser.add_argument('--saved_model',      default='weights/v1.6-best_accuracy.pth', help="path to saved_model to evaluation")
    """ Data processing """
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH',             type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW',             type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb',              action='store_true', help='use rgb input')
    parser.add_argument('--character',        type=str, default='0123456789abcdefghijklmnopqrstuvwxyz가나다라마거너더러머버서어저고노도로모보소오조구누두루무부수우주아바사자배하허호국합육해공울산대인천광전울산경기강원충북남제', help='character label')
    parser.add_argument('--sensitive',        action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD',              action='store_true', help='whether to keep ratio then pad for image resize')
    """ Model Architecture """
    parser.add_argument('--Transformation',   type=str, default='TPS', help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction',type=str, default='ResNet', help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, default='BiLSTM', help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction',       type=str, default='Attn', help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial',     type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel',    type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel',   type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size',      type=int, default=256, help='the size of the LSTM hidden state')

    opt = parser.parse_args()

    return opt
