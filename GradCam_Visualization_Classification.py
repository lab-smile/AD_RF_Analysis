# Import Base Library
import os
import random
import cv2
import numpy as np
import argparse
from PIL import Image

# Pytorch Related Library
import torch
from torchvision import transforms
from transformers import SwinForImageClassification
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, \
    EigenGradCAM, FullGrad
from pytorch_grad_cam.utils.image import show_cam_on_image


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_checkpoint', type=str,
                        default='./savedmodel/Swin_classification_multi/Swin_classification8loss0.13acc0.64.pth')
    parser.add_argument('--image_path', type=str,
                        default='./data/21015_fundus_left_1', help='Input image path')
    # default='/red/ruogu.fang/share/UKB/data/Eye/21015_fundus_left_1_good/', help='Input image path')
    parser.add_argument('--image', type=str, default='1000014_21015_0_0.png')
    parser.add_argument('--method', type=str, default='scorecam')

    parser.add_argument('--aug_smooth', dest='aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument('--eigen_smooth', dest='eigen_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')

    parser.add_argument('--no_aug_smooth', dest='aug_smooth', action='store_false',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument('--no_eigen_smooth', dest='eigen_smooth', action='store_false',
                        help='Apply test time augmentation to smooth the CAM')

    parser.add_argument('--predictor_idx', type=int, default=0, help='which risk factor prediction you want to extract')
    parser.add_argument('--class_idx', type=int, default=0,
                        help='which class of specific risk factor you want to extract')

    parser.set_defaults(aug_smooth=True, eigen_smooth=True)

    args = parser.parse_args()
    return args


class SwinforClassification(torch.nn.Module):
    def __init__(self, model_name_or_path='microsoft/swin-large-patch4-window12-384-in22k', n_label=7):
        super().__init__()
        model = SwinForImageClassification.from_pretrained(model_name_or_path, num_labels=n_label,
                                                           ignore_mismatched_sizes=True)
        self.swin = model.swin
        self.regressor_sex = torch.nn.Linear(1536, 2)
        self.regressor_smoking = torch.nn.Linear(1536, 3)
        self.regressor_sleeplessness = torch.nn.Linear(1536, 3)
        self.regressor_alcohol = torch.nn.Linear(1536, 6)
        self.regressor_depression = torch.nn.Linear(1536, 6)
        self.regressor_economic_status = torch.nn.Linear(1536, 5)

    def forward(self, x):
        y_swin = self.swin(x).pooler_output
        y_sex = self.regressor_sex(y_swin)
        y_smoking = self.regressor_smoking(y_swin)
        y_sleeplessness = self.regressor_sleeplessness(y_swin)
        y_alcohol = self.regressor_alcohol(y_swin)
        y_depression = self.regressor_depression(y_swin)
        y_economic_status = self.regressor_economic_status(y_swin)

        return y_sex, y_smoking, y_sleeplessness, y_alcohol, y_depression, y_economic_status


class SingleOutputModel(torch.nn.Module):
    def __init__(self, model, output_index=0):
        super().__init__()
        self.model = model
        self.output_index = output_index

    def forward(self, x):
        return self.model(x)[self.output_index]


class ClassifierOutputTarget:
    def __init__(self, category):
        self.category = category

    def __call__(self, model_output):
        return model_output[self.category]


def reshape_transform(tensor, height=18, width=18):
    # this part of the code is from https://github.com/jacobgil/pytorch-grad-cam
    result = tensor[:, :, :].reshape(tensor.size(0),
                                     height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def main():
    random_state = 0
    np.random.seed(random_state)
    random.seed(random_state)
    torch.manual_seed(random_state)
    os.environ["PYTHONHASHSEED"] = str(random_state)

    args = get_args()

    result_dir = os.path.join('./result', 'GradCAM_Classification', args.method)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    else:
        pass

    predictor_index = args.predictor_idx
    class_index = args.class_idx

    img = Image.open(os.path.join(args.image_path, args.image))
    img = img.resize((576, 576))
    img = np.array(img)
    img = img.astype(np.float32)
    img = img / 255
    convert_tensor = transforms.Compose([transforms.ToTensor(),
                                         transforms.Resize((576, 576))
                                         ])

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    input_tensor = convert_tensor(img).to(device)
    model = SwinforClassification(model_name_or_path='microsoft/swin-large-patch4-window12-384-in22k', n_label=7)
    checkpoint = torch.load(args.model_checkpoint, map_location=torch.device(device))
    model.load_state_dict(checkpoint)

    model_for_gradcam = SingleOutputModel(model, predictor_index)
    model.to(device)
    model.eval()

    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "fullgrad": FullGrad}

    targets = [ClassifierOutputTarget(class_index)]
    target_layers = [model_for_gradcam.model.swin.encoder.layers[-1].blocks[-1].layernorm_before]

    cam = methods[args.method](model=model_for_gradcam, target_layers=target_layers,
                               reshape_transform=reshape_transform)
    _ = model_for_gradcam(input_tensor.unsqueeze(0))
    output = cam(input_tensor=input_tensor.unsqueeze(0), targets=targets,
                 aug_smooth=args.aug_smooth, eigen_smooth=args.eigen_smooth)

    cam = output[0, :]
    cam_image = show_cam_on_image(img, cam)

    if args.aug_smooth == True and args.eigen_smooth == True:
        cv2.imwrite(os.path.join(result_dir, args.image[:-4] + '_' + 'all_smooth' + '_' + str(args.predictor_idx) + '_'
                                 + str(args.class_idx) + '.jpg'), cam_image)

    elif args.aug_smooth == True and args.eigen_smooth == False:
        cv2.imwrite(
            os.path.join(result_dir, args.image[:-4] + '_' + 'aug_smooth' + '_' + str(args.predictor_idx) + '_'
                         + str(args.class_idx) + '.jpg'), cam_image)

    elif args.aug_smooth == False and args.eigen_smooth == True:
        cv2.imwrite(
            os.path.join(result_dir, args.image[:-4] + '_' + 'eigen_smooth' + '_' + str(args.predictor_idx) + '_'
                         + str(args.class_idx) + '.jpg'), cam_image)

    else:
        cv2.imwrite(
            os.path.join(result_dir, args.image[:-4] + '_' + 'no_smooth' + '_' + str(args.predictor_idx) + '_'
                         + str(args.class_idx) + '.jpg'), cam_image)


if __name__ == '__main__':
    main()
