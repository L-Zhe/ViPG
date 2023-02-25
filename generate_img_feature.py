import  torch
from    tqdm import tqdm
from    torch.utils.data import DataLoader
import  os
from    PIL import Image, ImageFile
from    transformers import ViTFeatureExtractor
from    transformers import ViTModel
ImageFile.LOAD_TRUNCATED_IMAGES = True
import  numpy as np
from    math import ceil

class collate_fn:
    
    def __init__(self, path, vis_token_file):
        self.path = path
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(vis_token_file)

    
    def __call__(self, batch):
        image = [Image.open(os.path.join(self.path, file)).convert("RGB") for file in batch]
        return batch, self.feature_extractor(images=image, return_tensors='pt')


def get_data_loader(path, collate_fn):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # file_list = os.listdir(path)
    with open('/home/linzhe/DownloadConceptualCaptions-master/long_cap.img', 'r') as f:
        file_list = [line.strip('\n') for line in f.readlines()]
    size = ceil(len(file_list) / 2)
    index = 0
    file_list = file_list[index * size : min((index + 1) * size, len(file_list))]
    train_loader = DataLoader(dataset=file_list,
                              batch_size=1,
                              shuffle=False,
                              num_workers=16,
                              pin_memory=True,
                              collate_fn=collate_fn)
    return train_loader


def generate_feture(data_loader, model_file, save_path):
    ViT = ViTModel.from_pretrained(model_file).cuda()
    ViT.eval()
    with torch.no_grad():
        for file, input in tqdm(data_loader):
            for key in input.keys():
                input[key] = input[key].cuda()
            img_feture = ViT(**input).last_hidden_state.cpu()
            length = len(file)
            for i in range(length):
                np.save(os.path.join(save_path, file[i]) + '.feature', img_feture.numpy())


if __name__ == '__main__':
    path = '/home/linzhe/DownloadConceptualCaptions-master/training/'
    save_path = '/home/linzhe/DownloadConceptualCaptions-master/train_image_feature/'
    vis_token_file = '/home/linzhe/tools/pretrain_model/ViT'
    model_file = '/home/linzhe/tools/pretrain_model/ViT'
    collate_fn = collate_fn(path, vis_token_file)
    data_loader = get_data_loader(path, collate_fn)
    generate_feture(data_loader, model_file, save_path)
    # file_list = os.listdir(path)
    # cnt = 0
    # for file in tqdm(file_list):
    #     path_file = os.path.join(path, file)
    #     try:
    #         Image.open(path_file).convert("RGB")
    #     except:
    #         cnt += 1
    #         os.remove(path_file)
    # print(cnt)