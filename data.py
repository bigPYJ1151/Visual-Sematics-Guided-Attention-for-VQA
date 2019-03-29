
import cv2
import random
import os
import numpy as np
import torch 
import json
import h5py
from torch.utils import data
from torchvision import transforms
from PIL import Image
import matplotlib.image as im
from text_preprocess import prepare_answers, prepare_questions

class VQA(data.Dataset):
    def __init__(self, COCO_root, QA_root, processed_COCO_path, vocab_path, mode):
        super().__init__()
        self.mode = mode
        question_path = os.path.join(QA_root, "OpenEnded_mscoco_{0}2014_questions.json")
        answer_path = os.path.join(QA_root, "mscoco_{0}2014_annotations.json")

        if self.mode == 0:
            self.question_path = question_path.format("train")
            self.answer_path = answer_path.format("train")
            self.image_path = os.path.join(COCO_root, 'train2014')
        elif self.mode == 1:
            self.question_path = question_path.format("val")
            self.answer_path = answer_path.format("val")
            self.image_path = os.path.join(COCO_root, 'val2014')
        else:
            self.question_path = os.path.join(QA_root, "OpenEnded_mscoco_test2015_questions.json")
            self.image_path = os.path.join(COCO_root, 'test2015')

        with open(vocab_path, 'r') as f:
            vocab = json.load(f)
        self.vocab = vocab

        self.token_to_index = self.vocab['question'] #mapping question token to question vocab indice
        self.answer_to_index = self.vocab['answer'] #mapping answer to answer vocab indice

        self.index_to_token = {i: k for k, i in self.token_to_index.items()}
        self.index_to_answer = {i: k for k, i in self.answer_to_index.items()}

        with open(self.question_path, 'r') as f:
            question_json = json.load(f)
        self.questions_origin = list(prepare_questions(question_json))
        self.questions = [self._encode_question(q) for q in self.questions_origin]

        if hasattr(self, 'answer_path'):
            with open(self.answer_path) as f:
                answer_json = json.load(f)

            self._check_integrity(question_json, answer_json)
            self.answers_origin = list(prepare_answers(answer_json))
            self.answers = [self._encode_answer(a) for a in self.answers_origin]

        self.image_feature_path = processed_COCO_path
        self.COCOid_to_filename = self._get_COCOid_to_filename()

        if self.mode != 2:
            self.COCOid_to_index = self._create_COCO_to_index() #mapping COCO id to h5py file image features index
        
        self.COCOids = [q['image_id'] for q in question_json['questions']]

        self.transform = transforms.Compose([
                transforms.ToTensor(), #to tensor and normalize to [0,1]
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
                ])
        cv2.setNumThreads(0)

    def data_recover(self, item, a_index = None):     
        q =' '.join(self.questions_origin[item]) 

        if hasattr(self, 'answer_path') and a_index is None:
            a_encode = list(self.answers[item])
            a_index = a_encode.index(max(a_encode))
            a = self.index_to_answer[a_index]
        else:
            a = self.index_to_answer[a_index]

        v = im.imread(os.path.join(self.image_path, self.COCOid_to_filename[self.COCOids[item]]))

        return v, q, a

    def _get_COCOid_to_filename(self):
        id_to_filename = {}

        for filename in os.listdir(self.image_path):
            if not filename.endswith(".jpg"):
                continue
    
            name = (filename.split('_')[-1]).split('.')[0]
            id = int(name)
            id_to_filename[id] = filename
        
        return id_to_filename

    def _create_COCO_to_index(self):
        assert self.mode in [0,1], "Mode Error" 
        with h5py.File(self.image_feature_path, 'r') as f:
            if self.mode == 0:
                name = 'train_image'
            else:
                name = 'val_image'
            COCO_ids = f[name + '_ids'][:]
            COCOid_to_index = {id: i for i, id in enumerate(COCO_ids)}

        return COCOid_to_index

    def _encode_question(self, question):
        vec = torch.zeros((self.max_question_length,), dtype=torch.long)
        for i, token in enumerate(question):
            index = self.token_to_index.get(token, 0)
            vec[i] = index
        
        return vec, len(question)

    def _encode_answer(self, answer):
        vec = torch.zeros((len(self.answer_to_index),))
        for ans in answer:
            index = self.answer_to_index.get(ans)
            if index != None:
                vec[index] += 1
        
        return vec
    
    @property
    def num_tokens(self):
        return len(self.token_to_index) + 1

    @property
    def max_question_length(self):
        if not hasattr(self, 'max_length'):
            self.max_length = max(map(len, self.questions_origin))
        return self.max_length

    def _check_integrity(self, question_json, answer_json):
        qa_pairs = list(zip(question_json['questions'], answer_json['annotations']))

        assert all(q['question_id'] == a['question_id'] for q, a in qa_pairs), 'Questions not aligned with answers'
        assert all(q['image_id'] == a['image_id'] for q, a in qa_pairs), 'Image id of question and answer don\'t match'
        assert question_json['data_type'] == answer_json['data_type'], 'Mismatched data types'
        assert question_json['data_subtype'] == answer_json['data_subtype'], 'Mismatched data subtypes'
    
    def _get_image_feature(self, COCOid):
        assert self.mode in [0,1], "Mode Error" 
        if self.mode == 0:
            name = 'train_image'
        else:
            name = 'val_image'
        if not hasattr(self, 'features_file'):
            self.features_file = h5py.File(self.image_feature_path, 'r')

        index = self.COCOid_to_index[COCOid]
        data = self.features_file[name+'_feature']
        img = data[index].astype('float32')
        data = self.features_file[name+'_semantic']
        label = data[index].astype('float32')
        return torch.from_numpy(img), torch.from_numpy(label)

    def __getitem__(self, item):
        q, qlen = self.questions[item]
        image_id = self.COCOids[item]

        if self.mode in [0, 1]:
            a = self.answers[item]
            v, l = self._get_image_feature(image_id)
            return qlen, q, a, v, l, item
        
        else:
            filename = self.COCOid_to_filename[image_id]
            v = cv2.imread(os.path.join(self.image_path, filename), cv2.IMREAD_COLOR)
            v = cv2.resize(v, (448, 448), interpolation=cv2.INTER_LINEAR)
            v = v[:, :, ::-1] #bgr to rgb
            v = self.transform(v.copy())
            return qlen, q, v, item
        
    def __len__(self):
        return len(self.questions)

class COCO(data.Dataset):
    def __init__(self, root, split, req_label=False, req_augment=False,  
                crop_size=448, scales=(1), flip=False):

        super().__init__()

        self.root = root
        self.split = split
        self.image_path = os.path.join(root, split)
        self.id_to_filename = self._get_COCOid_to_filename()
        self.ids = list(self.id_to_filename.keys())

        self.req_label = req_label
        if req_label:
            self.label_path = os.path.join(root, "label2017", split)

        self.req_augment = req_augment
        self.crop_size = crop_size
        self.scales = scales
        self.flip = flip

        self.transform = transforms.Compose([
                transforms.ToTensor(), #to tensor and normalize to [0,1]
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
                ])

        cv2.setNumThreads(0)

    def _get_COCOid_to_filename(self):
        id_to_filename = {}

        for filename in os.listdir(self.image_path):
            if not filename.endswith(".jpg"):
                continue
    
            name = (filename.split('_')[-1]).split('.')[0]
            id = int(name)
            id_to_filename[id] = filename
        
        return id_to_filename

    def _augmentation(self, image, label):
        scale_factor = random.choice(self.scales)
        h, w = label.shape
        th, tw = int(scale_factor * h), int(scale_factor * w)
        
        image = cv2.resize(image, (th, tw), interpolation=cv2.INTER_LINEAR)
        label = Image.fromarray(label).resize((th, tw), resample=Image.NEAREST)
        label = np.asarray(label, dtype=np.uint8)

        h, w = label.shape
        start_h = random.randint(0, h - self.crop_size)
        start_w = random.randint(0, w - self.crop_size)
        end_h = start_h + self.crop_size
        end_w = start_w + self.crop_size
        image = image[start_h:end_h, start_w:end_w]
        label = label[start_h:end_h, start_w:end_w]

        if self.flip:
            if random.random() < 0.5:
                image = np.fliplr(image).copy()  # HWC
                label = np.fliplr(label).copy()  # HW

        return image, label

    def _load_data(self, index):
        COCOid = self.ids[index]
        filename = self.id_to_filename[COCOid]

        image = cv2.imread(os.path.join(self.image_path, filename), cv2.IMREAD_COLOR)
        
        if self.req_label:
            filename = filename.split('.')[0] + ".png"
            label = cv2.imread(os.path.join(self.label_path, filename), cv2.IMREAD_GRAYSCALE)
        else:
            label = None

        return image, label

    def __getitem__(self, index):
        image, label = self._load_data(index)
        image = cv2.resize(image, (448, 448), interpolation=cv2.INTER_LINEAR)
     
        if self.req_label:
            label = Image.fromarray(label).resize((448, 448), resample=Image.NEAREST)
            label = np.asarray(label, dtype=np.uint8)

            if self.req_augment:
                image, label = self._augmentation(image, label)

        image = image[:, :, ::-1] #bgr to rgb
        image = self.transform(image.copy())

        if label is None:
            return image, self.ids[index]
        else:
            return image, torch.from_numpy(label.astype(np.int64)), self.ids[index]

    def __len__(self):
        return len(self.ids)


    
    


