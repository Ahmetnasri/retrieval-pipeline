from io import BytesIO
import aiofiles
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import logging
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import uvicorn
import chromadb
from fastapi import FastAPI, HTTPException, Request, Response, File, UploadFile, Body
from fastapi.responses import StreamingResponse, JSONResponse
from safetensors.torch import load_file,save_file
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct, Filter, FieldCondition, MatchValue, MatchAny
from qdrant_client.http import models as rest
import uuid
import json
from main_decision_maker import decision_maker
from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import rbd, load_image
from lightweight_model import byol_encoder,mobile_encoder,SSCD_model
from torchvision.transforms.functional import pil_to_tensor
from torchvision.transforms import functional as torchvision_F
from torchvision import transforms
from helpers.config import get_settings

settings = get_settings()



log_level=os.getenv('LOG_LEVEL',"INFO")
if log_level=="INFO":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
elif log_level=="DEBUG":
    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
    )
elif log_level=="WARNING":
    logging.basicConfig(
        level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s"
    )

class vector_db:

    def __init__(self,collection_name_phase1='my_documents1',collection_name_phase2='my_documents2',
                 port=6333,extractor_model='disk',keypoints=256,
                 lightweight_model_phase1='sscd', lightweight_model_phase2='sscd',
                 lightweight_model_path_phase1='models/sscd_disc_large.torchscript.pt',
                 lightweight_model_path_phase2='models/sscd_imagenet_mixup.torchscript.pt',
                 database_path_lightglue='rok'):
        """"
        Initialize the vector database connection and models.
        Phase 1: for SSCD embeddings of the whole image
        Phase 2: for SSCD embeddings of the cutted between hash image
        """
        self.extractor_model = extractor_model
        self.decision_model=decision_maker(extractor_model=extractor_model,keypoints=keypoints)
        if extractor_model == 'superpoint':
            self.lightglue_model = SuperPoint(max_num_keypoints=keypoints).eval().cuda().half()
        elif extractor_model == 'disk':
            self.lightglue_model = DISK(max_num_keypoints=keypoints).eval().cuda() 
        self.matcher = LightGlue(features=extractor_model).eval().cuda()  # load the matcher
        self.collection_name_phase1 = collection_name_phase1
        self.collection_name_phase2 = collection_name_phase2


        self.client_qdrant = QdrantClient(host=QDRANT_HOST, port=port)
        collections = self.client_qdrant.get_collections().collections
        collection_names = [col.name for col in collections]
        
        if lightweight_model_phase1 == 'sscd':
            self.lightweight_phase1=SSCD_model(lightweight_model_path_phase1)
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
            )
            self.transform = transforms.Compose([
                transforms.Resize(288),
                normalize,
            ])

        if lightweight_model_phase2 == 'sscd':
            self.lightweight_phase2=SSCD_model(lightweight_model_path_phase2)
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
            )
            self.transform = transforms.Compose([
                transforms.Resize(288),
                normalize,
            ])

        if self.collection_name_phase1 in collection_names:
            info = self.client_qdrant.get_collection(self.collection_name_phase1)
            total_points = info.points_count
            logging.info(f'Qdrant collection of phase 1 {self.collection_name_phase1} exists with {total_points} points.')
        else:
            logging.info(f'Creating Qdrant collection for phase 1: {self.collection_name_phase1}')
            self.client_qdrant.recreate_collection(
                collection_name=self.collection_name_phase1,
                vectors_config=VectorParams(size=DB_DIMENSIONALITY_PHASE1, distance=Distance.COSINE)  # 384 is embedding size for the model used
            )
        if self.collection_name_phase2 in collection_names:
            info = self.client_qdrant.get_collection(self.collection_name_phase2)
            total_points = info.points_count
            logging.info(f'Qdrant collection of phase 2 {self.collection_name_phase2} exists with {total_points} points.')
        else:
            logging.info(f'Creating Qdrant collection for phase 2: {self.collection_name_phase2}')
            self.client_qdrant.recreate_collection(
                collection_name=self.collection_name_phase2,
                vectors_config=VectorParams(size=DB_DIMENSIONALITY_PHASE2, distance=Distance.COSINE)  # 384 is embedding size for the model used
            )

    def add_to_database(self,embed,path,file_name,point_num: str='main',phase: int=2):
        if phase ==1:
            collection_name=self.collection_name_phase1
        elif phase ==2:
            collection_name=self.collection_name_phase2 
        try:
            points=[]
            id = str(uuid.uuid4())
            points.append(PointStruct(
                id=id,#,
                vector=embed,
                payload={"file_name": file_name,
                    "image_id": file_name.split('_between_')[0].split('.')[0],
                    "path": path,
                    "point_num": point_num}  
            ))
            self.client_qdrant.upsert(collection_name=collection_name, points=points)
            logging.info(f'{file_name} - {point_num} added to DB phase {phase} with ID: {id}')
        except:
            logging.info(f'{file_name} - {point_num} not added')
            raise HTTPException(status_code=404, detail="ERR_ADDING_TO_DB_PHASE_"+str(phase))

    def query(self,embed,phase: int=1,candidates=None):

        if phase ==1:
            collection_name=self.collection_name_phase1
            name_filter = None
            db_limit=DB_LIMIT_PHASE1
        elif phase ==2:
            collection_name=self.collection_name_phase2
            db_limit=DB_LIMIT_PHASE2
            if candidates is None:
                logging.error("Candidates must be provided for phase 2 queries")
                raise HTTPException(status_code=400, detail="ERR_CANDIDATES_MUST_BE_PROVIDED_PHASE_2")
            #len(name_filter.dict()['must'][0]['match']['any'])
            name_filter = rest.Filter(
            must=[
                    rest.FieldCondition(
                        key="image_id",
                        match=rest.MatchAny(any=candidates)
                    )
                ]
            )

        results = {}
        results_qdrant = self.client_qdrant.search(
            collection_name=collection_name,
            query_vector=embed,
            limit=db_limit,  # Number of top matches to return
            with_payload=True,  # Include metadata in results
            query_filter=name_filter
        )
        if  not results_qdrant:
            logging.info(f"No results found in Qdrant database for phase {phase}")
            raise HTTPException(status_code=404, detail="ERR_NO_RESULTS_FROM_DB_PHASE_"+str(phase))
        try:
            results['ids'] = [i.id for i in results_qdrant]
            results['file_name'] = [i.payload['file_name'] for i in results_qdrant]
            results['path'] = [i.payload['path'] for i in results_qdrant]
            results['point_num'] = [i.payload['point_num'] for i in results_qdrant]
            results['scores'] = [i.score for i in results_qdrant]
        except:
            logging.error("Error querying database", exc_info=True)
            raise HTTPException(status_code=404, detail="ERR_QUERYING_DB_PHASE_"+str(phase))
            
        return results

        #def extract_disk_feat(self,image):

    def extract_descriptors(self,image,file_name,SAVE_FEATS=False,rotate: bool=False):
        feats_list=[]
        if rotate:
            rotate_time=4
        else:
            rotate_time=1
        for i in range(rotate_time):
            image_rot=torchvision_F.rotate(image,90*i)
            feats=self.decision_model(image_rot[-1].cuda().half(),height=2048,width=1536,batch_size=1,rotate=True,num_patch=NUM_PATCH)
            feats_list.append(feats)
        #feats1=feats1.cpu()
        if SAVE_FEATS:
            path = DATABASE_PATH  + file_name.split('.')[0] + '.safetensors'
            logging.info(f'path of saving safetensor: {path}')
            save_file({'array':torch.Tensor(feats)},path)

        return feats_list
    
    def extract_lightglue_featutres(self,image,file_name,SAVE_FEATS=False,rotate: bool=False, resize: bool=False):
        feats_list=[]
        if rotate:
            rotate_time=4
        else:
            rotate_time=1
        if resize:
            image = F.interpolate(
                image,
                size=(image.shape[2] // 2, image.shape[3] // 2),
                mode="bilinear",
                align_corners=False
            )
        for i in range(rotate_time):
            image_rot=torchvision_F.rotate(image,90*i)
            with torch.no_grad():
                feats = self.lightglue_model({'image':(image_rot.float() / 255.0 ).cuda().half()})
            feats_list.append(feats)
        if SAVE_FEATS:
            path = DATABASE_PATH_LIGHTGLUE  + file_name.split('.')[0] + '.safetensors'
            logging.info(f'path of saving safetensor: {path}')
            save_file(feats,path)
        return feats_list
    

    def embed(self,images,phase: int=1):
        """
        Embed the image using the lightweight model.
        The input image is expected to be a torch tensor of the shape [batch, channels, height, width].
        The input image is expected to be sent to the tansform funtion of the lightweight model then .type(torch.float)/255).
        If there are multiple croppeds, they should be sent as a list of tensors using cat: self.lightweight(torch.cat(croppeds,dim=0))
        Args:
            images: The list of images to be embedded as torch tensors.
        Returns:
            embed_list: The list of embeddings as torch tensors with the shape [batch, embedding_dim].
        """
        if phase ==1:
            lightweight_model=self.lightweight_phase1
        elif phase ==2:
            lightweight_model=self.lightweight_phase2
        images_transformed = torch.cat([self.transform(img.type(torch.float)/255) for img in images], dim=0)
        with torch.no_grad():
            embed_list = lightweight_model(images_transformed)
        return embed_list  
    

    def compare_disk_feats(self,image,feats,candidates,thresholds,NUM_PATCH: int=1,device='cuda',rotate: bool=True):
        #NUM_PATCH=NUM_PATCH[0]
        if NUM_PATCH == 1:
            topk = 4
        if NUM_PATCH == 2:
            topk = 6
        if NUM_PATCH == 3:
            topk = 9
        scores_threshold1=[]
        scores_threshold2=[]
        scores_threshold3=[]
        #candidates = os.listdir(database_path)
        candidates_ids = ['_'.join(i.split('_')[0:5]) for i in candidates]

        for cand, cand_id in zip(candidates,candidates_ids): 
            #if cand_id != '20251117193451_7DD8_48DB5658_I001_Bottom1':
            #    continue
            feats0=load_file(DATABASE_PATH+cand)['array'].to(device)
            max_score = 0
            for i,threshold in enumerate(thresholds):
                topk_scores = []
                for feat in feats:
                    calculations = torch.einsum('pkd,ncd->pnkc',feat,feats0)
                    score=F.relu(calculations-threshold).max(dim=-1).values.sum(dim=-1)
                    topk_score = score.max(dim=-1).values.topk(topk).values.sum().item()
                    topk_scores.append(topk_score)

                max_score = max(topk_scores)
                if i == 0:
                    scores_threshold1.append(max_score)
                if i == 1:
                    scores_threshold2.append(max_score)
                if i == 2:
                    scores_threshold3.append(max_score)
        top5_threshold1 = sorted(scores_threshold1, reverse=True)[:5]
        max_val = top5_threshold1[0] if top5_threshold1 else 0
        conf1 = 0 if max_val == 0 else max_val / sum(top5_threshold1)

        
        top5_threshold2 = sorted(scores_threshold2, reverse=True)[:5]
        max_val2 = top5_threshold2[0] if top5_threshold2 else 0
        conf2 = 0 if max_val2 == 0 else max_val2 / sum(top5_threshold2)

        top5_threshold3 = sorted(scores_threshold3, reverse=True)[:5]
        max_val3 = top5_threshold3[0] if top5_threshold3 else 0
        conf3 = 0 if max_val3 == 0 else max_val3 / sum(top5_threshold3)


        confidences = torch.tensor([conf1, conf2, conf3])  #solved
        confidences = torch.where(torch.isnan(confidences), torch.tensor(float('-inf')), confidences)
        scores = [scores_threshold1,scores_threshold2,scores_threshold3][torch.argmax(confidences)]

        return scores
    
    def query_lightglue(self,feats,candidates):
        candidates_ids = ['_'.join(i.split('_')[0:5]) for i in candidates]
        scores = []
        for cand, cand_id in zip(candidates,candidates_ids): 
            feats0=load_file(DATABASE_PATH_LIGHTGLUE+cand)
            max_score = 0
            feats0['keypoints'] = feats0['keypoints'].cuda()
            feats0['keypoint_scores'] = feats0['keypoint_scores'].cuda()
            feats0['descriptors'] = feats0['descriptors'].cuda()
            max_score = 0
            for feat in feats:
                # match the features
                with torch.no_grad():
                    matches01 = self.matcher({'image0': feats0, 'image1': feat})
                    feats0_, feats1, matches01 = [rbd(x) for x in [feats0, feat, matches01]]  # remove batch dimension
                matches = matches01['matches']  # indices with shape (K,2)
                points0 = feats0_['keypoints'][matches[..., 0]]  # coordinates in image #0, shape (K,2)
                points1 = feats1['keypoints'][matches[..., 1]]  # coordinates in image #1, shape (K,2)
                score = len(points1)
                if score >= max_score:
                    max_score = score
            scores.append(max_score)
        return scores



DEVICE = settings.DEVICE
QDRANT_HOST = settings.QDRANT_HOST
DB_COLLECTION_NAME_PHASE1 = settings.DB_COLLECTION_NAME_PHASE1
DB_COLLECTION_NAME_PHASE2 = settings.DB_COLLECTION_NAME_PHASE2

DB_LIMIT_PHASE1 = int(settings.DB_LIMIT_PHASE1)
DB_LIMIT_PHASE2 = int(settings.DB_LIMIT_PHASE2)
DB_DIMENSIONALITY_PHASE1 = int(settings.DB_DIMENSIONALITY_PHASE1)
DB_DIMENSIONALITY_PHASE2 = int(settings.DB_DIMENSIONALITY_PHASE2)
DB_TYPE = settings.DB_TYPE
SAVE_INPUT_QUERY = settings.SAVE_INPUT_QUERY
INPUT_QUERY_PATH = settings.INPUT_QUERY_PATH
DATABASE_PATH = settings.DATABASE_PATH
DATABASE_PATH_LIGHTGLUE = settings.DATABASE_PATH_LIGHTGLUE
NUM_PATCH = int(settings.NUM_PATCH)
DISK_KEYPOINTS = int(settings.DISK_KEYPOINTS)
THRESHOLDS = settings.THRESHOLDS
EXTRACTOR_MODEL = settings.EXTRACTOR_MODEL
EMBEDDING_MODEL_PHASE1 = settings.EMBEDDING_MODEL_PHASE1
EMBEDDING_MODEL_PHASE2 = settings.EMBEDDING_MODEL_PHASE2
EMBEDDING_MODEL_PATH_PHASE1 = settings.EMBEDDING_MODEL_PATH_PHASE1
EMBEDDING_MODEL_PATH_PHASE2 = settings.EMBEDDING_MODEL_PATH_PHASE2

if SAVE_INPUT_QUERY == "False":
    SAVE_INPUT_QUERY = False
else:
    SAVE_INPUT_QUERY = True
    

logging.info(f'DB_TYPE: {DB_TYPE}')
logging.info(f'DB_LIMIT_PHASE1: {DB_LIMIT_PHASE1}')
logging.info(f'DB_LIMIT_PHASE2: {DB_LIMIT_PHASE2}')
logging.info(f'DATABASE_PATH: {DATABASE_PATH}')
logging.info(f'DATABASE_PATH_LIGHTGLUE: {DATABASE_PATH_LIGHTGLUE}')
logging.info(f'DB_COLLECTION_NAME_PHASE1: {DB_COLLECTION_NAME_PHASE1}')
logging.info(f'DB_COLLECTION_NAME_PHASE2: {DB_COLLECTION_NAME_PHASE2}') 
logging.info(f'DB_DIMENSIONALITY_PHASE1: {DB_DIMENSIONALITY_PHASE1}')
logging.info(f'DB_DIMENSIONALITY_PHASE2: {DB_DIMENSIONALITY_PHASE2}')
logging.info(f'QDRANT_HOST: {QDRANT_HOST}')
logging.info(f'NUM_PATCH: {NUM_PATCH}')
logging.info(f'DISK_KEYPOINTS: {DISK_KEYPOINTS}')
logging.info(f'THRESHOLDS: {THRESHOLDS}')
logging.info(f'EXTRACTOR_MODEL: {EXTRACTOR_MODEL}')


db = vector_db(collection_name_phase1=DB_COLLECTION_NAME_PHASE1, 
               collection_name_phase2=DB_COLLECTION_NAME_PHASE2, 
               extractor_model=EXTRACTOR_MODEL, keypoints=DISK_KEYPOINTS)

if __name__ == "__main__":

    
    # Get host and port from environment variables with secure defaults
    host = os.getenv("APP_HOST", "127.0.0.1")
    port = int(os.getenv("APP_PORT", 8003))
    workers = int(os.getenv("APP_WORKERS", 1))

    
    

    uvicorn.run(
        "retrieve:app", host=host, port=port, reload=False, loop="asyncio", workers=workers
    )
    