from io import BytesIO
import aiofiles
from datetime import datetime
import torch
import os
import logging
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import uvicorn
from fastapi import FastAPI, Form, HTTPException, Request, Response, File, UploadFile, Body
from fastapi.responses import StreamingResponse, JSONResponse
import json
from torchvision.transforms.functional import pil_to_tensor
from helpers.config import get_settings
from torchvision.transforms import functional as torchvision_F

settings = get_settings()

from retrieve import vector_db

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

app = FastAPI()

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})

@app.get("/health")
def health_check():
    return {"status": "healthy", "version": "1.0.0"}



@app.post("/add_descriptors_to_database")
async def add_descriptors_to_database(image_file: UploadFile = File(...)):
    file_name = image_file.filename
    contents = await image_file.read() 
    pil_image = Image.open(BytesIO(contents)).convert("RGB")
    tensor_org = pil_to_tensor(pil_image).unsqueeze(0)
    db.extract_descriptors(image=tensor_org,file_name=file_name,SAVE_FEATS=True)
    #db.add_to_database(embed_list,feats,paths_list)
    return {"status": "Item was added successfully"}

@app.post("/add_lightglue_to_database")
async def add_lightglue_to_database(image_file: UploadFile = File(...)):
    file_name = image_file.filename
    contents = await image_file.read() 
    pil_image = Image.open(BytesIO(contents)).convert("L")
    tensor_org = pil_to_tensor(pil_image).unsqueeze(0)
    db.extract_lightglue_featutres(image=tensor_org,file_name=file_name,SAVE_FEATS=True)
    #db.add_to_database(embed_list,feats,paths_list)
    return {"status": "Item was added successfully"}

@app.post('/add_embedding_to_database')
async def add_embedding_to_database(image_file: UploadFile = File(...), phase: int = 2):
    file_name = image_file.filename
    contents = await image_file.read() 
    pil_image = Image.open(BytesIO(contents)).convert("RGB")
    tensor_org = pil_to_tensor(pil_image).unsqueeze(0)
    embed = db.embed([tensor_org], phase=phase)
    db.add_to_database(embed.tolist()[0], path='N/A', file_name=file_name, phase=phase)
    return {"status": "Embedding was added successfully"}

@app.post("/query_lightweight")
async def query_lightweight(image_file: UploadFile = File(...), phase: int = 2):
    """
    Make a query with vector database by sending an image.

    Args:
        image_file: The image from any local directory
    Returns:
        dict: Contains the ID and the path of the top k images
    """
    file_name = image_file.filename
    contents = await image_file.read() 
    pil_image = Image.open(BytesIO(contents)).convert("L")
    tensor_org = pil_to_tensor(pil_image).unsqueeze(0).repeat(1, 3, 1, 1) 


    embed = db.embed([tensor_org],phase=phase)
    results = db.query(embed[0].tolist())
    return {"results": results}

@app.post("/query_with_vectordb")
async def make_query_lightweight(image_file: UploadFile = File(...)):
    """
    Make a full query by sending an image and a metadata json file.

    Args:
        image_file: The image from any local directory
        metadata: The metadata of the parcel as a Json file

    Returns:
        dict: Contains the ID and the path of the top 1 image
    """
    file_name = image_file.filename
    contents = await image_file.read() 
    image_path=None
    if SAVE_INPUT_QUERY:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        upload_folder = os.path.join(INPUT_QUERY_PATH, timestamp)
        os.makedirs(upload_folder, exist_ok=True)

        # Save image file
        image_file_name = image_file.filename
        image_path = os.path.join(upload_folder, image_file_name)
        async with aiofiles.open(image_path, 'wb') as out_file:
            await out_file.write(contents)

    feats = db.extract_descriptors(image=BytesIO(contents),query=True)

    #embed_list_all, croppeds_all, paths_all = db.embed(content=BytesIO(contents),file_name=file_name,save_cutted=image_path)
    #feats = db.extract_descriptors(croppeds_all=croppeds_all,query=True)


    return {"status": 'Ok'}


@app.post('/query_with_multiplication')
async def query_with_multiplication(image_file: UploadFile = File(...)):
    file_name = image_file.filename
    contents = await image_file.read() 
    
    candidates = os.listdir(DATABASE_PATH)
    pil_image = Image.open(BytesIO(contents)).convert("L")
    tensor_org = pil_to_tensor(pil_image).unsqueeze(0)#.repeat(1, 3, 1, 1) 

    feats = db.extract_descriptors(image=tensor_org,file_name=file_name,SAVE_FEATS=False,rotate=True)
    scores = db.compare_disk_feats(image=tensor_org,feats=feats,candidates=candidates,thresholds=THRESHOLDS,NUM_PATCH=NUM_PATCH,device=DEVICE)
    return {"candidates": candidates, "scores": [i for i in scores]}


@app.post('/query_with_lightglue')
async def query_with_lightglue(image_file: UploadFile = File(...)):
    file_name = image_file.filename
    contents = await image_file.read() 
    
    candidates = os.listdir(DATABASE_PATH_LIGHTGLUE)
    pil_image = Image.open(BytesIO(contents)).convert("L")
    tensor_org = pil_to_tensor(pil_image).unsqueeze(0)#.repeat(1, 3, 1, 1) 

    feats = db.extract_lightglue_featutres(image=tensor_org,file_name=file_name,SAVE_FEATS=False,rotate=False)
    scores = db.query_lightglue(image=tensor_org,feats=feats,candidates=candidates,thresholds=THRESHOLDS,NUM_PATCH=NUM_PATCH,device=DEVICE)
    return {"candidates": candidates, "scores": [i for i in scores]}

def combine_qdrant_results(results, results_rot, DB_LIMIT):
    combined_results1 = {}
    combined_results1['ids'] = []
    combined_results1['file_name'] = []
    combined_results1['path'] = []
    combined_results1['point_num'] = []
    combined_results1['scores'] = []
    for i in range(len(results["ids"])):
        item_id = results["ids"][i]
        score = results["scores"][i]
        if item_id not in combined_results1['ids']:
            combined_results1['ids'].append(item_id)
            combined_results1['file_name'].append(results["file_name"][i])
            combined_results1['path'].append(results["path"][i])
            combined_results1['point_num'].append(results["point_num"][i])
            combined_results1['scores'].append(score)
        elif score > combined_results1["scores"][i]:
            combined_results1['ids'].append(item_id)
            combined_results1['file_name'].append(results["file_name"][i])
            combined_results1['path'].append(results["path"][i])
            combined_results1['point_num'].append(results["point_num"][i])
            combined_results1['scores'].append(score)

    for i in range(len(results_rot["ids"])):
        item_id = results_rot["ids"][i]
        score = results_rot["scores"][i]
        if item_id not in combined_results1['ids']:
            combined_results1['ids'].append(item_id)
            combined_results1['file_name'].append(results_rot["file_name"][i])
            combined_results1['path'].append(results_rot["path"][i])
            combined_results1['point_num'].append(results_rot["point_num"][i])
            combined_results1['scores'].append(score)
        elif score > combined_results1["scores"][i]:
            combined_results1['ids'].append(item_id)
            combined_results1['file_name'].append(results_rot["file_name"][i])
            combined_results1['path'].append(results_rot["path"][i])
            combined_results1['point_num'].append(results_rot["point_num"][i])
            combined_results1['scores'].append(score)
    
    sorted_idx = sorted(
    range(len(combined_results1["scores"])),
    key=lambda i: combined_results1["scores"][i],
    reverse=True
    )[:DB_LIMIT]

    # Reorder each field using the sorted indices
    results = {
        key: [combined_results1[key][i] for i in sorted_idx]
        for key in combined_results1
    }

    return results


@app.post('/full_query')
async def full_query(image_file_phase1: UploadFile = File(...), 
                     image_file_phase2: UploadFile = File(...),
                     rotate_embeddding_phase1: bool = Form(False),
                     rotate_embeddding_phase2: bool = Form(False),
                     rotate_lightglue: bool = Form(False)):
    """
    Make a full query with vector database and disk/superpoint features by sending an image.

    Args:
        image_file: The image from any local directory
    Returns:
        dict: Contains the ID and the path of the top k images
    """
    # Read image file
    file_name_phase1 = image_file_phase1.filename
    contents_phase1 = await image_file_phase1.read() 
    file_name_phase2 = image_file_phase2.filename
    contents_phase2 = await image_file_phase2.read()


    # Read image and convert to tensor
    pil_image_phase1 = Image.open(BytesIO(contents_phase1)).convert("L")
    tensor_org_phase1 = pil_to_tensor(pil_image_phase1).unsqueeze(0).repeat(1, 3, 1, 1) 
    pil_image_phase2 = Image.open(BytesIO(contents_phase2)).convert("L")
    tensor_org_phase2 = pil_to_tensor(pil_image_phase2).unsqueeze(0).repeat(1, 3, 1, 1) 

    # Get embedding from lightweight model
    embed_phase1 = db.embed([tensor_org_phase1], phase=1)
    embed_phase2 = db.embed([tensor_org_phase2], phase=2)

    # Query phase 1
    results1 = db.query(embed_phase1[0].tolist())

    if rotate_embeddding_phase1:
        tensor_org_phase1_rot=torchvision_F.rotate(tensor_org_phase1,180)
        embed_phase1_rot = db.embed([tensor_org_phase1_rot], phase=1)
        results1_rot = db.query(embed_phase1_rot[0].tolist())
        results1 = combine_qdrant_results(results1, results1_rot, DB_LIMIT_PHASE1)
    file_names1 = results1['file_name']
    image_ids = [i.split('.')[0] for i in file_names1]

    # Query phase 2
    results2 = db.query(embed_phase2[0].tolist(),phase=2,candidates=image_ids)
    if rotate_embeddding_phase2:
        tensor_org_phase2_rot=torchvision_F.rotate(tensor_org_phase2,180)
        embed_phase2_rot = db.embed([tensor_org_phase2_rot], phase=2)
        results2_rot = db.query(embed_phase2_rot[0].tolist(),phase=2,candidates=image_ids)
        results2 = combine_qdrant_results(results2, results2_rot, DB_LIMIT_PHASE2)
    file_names2 = results2['file_name']
    
    del embed_phase1
    del embed_phase2
    torch.cuda.empty_cache()      # releases cached memory (not allocated memory)
    torch.cuda.synchronize()
    # Query lightglue features
    candidates = [fn.split('.')[0] + '.safetensors' for fn in file_names2]
    #candidates = [i.split('_between_')[0] + '.safetensors' for i in candidates]
    feats = db.extract_lightglue_featutres(image=tensor_org_phase2,file_name=file_name_phase2,SAVE_FEATS=False,rotate=True,resize=False)
    #scores, points_cands, points_q = db.query_lightglue(feats=feats,candidates=candidates)
    scores = db.query_lightglue(feats=feats,candidates=candidates)
    del feats
    if rotate_lightglue:
        feats_rot = db.extract_lightglue_featutres(image=tensor_org_phase2_rot,file_name=file_name_phase2,SAVE_FEATS=False,rotate=True,resize=False)
        scores_rot = db.query_lightglue(feats=feats_rot,candidates=candidates,thresholds=THRESHOLDS,NUM_PATCH=NUM_PATCH,device=DEVICE)
        scores = [max(score,score_rot) for score,score_rot in zip(scores,scores_rot)]
    
    torch.cuda.empty_cache()      # releases cached memory (not allocated memory)
    torch.cuda.synchronize()
    #return {"qdrant_results_phase1": results1, "qdrant_results_phase2": results2, "disk_scores": scores, "points_q":[i.tolist() for i in points_q]}
    return {"qdrant_results_phase1": results1, "qdrant_results_phase2": results2, "disk_scores": scores}





DEVICE = settings.DEVICE
QDRANT_HOST = settings.QDRANT_HOST
DB_COLLECTION_NAME_PHASE1 = settings.DB_COLLECTION_NAME_PHASE1
DB_COLLECTION_NAME_PHASE2 = settings.DB_COLLECTION_NAME_PHASE2
DB_LIMIT_PHASE1 = settings.DB_LIMIT_PHASE1
DB_LIMIT_PHASE2 = settings.DB_LIMIT_PHASE2
DB_DIMENSIONALITY_PHASE1 = settings.DB_DIMENSIONALITY_PHASE1
DB_DIMENSIONALITY_PHASE2 = settings.DB_DIMENSIONALITY_PHASE2
DB_TYPE = settings.DB_TYPE
SAVE_INPUT_QUERY = settings.SAVE_INPUT_QUERY
INPUT_QUERY_PATH = settings.INPUT_QUERY_PATH
DATABASE_PATH = settings.DATABASE_PATH
DATABASE_PATH_LIGHTGLUE = settings.DATABASE_PATH_LIGHTGLUE
NUM_PATCH = settings.NUM_PATCH
DISK_KEYPOINTS = settings.DISK_KEYPOINTS
THRESHOLDS = settings.THRESHOLDS
EXTRACTOR_MODEL = settings.EXTRACTOR_MODEL
EMBEDDING_MODEL_PHASE1 = settings.EMBEDDING_MODEL_PHASE1
EMBEDDING_MODEL_PATH_PHASE1 = settings.EMBEDDING_MODEL_PATH_PHASE1
EMBEDDING_MODEL_PHASE2 = settings.EMBEDDING_MODEL_PHASE2
EMBEDDING_MODEL_PATH_PHASE2 = settings.EMBEDDING_MODEL_PATH_PHASE2
NUM_PATCH = settings.NUM_PATCH

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
logging.info(f'EMBEDDING_MODEL_PHASE1: {EMBEDDING_MODEL_PHASE1}')
logging.info(f'EMBEDDING_MODEL_PATH_PHASE1: {EMBEDDING_MODEL_PATH_PHASE1}')
logging.info(f'EMBEDDING_MODEL_PHASE2: {EMBEDDING_MODEL_PHASE2}')
logging.info(f'EMBEDDING_MODEL_PATH_PHASE2: {EMBEDDING_MODEL_PATH_PHASE2}')

db = vector_db(collection_name_phase1=DB_COLLECTION_NAME_PHASE1, 
               collection_name_phase2=DB_COLLECTION_NAME_PHASE2, 
               extractor_model=EXTRACTOR_MODEL, keypoints=DISK_KEYPOINTS,
               lightweight_model_phase1=EMBEDDING_MODEL_PHASE1, 
               lightweight_model_path_phase1=EMBEDDING_MODEL_PATH_PHASE1,
               lightweight_model_phase2=EMBEDDING_MODEL_PHASE2, 
               lightweight_model_path_phase2=EMBEDDING_MODEL_PATH_PHASE2,
               database_path_lightglue=DATABASE_PATH_LIGHTGLUE)
