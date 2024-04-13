import datasets
from datasets.utils.logging import get_logger
from datasets import Sequence
import io
import os 
from glob import glob
from tqdm import tqdm
import numpy as np
import pandas as pd
from PIL import Image
import scipy
import scipy.ndimage
import dlib
import threading 
from multiprocessing import cpu_count
import importlib
import hashlib

import traceback 
import sys 
import pathlib

file_path = pathlib.Path().resolve()
print(f"Adding {file_path} to the system path")
sys.path.append(file_path)
from loader_configs import *

ALIGN_FACE = False 

######################## CONFIGS ###########################


dataset_path = os.getenv("LP_FAIRFACE_PATH")
dataset_path_fallback = os.getenv("LP_DATSET_BASE_PATH")
if dataset_path is None:
    dataset_path = dataset_path_fallback
print(dataset_path)

csv_paths = ["fairface_label_train.csv", "fairface_label_val.csv"]

dataset_source = "fairface"

######################## CONFIGS ###########################

logger = get_logger("LatentPlayDataset")
dataset_path = dataset_path
predictor = dlib.shape_predictor(dlib_landmark_detector_path)
gender_map = gender_map
race_map = race_map
preprocessing_target_size = preprocessing_target_size
source = dataset_source


class LatentPlayDataset(datasets.GeneratorBasedBuilder):
    """LatentPlayDataset base builder """


    def _info(self):
        return datasets.DatasetInfo(
        description="Dataset of multiple human face images with attributes including race and gender",
        features=datasets.Features(
            dataset_features
        ),
        supervised_keys=None,
        homepage="latentplay",
        citation="",
        version="0.5.1"
    )

    def _split_generators(self, dl_manager: datasets.DownloadManager):
        num_shards = max(1, cpu_count()//2)
        self.dataset_base_path = dataset_path
        self.csv_sources = pd.DataFrame()
        for csv_path in csv_paths: 
            self.csv_sources = pd.concat((pd.read_csv(os.path.join(dataset_path, csv_path)), self.csv_sources))

        self.sample_lengths = self.csv_sources.shape[0]
        print("#\n"*3, self.sample_lengths)
        step_size = self.sample_lengths // num_shards
        indices = list()
        for i in range(num_shards):
            lower = i*step_size
            upper = min((i+1)*step_size, self.sample_lengths)
            print(lower, upper)
            indices.append(np.arange(lower, upper, 1))

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"indices": indices}),
        ]   

    def _generate_examples(self, indices):
        kwargs = {}
        indices = indices[0]        
        for index in indices: 
            data = self.csv_sources.iloc[index]
            image_path = os.path.join(self.dataset_base_path, data['file']) 
            age = data['age']
            gender = data['gender']
            race = data['race']

            race = race_map[race]
            gender = gender_map[gender]

            unique_id = hashlib.sha256(image_path.encode('utf-8')).hexdigest()[:10] + str(index)

            image_data = io.BytesIO()
            Image.open(image_path).save(image_data, "JPEG")
        
            processed_image_data = io.BytesIO()
            image_process_status = False
            if ALIGN_FACE:
                try:
                    align_face(image_path, predictor, preprocessing_target_size).save(processed_image_data, "JPEG")
                    image_process_status = True
                except: 
                    logger.error("Can't generate processed aligned image for {}\n{}".format(image_path, traceback.format_exc()))

                kwargs={
                    "image_dlib_aligned": processed_image_data.getvalue(),
                    "dlib_align_status": image_process_status,
                }

            yield unique_id, {
                "person_id": dataset_source + naming_splitter + image_path.replace('/', '-'),
                "source": dataset_source,
                "image": image_data.getvalue(),
                "race": race, 
                "gender": gender,
                "human": True,
                "age": age,
                **kwargs
            }

                        


                        


        

        
