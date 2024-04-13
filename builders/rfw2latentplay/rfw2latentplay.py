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


num_shards = max(1, cpu_count()//2)

######################## CONFIGS ###########################


dataset_path = os.getenv("LP_RFW_PATH")

dataset_source = "rfw"

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
        # TODO: a way to pass the number of shards as argument? 
        
        self.base_path = dataset_path
        # Generating list of jobs per identity
        self.shards = list()
        for race_dir in os.listdir(self.base_path): 
            logger.info("Working on {} Directory ... ".format(race_dir))
            for id in os.listdir(os.path.join(self.base_path, race_dir)):
                self.shards.append({
                    "race": race_dir,
                    "id": id, 
                    "images_folder": os.path.join(self.base_path, race_dir, id)
                })

        self.sample_lengths = len(self.shards)
        step_size = self.sample_lengths // num_shards
        indices = list()
        for i in range(num_shards):
            lower = i*step_size
            upper = min((i+1)*step_size, self.sample_lengths)
            print(lower, upper)
            indices.append(np.arange(lower, upper, 1))


        # Ethnicity based parallelization ... s
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"indices": indices}),
        ]   

    def _generate_examples(self, indices):
        indices = indices[0]        
        for index in indices: 
            entry = self.shards[index]
            print(entry)

            for image in os.listdir(entry["images_folder"]):

                image_path = os.path.join(entry["images_folder"], image) 
                
                age = "unknown"
                gender = gender_map["unknown"]
                race = race_map[entry["race"].lower()]

                unique_id = hashlib.sha256(image_path.encode('utf-8')).hexdigest()[:10] + str(index)

                image_data = io.BytesIO()
                try:
                    Image.open(image_path).save(image_data, "JPEG")
                except: 
                    logger.error("Something went wrong loading file {}".format(image_path))
                    image_data = io.BytesIO()

                processed_image_data = io.BytesIO()
                image_dlib_aligned = False
                try:
                    align_face(image_path, predictor, preprocessing_target_size).save(processed_image_data, "JPEG")
                    image_dlib_aligned = True
                except: 
                    logger.error("Can't generate processed aligned image for {}\n{}".format(image_path, traceback.format_exc()))
            
                data = {
                    "person_id": dataset_source + naming_splitter + entry["id"],

                    "source": dataset_source,
                    "image": image_data.getvalue(),
                    "dlib_align_status": image_dlib_aligned,
                    "image_dlib_aligned": processed_image_data.getvalue(),

                    "gender": gender,
                    "race": race, 
                    "age": age,
                    "human": True
                }

                yield unique_id, data

                        


                        


        

        
