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
import pandas as pd
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


dataset_path = os.getenv("LP_MORPH_PATH")
dataset_base_file = "morph_2008_nonCommercial.csv"

dataset_source = "morph"

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

    # def _split_generators(self):
    #     return super()._split_generators()

    def _split_generators(self, dl_manager: datasets.DownloadManager):
        # TODO: a way to pass the number of shards as argument? 
        num_shards = 16
        
        self.base_path = dataset_path
        # Generating list of jobs per identity
        self.shards = list()

        self.dataset = pd.read_csv(os.path.join(dataset_path, dataset_base_file))
        len_dataset = len(self.dataset)

        indices = list()
        step = len_dataset // num_shards
        for i in range(num_shards):
            lower = i*step
            upper = min((i+1)*step, len_dataset)
            indices.append((lower, upper))        
        
        # Ethnicity based parallelization ... s
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"indices": indices}),
        ]   

    def _generate_examples(self, indices):
        shard = indices[0]

        for row in self.dataset.iloc[shard[0]:shard[1]].iterrows() :
            entry = row[1]
            #print(entry)

            #MORPH's stupid hierarchy
            images_parent_folder = "Album2"
            image_subfolder = entry["photo"].split('/')[-1][:3]
            image_name = entry["photo"].split('/')[-1]            
            image_path = os.path.join(dataset_path, images_parent_folder, image_subfolder, image_name)
            
            age = int(entry["age"])
            gender = gender_map[entry["gender"].upper()]
            race = race_map[entry["race"].upper()]

            unique_id = hashlib.sha256(image_path.encode('utf-8')).hexdigest()[:10] 

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
                "person_id": dataset_source + naming_splitter + str(entry["id_num"]),

                "source": dataset_source,
                "image": image_data.getvalue(),
                "dlib_align_status": image_dlib_aligned,
                "image_dlib_aligned": processed_image_data.getvalue(),

                # "attributes": {
                #     "gender": gender,
                #     "race": race, 
                #     "age": age,
                #     "human": True
                # }

                "gender": gender,
                "race": race, 
                "age": age,
                "human": True
            }

            yield unique_id, data

                        


                        


        

        
