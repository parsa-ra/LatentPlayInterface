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

import importlib
import hashlib

import traceback 
import sys 
import pathlib
from multiprocessing import cpu_count

file_path = pathlib.Path().resolve()
print(f"Adding {file_path} to the system path")
sys.path.append(file_path)
from loader_configs import *


num_shards = max(1, cpu_count()//2)

######################## CONFIGS ###########################


dataset_path = os.getenv("LP_UTKFACE_PATH")
print("\n########"*5)
print(dataset_path)
dataset_source = "utkface"

######################## CONFIGS ###########################

logger = get_logger("LatentPlayDataset")
dataset_path = dataset_path
predictor = dlib.shape_predictor(dlib_landmark_detector_path)
gender_map = gender_map
race_map = race_map
preprocessing_target_size = preprocessing_target_size
source = dataset_source

error_file = open(f"{dataset_source}_failed.txt", mode="a")

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

        self.dataset_base_path = dataset_path
        parts = [os.path.join(self.dataset_base_path, f"part{i}") for i in [1,2,3]]
        self.images_list = list()
        extensions = ["jpg", "png", "jpeg"]
        for part in parts: 
            for extension in extensions: 
                print(f"Scanning for {extension} images in {part} directory")
                glob_path = os.path.join(part, "**", f"*.{extension}")
                print(glob_path)
                found_images = glob(glob_path, recursive=True)
                print(f"Found {len(found_images)} images ... ")
                self.images_list.extend(found_images)
        
        self.sample_lengths = len(self.images_list)
        step_size = self.sample_lengths // num_shards
        indices = list()
        for i in range(num_shards):
            lower = i*step_size
            upper = min((i+1)*step_size, self.sample_lengths)
            print(lower, upper)
            indices.append(np.arange(lower, upper, 1))

        # Ethnicity based parallelization ...
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"indices": indices}),
        ]   
    
    def _generate_examples(self, indices):
        indices = indices[0]      
        for index in indices:
            image_path = self.images_list[index]
            image_name = image_path.split('/')[-1]
            image_name_parts = image_name.split("_")
            try:
                age = image_name_parts[0]
                gender = image_name_parts[1]
                race = image_name_parts[2]

                race = race_map[race]
                gender = gender_map[gender]
            except: 
                print(age, gender, race)
                continue


            unique_id = hashlib.sha256(image_path.encode('utf-8')).hexdigest()[:10] + str(index)

            image_data = io.BytesIO()

            try: 
                image_file = Image.open(image_path)
                if image_file.mode != "RGB": 
                    print(f"\n\n\nImage mode of {image_path} is not RGB ... \n\n\ns")
                    image_file = image_file.convert(mode="RGB")

                image_file.save(image_data, "JPEG")
            except: 
                logger.error("Can't generate processed aligned image for {}\n{}".format(image_path, traceback.format_exc()))
                error_file.write(f"{image_path}\n")
                continue

            processed_image_data = io.BytesIO()
            image_process_status = False
            try: 
                align_face(image_path, predictor, preprocessing_target_size).save(processed_image_data, "JPEG")
                image_process_status = True
            except: 
                logger.error("Can't generate processed aligned image for {}\n{}".format(image_path, traceback.format_exc()))
        
            yield unique_id, {
                "person_id": image_name,
                #"image_path": image_path,
                "image": image_data.getvalue(),
                "dlib_align_status": image_process_status,
                "image_dlib_aligned": processed_image_data.getvalue(),
                "race": race, 
                "gender": gender,
                "human": True,
                "age": age,
            }
 
                                               
