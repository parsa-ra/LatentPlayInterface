from enum import unique
from utils import get_logger
from sklearn.manifold import TSNE
from bokeh.plotting import figure, show
from bokeh.transform import factor_cmap, factor_mark
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource
from typing import List
from copy import copy, deepcopy
from joblib import dump, load
from glob import glob
import numpy as np 
import pandas as pd 
from datasets import load_from_disk, Dataset
from copy import copy 
import matplotlib.pyplot as plt
import numpy as np
import matplotlib 
import hydra
from omegaconf import DictConfig, OmegaConf
from multiprocessing import cpu_count

import os

logger = get_logger("Manifold", True, False)

experiments_base_path = "./Experiments"
projection_base_path = os.path.join(experiments_base_path, 'projection')

fw, fh = 1200, 1200

np.random.seed(13)

def generate_distinct_colors(n):
    colors = plt.cm.get_cmap("hsv", n)
    return [matplotlib.colors.rgb2hex(colors(i)[:3]) for i in range(n)]


MARKERS = ['hex', 'circle_x', 'triangle', 'square', 'circle', 'dot', 'asterisk', 'diamond']
COLORS = generate_distinct_colors(20)
np.random.shuffle(COLORS)
print(COLORS)

change_label_name = False
embeddings = None
metas = None
FORCE_RERUN = False

print("Creating projection base path ... ")
os.makedirs(projection_base_path, exist_ok=True)

colormap = dict()
markermap = dict()

@hydra.main(version_base=None, config_path="config/config_vis", config_name="config.yaml")
def tsne(cfg: DictConfig):
    cfg = cfg.vis

    name = cfg.name

    dataset_full_path = cfg.dataset_full_path
    print(dataset_full_path)

    embedding_key = cfg.embedding_key
    label_keys = cfg.label_keys
 
    assert len(label_keys) <= 2 , "Right know for tnsne we can only show two label at a time"
    label_map = dict()

    label_precedence = ["colormap", "markermap"]
    value_precedence = [ "COLORS", "MARKERS" ]
    values = ["color", "marker"]

    dataset: Dataset = load_from_disk(dataset_full_path)
    dataset_keys = list()
    dataset_keys.append(embedding_key)
    dataset_keys.extend(label_keys)

    #prefix_name= dataset['train']._fingerprint # Inferfrom name or fingerprint
    prefix_name = f"{cfg.proj_algorithm.upper()}s"
    exp_name = "-".join([prefix_name, embedding_key, *label_keys, dataset['train']._fingerprint])
    
    status_prefix = "status"
    for key in dataset:
        status_key_name = status_prefix + "_"+embedding_key
        dataset_check = dataset[key].select_columns(column_names=[embedding_key, status_key_name])
        dataset_check = dataset_check.to_pandas()
        print(dataset_check[status_key_name].value_counts().to_dict())

    
    if cfg.discard: 
        print(f"Filtering enteries with their {cfg.discard + embedding_key} set to true ...")
        dataset = dataset.filter(lambda entry: entry[ cfg.discard + embedding_key], 
        num_proc=min(1,cpu_count()//2)) # There is some problem with the datasets' filtering process with multiple processes
        print("Filtering finished.")

    dataset.set_format("numpy", columns=dataset_keys)

    if 'test' not in dataset and 'train' in dataset: 
        logger.warn('Test shard is not in the dataset, will split the dataset automatically')
        dataset = dataset['train'].train_test_split(test_size=0.1)
    else: 
        logger.error('Unknown split names, the dataset should either contain an `train` or `test`split')
    
    train_embeddings = dataset['train'][embedding_key]
    test_embeddings = dataset['test'][embedding_key]

    test_labels = list()

    for label_key in label_keys: 
        test_labels.append(dataset['test'][label_key])

    unique_keys: List[List] = list()

    for label_idx, label_key in enumerate(label_keys):
        unique_keys.append(list())

        cur_labels = test_labels[label_idx]
        for cur_label in cur_labels: 
            if cur_label not in unique_keys[-1]: 
                print(f"Adding {cur_label} for {label_key}")
                unique_keys[-1].append(cur_label)


    # import sys  
    print("\n\n\n", unique_keys)
    max_unique_key_len = 0 
    for unique_key in unique_keys:
        max_unique_key_len = max(len(unique_key), max_unique_key_len)
    print(max_unique_key_len, "\n\n\n")
    # sys.exit(0)

    plots = list()
        
    title="-".join([prefix_name, embedding_key, *label_keys])
    segmented_title = ""
    cur_pos = 0 
    pos_step = 25
    while cur_pos < len(title):
        segmented_title += title[cur_pos: cur_pos+pos_step] + "\n"
        cur_pos += pos_step
 
 
    print(f"Original Title was: {title}\nSegmented title is: {segmented_title}\n")
    p = figure(title = segmented_title, frame_width=fw, frame_height=fh, background_fill_color="#fafafa")
   
    cur_embeddings = train_embeddings
    projector_name = exp_name
    projector_path = os.path.join(projection_base_path ,f"{projector_name}_manifoldlearning.pickle")
    projector_sklearn_api = None
    if os.path.exists(projector_path) and not FORCE_RERUN:
        logger.info(f"It seems that the projector file for current config already exists in {projector_path}, ... Loading it from disk instead of re-running the tsne")
        with open(projector_path, 'rb') as file:
            projector_sklearn_api  = load(file)
    else: 
        logger.info(f"The tsne file on disk cannot be found at {projector_path}, running the tsne training ... ")

        if cfg.proj_algorithm == "tsne":
            logger.info(f"Setting projection algorithm to TSNE")
            projector_sklearn_api = TSNE(
                n_components=2,
                learning_rate="auto",
                perplexity=50,
                n_iter=1000,
                n_jobs=-1,
                init="pca"
            )
        elif cfg.proj_algorithm == "umap":
            logger.info(f"Setting projection algorithm to UMAP")
            import umap
            projector_sklearn_api = umap.UMAP()
        else: 
            logger.error(f"Unsupported projection algorithm {cfg.proj_algorithm}")
            raise ValueError()

        projector_sklearn_api.fit_transform(train_embeddings)

        with open(projector_path, 'wb') as file: 
            dump(projector_sklearn_api, file)

    colormap = dict()
    markermap = dict()

    for label_idx, label_key in enumerate(label_keys): 
        print(label_key, label_idx)
        for idx, entry in enumerate(unique_keys[label_idx]): 
            print(entry, idx)
            print(label_precedence[label_idx], value_precedence[label_idx])

            #print(globals()[label_precedence[label_idx]])
            #print(globals()[value_precedence[label_idx]])

            globals()[label_precedence[label_idx]][entry] = globals()[value_precedence[label_idx]][idx]

            colormap[entry] = COLORS[idx]

            # colormap[entry] = COLORS[idx]
            # markermap[entry] = MARKERS[idx]

    #print("\n\n\nColorMap", colormap)

    # if change_label_name:
    #     mapping_label_name = dict()
    #     for key in unique_keys: 
    #         alternative_name = str(input(f"Enter name alternative name for the {key}: "))
    #         mapping_label_name[key] = alternative_name  
    #     for idx, meta in enumerate(metas):
    #         metas[idx] = mapping_label_name[meta]

    projected = projector_sklearn_api.fit_transform(test_embeddings)

    #print(list(colormap.keys()))
    source = ColumnDataSource(dict(
            x=projected[:,0],
            y=projected[:,1],
            # marker = [ markermap[meta] for meta in test_labels[1] ],
            color =  [ colormap[meta] for meta in test_labels[0] ],
            label = test_labels[0]
        )
    )

    print("Plotting scatter plot ... ")
    p.scatter(
        x = 'x',
        y = 'y',
        source = source,
        size = 14,
        color ='color',
        #marker ='marker',
        legend_group ='label',
        fill_alpha =0.5,
    )
    p.legend.location = "top_left"
    p.legend.title = "-".join([prefix_name, embedding_key, *label_keys])
    print("plotting scatter done")

    p.legend.background_fill_alpha = 0.3
    p.legend.title_text_font_size = '35pt'
    p.legend.label_text_font_size = '38pt'
    p.title.text_font_size = "40pt"

    plots.append(copy(p)) 

    # from joblib import dump 
    # with open('./plots_tsne_vaule_wp', 'wb') as file:
    #     dump(plots, file)
    # num_columns = max(0, int(embeddings.shape[1] ** (1/2)))
    # num_rows = embeddings.shape[1] // num_columns + 1 
    num_columns = 1 
    num_rows = 1 
    output_width = fw * num_columns
    output_height = fh * num_rows
    grid = gridplot(plots, ncols= num_columns, width=output_width, height=output_height)

    from bokeh.io import export_png, save, export_svg
    from pathlib import Path 
    
    res_dir = Path(dataset_full_path).parent / cfg.proj_algorithm / name 
    os.makedirs(str(res_dir), exist_ok=True)
    image_output_path = os.path.join(res_dir, "_".join([cfg.proj_algorithm ,embedding_key, *label_keys]) + ".png")
    svg_output_path = os.path.join(res_dir, "_".join([cfg.proj_algorithm, embedding_key, *label_keys]) + ".svg")
    html_save_path = os.path.join(res_dir, "_".join([cfg.proj_algorithm, embedding_key, *label_keys]) + ".html")
    
    print(f"Saving image to: {image_output_path}")

    save(grid, filename=html_save_path)
    export_svg(grid, filename=svg_output_path, width=output_width, height=output_height)
    export_png(grid, filename=image_output_path , width=output_width , height=output_height)

    print("Done and done ... ") 

if __name__ == "__main__":
    tsne()