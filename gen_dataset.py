from datasets import load_dataset, logging, disable_caching
from argparse import ArgumentParser
from multiprocessing import cpu_count
import os 

parser = ArgumentParser()
parser.add_argument("dataset_name", choices=[
    "fairface",
    "morph",
    "rfw",
    "utkface",
])
parser.add_argument("--output_path", default="datasets") 

args = parser.parse_args()
os.makedirs(args.output_path, exist_ok=True)
output_path = args.output_path

logging.set_verbosity_info()

dataset_builder_name = args.dataset_name + "2latentplay"
data = load_dataset(f"./builders/{dataset_builder_name}", num_proc=cpu_count()-1)  

output_path = os.path.join(args.output_path, dataset_builder_name + ".hf")
print(f"Dataset will be saved at: {output_path}")
data.save_to_disk(output_path)
