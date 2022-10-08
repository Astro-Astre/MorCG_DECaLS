from torch.utils.data import Dataset
from astre_utils.img.prepare import *
from prefetch_generator import BackgroundGenerator
from utils.label_metadata import *
from utils.schemas import *
from tqdm import tqdm


question_answer_pairs = gz2_pairs
dependencies = gz2_and_decals_dependencies
schema = Schema(question_answer_pairs, dependencies)
label_cols = schema.label_cols


def get_galaxy_label(galaxy, label_cols):
    # no longer casts to int64, user now responsible in df. If dtype is mixed, will try to infer with infer_objects
    return galaxy[label_cols].infer_objects().values.squeeze()  # squeeze for if there's one label_col


class GalaxyDataset(Dataset):
    def __init__(self, annotations_file, transform):
        with open(annotations_file, "r") as file:
            imgs = []
            for line in file:
                line = line.strip("\n")
                line = line.rstrip("\n")
                words = line.split()
                label = str(line)[:-1].split("label:")
                hidden = list(label[-1][:].split(" "))
                hidden = hidden[:-1]
                votes = []
                for vote in hidden:
                    votes.append(float(vote))
                imgs.append((words[0], votes))
        self.imgs = imgs
        self.transform = transform

    def __getitem__(self, index):
        path, label = self.imgs[index]
        astro_img = np.array(AstroImg(path).load()).astype(np.float32)
        return np.nan_to_num(astro_img), np.array(label)

    def __len__(self):
        return len(self.imgs)

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())
