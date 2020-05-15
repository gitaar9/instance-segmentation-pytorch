from os.path import join

import numpy as np
import random
from PIL import Image


def show_annotations(instance_annotation_path, semantic_annotation_path):
    instance_mask = np.load(instance_annotation_path)
    img_height, img_width, real_number_of_instances = instance_mask.shape

    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
              for _ in range(real_number_of_instances)]

    image_array = np.ones((img_height, img_width, 3,)).astype(np.uint8) * 255

    for mask_idx in range(real_number_of_instances):
        image_array[instance_mask[:, :, mask_idx] == 1] = colors[mask_idx]

    Image.fromarray(image_array.astype(np.uint8), mode='RGB').show()

    semantic_mask = np.load(semantic_annotation_path)
    Image.fromarray((semantic_mask * 255).astype(np.uint8), mode='L').show()


processed_dir = "/home/gitaar9/AI/HR/instance-segmentation-pytorch/data/processed"
hebrew_instance_annotation_path = join(processed_dir, "HEBREW/instance-annotations/hebrew_fake_image_0.npy")
hebrew_semantic_annotation_path = join(processed_dir, "HEBREW/semantic-annotations/hebrew_fake_image_0.npy")
show_annotations(hebrew_instance_annotation_path, hebrew_semantic_annotation_path)

cvppp_instance_annotation_path = join(processed_dir, "CVPPP/instance-annotations/plant001.npy")
cvppp_semantic_annotation_path = join(processed_dir, "CVPPP/semantic-annotations/plant001.npy")
show_annotations(cvppp_instance_annotation_path, cvppp_semantic_annotation_path)
