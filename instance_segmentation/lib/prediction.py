import cv2
import numpy as np
import torch
from sklearn.cluster import KMeans, MeanShift
from sklearn.manifold import TSNE
from torchvision.transforms import transforms

from instance_segmentation.lib.archs.modules.coord_conv import AddCoordinates
from instance_segmentation.lib.utils import ImageUtilities
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt



class Prediction(object):

    def __init__(self, resize_height, resize_width, mean,
                 std, use_coordinates, model, n_workers):

        self.normalizer = ImageUtilities.image_normalizer(mean, std)
        self.use_coordinates = use_coordinates

        self.resize_height = resize_height
        self.resize_width = resize_width
        self.model = model

        self.n_workers = n_workers

        self.img_resizer = ImageUtilities.image_resizer(
            self.resize_height, self.resize_width)

        if self.use_coordinates:
            self.coordinate_adder = AddCoordinates(with_r=True,
                                                   usegpu=False)

    def get_image(self, image_path):

        img = ImageUtilities.read_image(image_path)
        image_width, image_height = img.size

        img = self.img_resizer(img)
        # img = self.normalizer(img)
        img = transforms.Compose([transforms.ToTensor()])(img)
        # Make biggest value in tensor equal to one
        img /= torch.max(img)

        # thresholding
        t = torch.Tensor([0.3])
        img = (~(img > t)).float() * 1.0


        return img, image_height, image_width

    def get_annotation(self, annotation_path):

        img = ImageUtilities.read_image(annotation_path)
        return img

    def upsample_prediction(self, prediction, image_height, image_width):

        return cv2.resize(prediction, (image_width, image_height),
                          interpolation=cv2.INTER_NEAREST)

    def cluster(self, sem_seg_prediction, ins_seg_prediction,
                n_objects_prediction):

        sem_seg_prediction = sem_seg_prediction.squeeze(0)
        ins_seg_prediction = ins_seg_prediction.squeeze(0)
        n_objects_prediction = n_objects_prediction.squeeze(0)

        seg_height, seg_width = ins_seg_prediction.shape[1:]

        sem_seg_prediction = sem_seg_prediction.cpu().numpy()
        sem_seg_prediction = sem_seg_prediction.argmax(0).astype(np.uint8)

        embeddings = ins_seg_prediction.cpu()
        if self.use_coordinates:
            embeddings = self.coordinate_adder(embeddings)
        embeddings = embeddings.numpy()
        embeddings = embeddings.transpose(1, 2, 0)  # h, w, c

        n_objects_prediction = n_objects_prediction.cpu().numpy()[0]

        embeddings = np.stack([embeddings[:, :, i][sem_seg_prediction != 0]
                               for i in range(embeddings.shape[2])], axis=1)

        print(f"Embeddings size: {embeddings.shape}\nNumber of instances: {n_objects_prediction}")
        print(f"First embedding: {embeddings[0]}")

        labels = KMeans(n_clusters=n_objects_prediction,
                        n_init=35, max_iter=500,
                        n_jobs=self.n_workers).fit_predict(embeddings)
        # labels = MeanShift(bandwidth=1, min_bin_freq=n_objects_prediction - 4, max_iter=700,
        #                    n_jobs=self.n_workers).fit_predict(embeddings)
        print(labels.shape)
        self.scatter_embeddings(embeddings, labels)

        instance_mask = np.zeros((seg_height, seg_width), dtype=np.uint8)

        fg_coords = np.where(sem_seg_prediction != 0)
        for si in range(len(fg_coords[0])):
            y_coord = fg_coords[0][si]
            x_coord = fg_coords[1][si]
            _label = labels[si] + 1
            instance_mask[y_coord, x_coord] = _label

        return sem_seg_prediction, instance_mask, n_objects_prediction

    def predict(self, image_path):
        print(f"Prediction for {image_path}")
        image, image_height, image_width = self.get_image(image_path)
        print(image.shape)
        image = image.unsqueeze(0)

        sem_seg_prediction, ins_seg_prediction, n_objects_prediction = \
            self.model.predict(image)

        sem_seg_prediction, ins_seg_prediction, \
            n_objects_prediction = self.cluster(sem_seg_prediction,
                                                ins_seg_prediction,
                                                n_objects_prediction)

        sem_seg_prediction = self.upsample_prediction(
            sem_seg_prediction, image_height, image_width)
        ins_seg_prediction = self.upsample_prediction(
            ins_seg_prediction, image_height, image_width)

        raw_image_pil = ImageUtilities.read_image(image_path)
        raw_image = np.array(raw_image_pil)

        return raw_image, sem_seg_prediction, ins_seg_prediction, \
            n_objects_prediction

    def scatter_embeddings(self, embeddings, labels, amount_of_sampled_points=5000):
        _n_fg_samples = embeddings.shape[0]
        if _n_fg_samples > 0:
            sampled_indexes = np.random.choice(_n_fg_samples, amount_of_sampled_points, replace=False)
            print("Started the TSNE")

            tsne = TSNE(n_components=2, random_state=0, n_jobs=self.n_workers, perplexity=45)
            tsne_result = tsne.fit_transform(embeddings)
            print("Ended the TSNE")
            for x, y, label in zip(tsne_result[sampled_indexes][:, 0], tsne_result[sampled_indexes][:, 1], labels[sampled_indexes]):
                plt.scatter(x, y, marker=',', label=label)
            plt.show()

