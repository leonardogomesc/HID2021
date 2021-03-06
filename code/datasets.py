from torch.utils.data import Dataset, Sampler, DataLoader
from torchvision import transforms
import os
import random
from PIL import Image
import math
import numpy as np
import matplotlib.pyplot as plt
import torch


class CustomDataset(Dataset):

    def __init__(self, root, extensions, transform, n_frames=-1, isProbe=False, normLabels=False):
        self.root = root
        self.extensions = extensions
        self.transform = transform
        self.n_frames = n_frames
        self.individuals = {}

        if isProbe:
            self.files = []

            for video_id in os.listdir(self.root):
                video_path = os.path.normpath(os.path.join(self.root, video_id))
                if len(os.listdir(video_path)) != 0:
                    self.files.append(video_path)

            self.files.sort()

            self.subject_ids = []
            self.video_ids = []

            for file in self.files:
                file = file.split(os.sep)
                self.subject_ids.append(-1)
                self.video_ids.append(file[-1])
        else:
            self.files = []

            for subject_id in os.listdir(self.root):
                for video_id in os.listdir(os.path.join(self.root, subject_id)):
                    video_path = os.path.normpath(os.path.join(self.root, subject_id, video_id))
                    if len(os.listdir(video_path)) != 0:
                        self.files.append(video_path)

            self.files.sort()

            self.subject_ids = []
            self.video_ids = []

            for file in self.files:
                file = file.split(os.sep)
                self.subject_ids.append(int(file[-2]))
                self.video_ids.append(file[-1])

        if normLabels:
            norm_subject_ids = []

            id_dic = {}
            current_id = -1

            for s_id in self.subject_ids:

                norm_id = id_dic.get(s_id, -1)

                if norm_id == -1:
                    current_id += 1
                    id_dic[s_id] = current_id
                    norm_id = current_id

                norm_subject_ids.append(norm_id)

            self.subject_ids = norm_subject_ids

        for idx in range(len(self.subject_ids)):
            subject_id = self.subject_ids[idx]

            file_idx_list = self.individuals.get(subject_id, [])
            file_idx_list.append(idx)

            self.individuals[subject_id] = file_idx_list

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        video_path = self.files[idx]
        frame_paths = []

        for frame in os.listdir(video_path):
            if os.path.splitext(frame)[1] in self.extensions:
                frame_paths.append(os.path.join(video_path, frame))

        frame_paths.sort()

        if self.n_frames != -1:

            if len(frame_paths) < self.n_frames:
                ratio = math.floor(self.n_frames/len(frame_paths))
                remainder = self.n_frames % len(frame_paths)

                frame_paths_larger = []

                for frame in frame_paths:
                    for _ in range(ratio):
                        frame_paths_larger.append(frame)

                frame_paths_larger_2 = []
                split_ratio = round(len(frame_paths_larger)/(remainder+1))
                counter = 0

                for frame in frame_paths_larger:
                    frame_paths_larger_2.append(frame)
                    counter += 1

                    if counter == split_ratio:
                        frame_paths_larger_2.append(frame)
                        counter = 0

                if len(frame_paths_larger_2) > self.n_frames:
                    frame_paths_larger_2 = frame_paths_larger_2[:self.n_frames]
                elif len(frame_paths_larger_2) < self.n_frames:
                    last_elem = frame_paths_larger_2[-1]

                    while len(frame_paths_larger_2) < self.n_frames:
                        frame_paths_larger_2.append(last_elem)

                frame_paths = frame_paths_larger_2
            else:
                start_idx = random.randint(0, len(frame_paths)-self.n_frames)
                frame_paths = frame_paths[start_idx:start_idx+self.n_frames]

        tensor_frames = []

        for frame in frame_paths:
            frame = Image.open(frame).convert('RGB')
            tensor_frames.append(self.transform(frame))

        tensor_video = torch.stack(tensor_frames, dim=1)

        return tensor_video, self.subject_ids[idx], self.video_ids[idx]

    def get_num_classes(self):
        return len(list(self.individuals.keys()))


class BatchSampler(Sampler):
    def __init__(self, dataset, n_persons, n_videos):
        self.dataset = dataset
        self.n_persons = n_persons
        self.n_videos = n_videos

        self.len = math.ceil(len(self.dataset) / (self.n_persons * self.n_videos))

    def __iter__(self):
        for _ in range(self.len):
            anchors = []
            keys = list(self.dataset.individuals.keys())

            if len(keys) >= self.n_persons:
                keys = random.sample(keys, self.n_persons)

            for key in keys:
                objects = self.dataset.individuals[key]

                if len(objects) >= self.n_videos:
                    objects = random.sample(objects, self.n_videos)

                anchors.extend(objects)

            yield anchors

    def __len__(self):
        return self.len


def get_data_loader(data_path, extensions, transform, n_frames, isProbe, n_persons, n_videos):

    dataset = CustomDataset(data_path, extensions, transform, n_frames=n_frames, isProbe=isProbe, normLabels=True)

    batch_sampler = BatchSampler(dataset, n_persons, n_videos)

    data_loader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=4)

    return data_loader, dataset.get_num_classes()


def get_test_data_loader(data_path, extensions, transform, n_frames, is_Probe, batch_size):

    dataset = CustomDataset(data_path, extensions, transform, n_frames=n_frames, isProbe=is_Probe, normLabels=False)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return data_loader


def get_val_data_loader(data_path, extensions, transform, n_frames, is_Probe, batch_size):

    dataset = CustomDataset(data_path, extensions, transform, n_frames=n_frames, isProbe=is_Probe, normLabels=True)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return data_loader


def calculate_n_videos(dataset):
    min = np.inf
    max = -np.inf
    avg = 0
    counter = 0

    x = [v for v in range(1000)]
    y = [0 for _ in range(1000)]

    for key in dataset.individuals.keys():
        n_videos = len(dataset.individuals[key])

        if n_videos < len(x):
            y[n_videos] += 1

        if n_videos < min:
            min = n_videos

        if n_videos > max:
            max = n_videos

        avg += n_videos
        counter += 1

    avg /= counter

    # plt.plot(x, y, 'bo')
    # plt.show()

    return min, max, avg


def calculate_n_frames(dataset):
    min = np.inf
    max = -np.inf
    avg = 0
    counter = 0

    x = [v for v in range(4000)]
    y = [0 for _ in range(4000)]

    for key in dataset.individuals.keys():
        for idx in dataset.individuals[key]:
            n_frames = len(os.listdir(dataset.files[idx]))

            if n_frames < len(x):
                y[n_frames] += 1

            if n_frames < min:
                min = n_frames

            if n_frames > max:
                max = n_frames

            avg += n_frames
            counter += 1

    avg /= counter

    # plt.plot(x, y, 'bo')
    # plt.show()

    return min, max, avg


def test():
    train_path = 'C:\\Users\\Leonardo Capozzi\\Desktop\\hid\\train'
    gallery_path = 'C:\\Users\\Leonardo Capozzi\\Desktop\\hid\\HID2021_test_gallery\\HID2021_test_gallery'
    probe_path = 'C:\\Users\\Leonardo Capozzi\\Desktop\\hid\\HID2021_test_probe\\HID2021_test_probe'
    extensions = ['.jpg']

    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = CustomDataset(train_path, extensions, transform, n_frames=100, isProbe=False, normLabels=True)
    gallery_dataset = CustomDataset(gallery_path, extensions, transform, n_frames=-1, isProbe=False)
    probe_dataset = CustomDataset(probe_path, extensions, transform, n_frames=-1, isProbe=True)

    print(train_dataset.get_num_classes())
    print(gallery_dataset.get_num_classes())
    print(probe_dataset.get_num_classes())
    print(train_dataset.subject_ids)

    v, l, lv = train_dataset[0]

    print(v)
    print(v.size())
    print(l)
    print(lv)

    print(calculate_n_videos(train_dataset))
    print(calculate_n_videos(gallery_dataset))
    print(calculate_n_videos(probe_dataset))

    print()

    print(calculate_n_frames(train_dataset))
    print(calculate_n_frames(gallery_dataset))
    print(calculate_n_frames(probe_dataset))

    train_loader = get_data_loader(train_path, extensions, transform, 20, False, 10, 3)

    for t in train_loader:
        print(t[0].size())
        print(t[1].size())
        print(t)
        break


if __name__ == '__main__':
    test()
