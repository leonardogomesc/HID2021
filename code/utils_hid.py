import torch
from torchvision import transforms
import torch.nn as nn
import time
from model import MyModel
from datasets import get_data_loader, get_test_data_loader
from triplet import batch_hard_mine

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print(device)


def train_backbone(epochs, model, train_loader, optimizer, triplet_loss, cross_ent):
    every_x_minibatches = 10
    model.train()

    for epoch in range(epochs):

        running_loss = 0.0

        start = time.time()

        for i, (videos, labels, _) in enumerate(train_loader):

            # forward + backward + optimize
            videos = videos.to(device)
            labels = labels.to(device)

            features, fc = model(videos)

            a, p, n = batch_hard_mine(features, labels)

            loss_value = triplet_loss(a, p, n) + cross_ent(fc, labels)

            # zero the parameter gradients
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

            # print statistics
            running_loss += loss_value.item()
            if (i + 1) % every_x_minibatches == 0:
                print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / every_x_minibatches}')
                running_loss = 0.0

        end = time.time()

        print('epoch time: ' + str(end - start))

        torch.save(model.state_dict(), 'weights\\model.pth')
        # torch.save(model.state_dict(), 'weights/model.pth')

    print('Finished Training')


def main_train_backbone():
    train_path = 'C:\\Users\\Leonardo Capozzi\\Desktop\\hid\\train'
    # train_path = '/ctm-hdd-pool01/leocapozzi/HID2021/data/train'

    extensions = ['.jpg']

    train_transform = transforms.Compose([transforms.Resize((90, 90)),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.43216, 0.394666, 0.37645],
                                                               std=[0.22803, 0.22145, 0.216989])])

    train_loader, num_classes = get_data_loader(train_path, extensions, train_transform, 20, False, 9, 3)

    model = MyModel(num_classes)
    # model.load_state_dict(torch.load('/ctm-hdd-pool01/leocapozzi/HID2021/code/weights/model.pth', map_location='cpu'))
    model = model.to(device)

    triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
    cross_ent = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    train_backbone(100, model, train_loader, optimizer, triplet_loss, cross_ent)


def test(model, test_loader, test_loader_query):
    test_features = []
    test_subject_ids = []
    test_video_ids = []

    query_features = []
    query_subject_ids = []
    query_video_ids = []

    with torch.no_grad():
        model.eval()

        for i, data in enumerate(test_loader):
            videos, subject_ids, video_ids = data

            # forward
            videos = videos.to(device)

            features, fc = model(videos)
            features = features.cpu()

            test_features.append(features)
            test_subject_ids.append(subject_ids)
            test_video_ids.extend(list(video_ids))

            print(i)

        for i, data in enumerate(test_loader_query):
            videos, subject_ids, video_ids = data

            # forward
            videos = videos.to(device)

            features, fc = model(videos)
            features = features.cpu()

            query_features.append(features)
            query_subject_ids.append(subject_ids)
            query_video_ids.extend(list(video_ids))

            print(i)

        test_features = torch.cat(test_features, dim=0)
        test_subject_ids = torch.cat(test_subject_ids, dim=0)

        query_features = torch.cat(query_features, dim=0)
        query_subject_ids = torch.cat(query_subject_ids, dim=0)

        distance_matrix = torch.cdist(query_features, test_features, p=2)
        sorted_matrix = torch.argsort(distance_matrix, dim=1)

        f = open('submission.csv', 'w')
        f.write('videoID,label')
        f.write('\n')

        for i in range(len(query_features)):
            query_v_id = query_video_ids[i]

            idx_best_subject = sorted_matrix[i][0]
            best_subject_id = int(test_subject_ids[idx_best_subject])

            f.write(f'{query_v_id},{best_subject_id}')
            f.write('\n')

        f.close()


def main_test():

    test_path = 'C:\\Users\\Leonardo Capozzi\\Desktop\\hid\\HID2021_test_gallery\\HID2021_test_gallery'
    query_path = 'C:\\Users\\Leonardo Capozzi\\Desktop\\hid\\HID2021_test_probe\\HID2021_test_probe'

    extensions = ['.jpg']

    test_transform = transforms.Compose([transforms.Resize((90, 90)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.43216, 0.394666, 0.37645],
                                                              std=[0.22803, 0.22145, 0.216989])])

    test_loader = get_test_data_loader(test_path, extensions, test_transform, 2, False, 64)
    test_loader_query = get_test_data_loader(query_path, extensions, test_transform, 2, True, 64)

    model = MyModel(1)
    # model.load_state_dict(torch.load('/ctm-hdd-pool01/leocapozzi/TOPE/ReID/model.pth', map_location='cpu'))
    # model.load_state_dict(torch.load('weights\\model_with_mask.pth', map_location='cpu'))
    model = model.to(device)

    test(model, test_loader, test_loader_query)


if __name__ == '__main__':
    # main_train_backbone()
    main_test()


