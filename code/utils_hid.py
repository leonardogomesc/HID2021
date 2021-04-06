import torch
from torchvision import transforms
import torch.nn as nn
import time
from model import MyModel
from datasets import get_data_loader, get_test_data_loader
from triplet import batch_hard_mine

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print(device)


def train_backbone(epochs, model, train_loader, optimizer, loss):
    every_x_minibatches = 10
    model.train()

    for epoch in range(epochs):

        running_loss = 0.0

        start = time.time()

        for i, (images, labels) in enumerate(train_loader):

            # forward + backward + optimize
            images = images.to(device)
            labels = labels.to(device)

            features = model(images)

            a, p, n = batch_hard_mine(features, labels)

            loss_value = loss(a, p, n)

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

    print('Finished Training')


def main_train_backbone():
    # train_path = '/ctm-hdd-pool01/leocapozzi/TOPE/Datasets/Market-1501-v15.09.15/bounding_box_train'

    train_path = 'C:\\Users\\leona\\Documents\\Dataset\\Market-1501-v15.09.15\\bounding_box_train'
    # train_path = 'C:\\Users\\leona\\Documents\\Dataset\\DukeMTMC-reID\\bounding_box_train'
    # train_path = 'C:\\Users\\leona\\Documents\\Dataset\\cuhk03-np\\labeled\\bounding_box_train'
    # train_path = 'C:\\Users\\leona\\Documents\\Dataset\\cuhk03-np\\detected\\bounding_box_train'

    extensions = ['.jpg']
    # extensions = ['.png']

    train_transform = transforms.Compose([transforms.Resize((234, 117)),
                                          transforms.RandomCrop((224, 112)),
                                          transforms.RandomHorizontalFlip(p=0.5),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])
                                          ])

    train_loader = get_data_loader(train_path, extensions, train_transform, 20, 3)

    model = MyModel()
    model = model.to(device)

    triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)

    train_backbone(50, model, train_loader, optimizer, triplet_loss)


def test(model, test_loader, test_loader_query):
    test_features = []
    test_labels = []

    query_features = []
    query_labels = []

    with torch.no_grad():
        model.eval()

        for i, data in enumerate(test_loader):
            images, labels = data

            # forward
            images = images.to(device)

            features = model(images)
            features = features.cpu()

            test_features.append(features)
            test_labels.append(labels)

        for i, data in enumerate(test_loader_query):
            images, labels = data

            # forward
            images = images.to(device)

            features = model(images)
            features = features.cpu()

            query_features.append(features)
            query_labels.append(labels)

        test_features = torch.cat(test_features, dim=0)
        test_labels = torch.cat(test_labels, dim=0)

        query_features = torch.cat(query_features, dim=0)
        query_labels = torch.cat(query_labels, dim=0)

        distance_matrix = torch.cdist(query_features, test_features, p=2)
        sorted_matrix = torch.argsort(distance_matrix, dim=1)

        rank1 = sorted_matrix[:, :1]
        rank1_correct = 0

        rank5 = sorted_matrix[:, :5]
        rank5_correct = 0

        rank10 = sorted_matrix[:, :10]
        rank10_correct = 0

        total = 0

        for i in range(len(query_labels)):
            q_label = query_labels[i]

            if q_label in test_labels[rank1[i]]:
                rank1_correct += 1

            if q_label in test_labels[rank5[i]]:
                rank5_correct += 1

            if q_label in test_labels[rank10[i]]:
                rank10_correct += 1

            total += 1

        print('rank1 acc: ' + str(rank1_correct / total))
        print('rank5 acc: ' + str(rank5_correct / total))
        print('rank10 acc: ' + str(rank10_correct / total))

        # map

        expanded_test_labels = test_labels.repeat(query_labels.size()[0], 1)

        sorted_labels_matrix = torch.gather(expanded_test_labels, 1, sorted_matrix)

        query_mask = torch.unsqueeze(query_labels, 1) == sorted_labels_matrix

        cum_true = torch.cumsum(query_mask, dim=1)

        num_pred_pos = torch.cumsum(torch.ones_like(cum_true), dim=1)

        p = query_mask * (cum_true / num_pred_pos)

        ap = torch.sum(p, 1)/torch.sum(query_mask, 1)

        map = torch.mean(ap)

        # map = torch.sum(p)/torch.sum(query_mask)

        print('')
        print('map: ' + str(map.item()))


def main_test():
    # test_path = '/ctm-hdd-pool01/leocapozzi/TOPE/Datasets/Market-1501-v15.09.15/bounding_box_test'

    test_path = 'C:\\Users\\leona\\Documents\\Dataset\\Market-1501-v15.09.15\\bounding_box_test'
    query_path = 'C:\\Users\\leona\\Documents\\Dataset\\Market-1501-v15.09.15\\query'

    # test_path = 'C:\\Users\\leona\\Documents\\Dataset\\DukeMTMC-reID\\bounding_box_test'
    # query_path = 'C:\\Users\\leona\\Documents\\Dataset\\DukeMTMC-reID\\query'

    # test_path = 'C:\\Users\\leona\\Documents\\Dataset\\cuhk03-np\\labeled\\bounding_box_test'
    # query_path = 'C:\\Users\\leona\\Documents\\Dataset\\cuhk03-np\\labeled\\query'

    # test_path = 'C:\\Users\\leona\\Documents\\Dataset\\cuhk03-np\\detected\\bounding_box_test'
    # query_path = 'C:\\Users\\leona\\Documents\\Dataset\\cuhk03-np\\detected\\query'

    extensions = ['.jpg']
    # extensions = ['.png']

    test_transform = transforms.Compose([transforms.Resize((234, 117)),
                                         transforms.CenterCrop((224, 112)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225])
                                         ])

    test_loader = get_test_data_loader(test_path, extensions, test_transform, 128)
    test_loader_query = get_test_data_loader(query_path, extensions, test_transform, 128)

    model = MyModel()
    # model.load_state_dict(torch.load('/ctm-hdd-pool01/leocapozzi/TOPE/ReID/model.pth', map_location='cpu'))
    model.load_state_dict(torch.load('weights\\model_with_mask.pth', map_location='cpu'))
    model = model.to(device)

    test(model, test_loader, test_loader_query)


if __name__ == '__main__':
    main_train_backbone()
    # main_test()
    # main_test_mask()


