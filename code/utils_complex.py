import torch
from torchvision import transforms
import torch.nn as nn
import time
from model import MyUNetModel, MyComplexModel
from datasets import get_data_loader
from datasets import get_test_data_loader
from triplet import batch_hard_mine

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print(device)


def normalize_and_invert_masks(masks):
    masks = 1 - masks

    flat_masks = masks.view(masks.size()[0], -1)

    max_v = torch.max(flat_masks, dim=1).values
    min_v = torch.min(flat_masks, dim=1).values

    max_v = max_v.view(max_v.size()[0], 1, 1, 1)
    min_v = min_v.view(min_v.size()[0], 1, 1, 1)

    masks = (masks - min_v) / (max_v - min_v + 1e-6)
    # masks = torch.round(masks)
    masks = torch.clamp(masks, min=0.0, max=1.0)

    return masks


def calculate_mask_loss(mse, features_mask, features, masks):
    alpha = 0.95
    beta = 0.5

    feature_loss = mse(features_mask, features)
    '''mask_loss = torch.mean(masks)

    if mask_loss.item() < 0.3:
        mask_loss = 0'''

    # mask_loss = torch.abs(torch.mean(masks) - 0.3)
    mask_loss = torch.mean(masks)
    # extreme_mask_loss = torch.mean(-torch.abs(masks - 0.5))

    transition_loss_y = torch.mean(torch.abs(masks[:, :, 1:, :] - masks[:, :, :-1, :]))
    transition_loss_x = torch.mean(torch.abs(masks[:, :, :, 1:] - masks[:, :, :, :-1]))
    transition_loss = transition_loss_y + transition_loss_x

    # mask_model_loss = (100 * feature_loss) + mask_loss + (0.1 * extreme_mask_loss) + transition_loss
    # mask_model_loss = (100 * feature_loss) + mask_loss + extreme_mask_loss + transition_loss
    mask_features_loss = (beta * mask_loss) + transition_loss
    mask_model_loss = (alpha * feature_loss) + ((1 - alpha) * mask_features_loss)

    return mask_model_loss


def train_both(epochs, model, mask_model, train_loader, optimizer, mask_model_optimizer, loss):
    mse = torch.nn.MSELoss()

    model.train()
    mask_model.train()

    every_x_minibatches = 10

    for epoch in range(epochs):
        running_loss = 0
        running_loss_mask_model = 0
        start = time.time()

        for i, data in enumerate(train_loader):
            images, labels = data

            images = images.to(device)
            labels = labels.to(device)

            #####
            masks = mask_model(images)
            features_mask = model(images, masks, training_mask=True)

            neg_masks = normalize_and_invert_masks(masks.detach())
            global_out, dropblock, regularization, features = model(images, neg_masks, training_mask=False)

            mask_model_loss = calculate_mask_loss(mse, features_mask, features.detach(), masks)

            mask_model_optimizer.zero_grad()
            mask_model_loss.backward()
            mask_model_optimizer.step()

            #####

            triplet_loss_global = loss(*batch_hard_mine(global_out, labels))
            triplet_loss_dropblock = loss(*batch_hard_mine(dropblock, labels))
            triplet_loss_regularization = loss(*batch_hard_mine(regularization, labels))

            model_loss = triplet_loss_global + triplet_loss_dropblock + triplet_loss_regularization

            optimizer.zero_grad()
            model_loss.backward()
            optimizer.step()

            # print statistics
            running_loss += model_loss.item()
            running_loss_mask_model += mask_model_loss.item()

            if (i + 1) % every_x_minibatches == 0:
                print(f'[{epoch+1}, {i+1}] l: {running_loss/every_x_minibatches} ml: {running_loss_mask_model/every_x_minibatches}')
                running_loss = 0
                running_loss_mask_model = 0

            '''if (i + 1) % 50 == 0:
                transforms.ToPILImage()(images[0] * neg_masks[0]).show()
                transforms.ToPILImage()(images[0]).show()
                transforms.ToPILImage()(neg_masks[0]).show()'''

        end = time.time()

        print('epoch time: ' + str(end - start))

        torch.save(model.state_dict(), 'weights\\model_with_mask.pth')
        torch.save(mask_model.state_dict(), 'weights\\mask_with_model.pth')


def main_train_both():
    # train_path = '/ctm-hdd-pool01/leocapozzi/TOPE/Datasets/Market-1501-v15.09.15/bounding_box_train'

    # train_path = 'C:\\Users\\leona\\Documents\\Dataset\\Market-1501-v15.09.15\\bounding_box_train'
    # train_path = 'C:\\Users\\leona\\Documents\\Dataset\\DukeMTMC-reID\\bounding_box_train'
    train_path = 'C:\\Users\\leona\\Documents\\Dataset\\cuhk03-np\\labeled\\bounding_box_train'
    # train_path = 'C:\\Users\\leona\\Documents\\Dataset\\cuhk03-np\\detected\\bounding_box_train'

    # extensions = ['.jpg']
    extensions = ['.png']

    train_transform = transforms.Compose([transforms.Resize((234, 117)),
                                          transforms.RandomCrop((224, 112)),
                                          transforms.RandomHorizontalFlip(p=0.5),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])
                                          ])

    train_loader = get_data_loader(train_path, extensions, train_transform, 8, 3)

    model = MyComplexModel()
    # model.load_state_dict(torch.load('weights\\model_with_mask.pth', map_location='cpu'))
    model = model.to(device)

    mask_model = MyUNetModel()
    # mask_model.load_state_dict(torch.load('weights\\mask_with_model.pth', map_location='cpu'))
    mask_model = mask_model.to(device)

    triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
    mask_model_optimizer = torch.optim.Adam(mask_model.parameters())

    train_both(200, model, mask_model, train_loader, optimizer, mask_model_optimizer, triplet_loss)


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

            features = model(images, None, training_mask=False)
            features = features.cpu()

            test_features.append(features)
            test_labels.append(labels)

        for i, data in enumerate(test_loader_query):
            images, labels = data

            # forward
            images = images.to(device)

            features = model(images, None, training_mask=False)
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

    # test_path = 'C:\\Users\\leona\\Documents\\Dataset\\Market-1501-v15.09.15\\bounding_box_test'
    # query_path = 'C:\\Users\\leona\\Documents\\Dataset\\Market-1501-v15.09.15\\query'

    # test_path = 'C:\\Users\\leona\\Documents\\Dataset\\DukeMTMC-reID\\bounding_box_test'
    # query_path = 'C:\\Users\\leona\\Documents\\Dataset\\DukeMTMC-reID\\query'

    test_path = 'C:\\Users\\leona\\Documents\\Dataset\\cuhk03-np\\labeled\\bounding_box_test'
    query_path = 'C:\\Users\\leona\\Documents\\Dataset\\cuhk03-np\\labeled\\query'

    # test_path = 'C:\\Users\\leona\\Documents\\Dataset\\cuhk03-np\\detected\\bounding_box_test'
    # query_path = 'C:\\Users\\leona\\Documents\\Dataset\\cuhk03-np\\detected\\query'

    # extensions = ['.jpg']
    extensions = ['.png']

    test_transform = transforms.Compose([transforms.Resize((234, 117)),
                                         transforms.CenterCrop((224, 112)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225])
                                         ])

    test_loader = get_test_data_loader(test_path, extensions, test_transform, 128)
    test_loader_query = get_test_data_loader(query_path, extensions, test_transform, 128)

    model = MyComplexModel()
    # model.load_state_dict(torch.load('/ctm-hdd-pool01/leocapozzi/TOPE/ReID/model.pth', map_location='cpu'))
    model.load_state_dict(torch.load('weights\\model_with_mask.pth', map_location='cpu'))
    model = model.to(device)

    test(model, test_loader, test_loader_query)


def test_mask(mask_model, data_loader):
    mask_model.train()

    for i, data in enumerate(data_loader):
        images, _ = data

        images = images.to(device)

        #####
        masks = mask_model(images)

        masks = normalize_and_invert_masks(masks)

        transforms.ToPILImage()(images[0] * masks[0]).show()
        transforms.ToPILImage()(images[0]).show()
        transforms.ToPILImage()(masks[0]).show()
        input('Press Enter to continue...')


def main_test_mask():
    # train_path = '/ctm-hdd-pool01/leocapozzi/TOPE/Datasets/Market-1501-v15.09.15/bounding_box_train'

    # train_path = 'C:\\Users\\leona\\Documents\\Dataset\\Market-1501-v15.09.15\\bounding_box_train'
    # train_path = 'C:\\Users\\leona\\Documents\\Dataset\\DukeMTMC-reID\\bounding_box_train'
    train_path = 'C:\\Users\\leona\\Documents\\Dataset\\cuhk03-np\\labeled\\bounding_box_train'
    # train_path = 'C:\\Users\\leona\\Documents\\Dataset\\cuhk03-np\\detected\\bounding_box_train'


    # extensions = ['.jpg']
    extensions = ['.png']

    train_transform = transforms.Compose([transforms.Resize((234, 117)),
                                          transforms.CenterCrop((224, 112)),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])
                                          ])

    data_loader = get_test_data_loader(train_path, extensions, train_transform, 1)

    mask_model = MyUNetModel()
    mask_model.load_state_dict(torch.load('weights\\mask_with_model.pth', map_location='cpu'))
    mask_model = mask_model.to(device)

    test_mask(mask_model, data_loader)


if __name__ == '__main__':
    # main_train_both()
    main_test()
    # main_test_mask()

