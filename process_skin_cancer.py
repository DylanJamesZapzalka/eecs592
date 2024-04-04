import os
import numpy as np
import pandas as pd
import cv2
import shutil
import multiprocessing
import warnings
import torch

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

SKIN_CANCER_DATASET_DIR = '/home/dylanz/eecs592/data/skin_cancer'
SKIN_CANCER_RAW_DATA_DIR = '/home/dylanz/eecs592/data/raw_data'



def create_final_sample(df):
    '''
    Creates an image for the dataset that may change skin color
    :param df: Dataframe containing all information to create examples
    :return: A list of indices of examples that produced errors
    '''

    error_index_list = []
    for index, row in df.iterrows():
        old_img_location = row['old_img_location']
        new_img_location = row['new_img_location']
        segmentation_location = row['segmentation_location']
        mediator_img_location = row['mediator_img_location']
        skin_color_is_white = row['aux_label']

        error = False
        try:
            # Reading an image and resize to 256x256
            if not os.path.exists(old_img_location):
                error = True
            full_image = cv2.imread(old_img_location)
            full_image = cv2.resize(full_image, (256, 256))

            # Reading the segmentation and resize to 256x256
            if not os.path.exists(segmentation_location):
                error = True
            lesion_mask = cv2.imread(segmentation_location, 0)
            lesion_mask = cv2.resize(lesion_mask, (256, 256))

            background_mask = cv2.bitwise_not(lesion_mask)

            masked_lesion = cv2.bitwise_and(full_image, full_image, mask=lesion_mask)
            masked_background = cv2.bitwise_and(full_image, full_image, mask=background_mask)
            

            # Change the color of the skin
            if not skin_color_is_white:

                # Create a black image
                black_img  = np.full((256,256,3), (0,0,0), np.uint8)

                # add the black skin filter
                masked_background  = cv2.addWeighted(masked_background, 0.2, black_img, 0.2, 0)


            # Obtain final sample
            sample = cv2.bitwise_or(masked_lesion, masked_background)

            # Save images
            cv2.imwrite(new_img_location, sample.astype(np.uint8))
            cv2.imwrite(mediator_img_location, masked_lesion.astype(np.uint8))

        except:
            error = True

        if error:
            error_index_list.append(index)

    return error_index_list


def loss_weights(df):

    labels = torch.tensor(df['cancer'].to_numpy().astype(float))
    auxiliary_labels = torch.tensor(df['aux_label'].to_numpy().astype(float))

    pos_label_pos_aux_weight = torch.sum(labels * auxiliary_labels) / torch.sum(auxiliary_labels)
    neg_label_pos_aux_weight = torch.sum((1.0 - labels) * auxiliary_labels) / torch.sum(auxiliary_labels)

    pos_label_neg_aux_weight = torch.sum(labels * (1.0 - auxiliary_labels)) / torch.sum((1.0 - auxiliary_labels))
    neg_label_neg_aux_weight = torch.sum((1.0 - labels) * (1.0 - auxiliary_labels)) / torch.sum((1.0 - auxiliary_labels))

    # Positive weights
    weights_pos = labels * pos_label_pos_aux_weight + (1.0 - labels) * neg_label_pos_aux_weight
    weights_pos = 1 / weights_pos
    weights_pos = auxiliary_labels * weights_pos
    weights_pos = torch.mean(labels) * labels * weights_pos + \
        torch.mean(1 - labels) * (1.0 - labels) * weights_pos
    
    # Negative weights
    weights_neg = labels * pos_label_neg_aux_weight + (1.0 - labels) * neg_label_neg_aux_weight
    weights_neg = 1.0 / weights_neg
    weights_neg = (1.0 - auxiliary_labels) * weights_neg
    weights_neg = torch.mean(labels) * labels * weights_neg + \
        torch.mean(1.0 - labels) * (1.0 - labels) * weights_neg
    
    # Make sure there are no nan errors
    weights_pos = torch.nan_to_num(weights_pos)
    weights_neg = torch.nan_to_num(weights_neg)
    weights = weights_pos + weights_neg

    # Save weights in dataframe
    df['weights'] = weights
    df['weights_pos'] = weights_pos
    df['weights_neg'] = weights_neg

    return df



def generate_skin_cancer(train_dist, test_dists, seed=None):
    '''
    Generates the koa dataset
    :param train_dist: Percentage of training dataset that is has a spurious attribute
    :param test_dist: Percentage of testing dataset that is has a spurious attribute
    :param seed: Random seed that determines how the dataset is generated
    :param is_double: If true, dataset will contain two shortcuts
    '''

    # Set numpy seed
    np.random.seed(seed)

    # Get dataset directory
    dataset_dir = SKIN_CANCER_DATASET_DIR

    train_df = pd.read_csv(os.path.join(SKIN_CANCER_RAW_DATA_DIR, 'train_skin_cancer.csv'))
    test_df = pd.read_csv(os.path.join(SKIN_CANCER_RAW_DATA_DIR, 'test_skin_cancer.csv'))

    # Make sure directories exist
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError('Directory ' + dataset_dir + ' does not exist.')
    if not os.path.exists(SKIN_CANCER_RAW_DATA_DIR):
        raise FileNotFoundError('Directory ' + SKIN_CANCER_RAW_DATA_DIR + ' does not exist.')



    # Generate a training dataset
    training_dir = os.path.join(dataset_dir, f'training_{seed}')    
    if os.path.exists(training_dir):
        shutil.rmtree(training_dir)
    os.mkdir(training_dir)
    for setting in ['cancer_white', 'benign_white', 'cancer_black', 'benign_black']:
        os.mkdir(os.path.join(training_dir, setting))
    
    # Generate training dataset csv
    drop_list = []
    for index, row in train_df.iterrows():
        is_cancer = row['dx'] == 'bcc' or row['dx'] == 'akiec' or row['dx'] == 'mel'

        if is_cancer:
            white_skin = np.random.randint(100) / 100 < train_dist
        else:
            white_skin = np.random.randint(100) / 100 >= train_dist
    
        image_id = row['image_id']
        image_id_number = int(image_id[5:])
        if image_id_number < 29306:
            old_img_location = os.path.join(SKIN_CANCER_RAW_DATA_DIR, 'HAM10000_images_part_1', image_id + '.jpg')
        else:
            old_img_location = os.path.join(SKIN_CANCER_RAW_DATA_DIR, 'HAM10000_images_part_2', image_id + '.jpg')
        segmenation_image_id = image_id + '_segmentation.png'
        segmenation_location = os.path.join(SKIN_CANCER_RAW_DATA_DIR, 'HAM10000_segmentations', segmenation_image_id)

        if is_cancer and white_skin:
            new_location = os.path.join(training_dir, 'cancer_white', image_id + '.png')
            mediator_img_location = os.path.join(training_dir, 'cancer_white', segmenation_image_id)
        elif is_cancer and not white_skin:
            new_location = os.path.join(training_dir, 'cancer_black', image_id + '.png')
            mediator_img_location = os.path.join(training_dir, 'cancer_black', segmenation_image_id)
        elif not is_cancer and white_skin:
            new_location = os.path.join(training_dir, 'benign_white', image_id + '.png')
            mediator_img_location = os.path.join(training_dir, 'benign_white', segmenation_image_id)
        elif not is_cancer and not white_skin:
            new_location = os.path.join(training_dir, 'benign_black', image_id + '.png')
            mediator_img_location = os.path.join(training_dir, 'benign_black', segmenation_image_id)

        train_df.loc[index, ['old_img_location']] = old_img_location
        train_df.loc[index, ['new_img_location']] = new_location
        train_df.loc[index, ['segmentation_location']] = segmenation_location
        train_df.loc[index, ['mediator_img_location']] = mediator_img_location
        train_df.loc[index, ['cancer']] = int(is_cancer)
        train_df.loc[index, ['group']] = int(white_skin) * 2 + int(is_cancer)
        train_df.loc[index, ['aux_label']] = int(white_skin)

    # Create training dataset
    num_cores = multiprocessing.cpu_count() - 1
    df_split = np.array_split(train_df, num_cores)
    pool = multiprocessing.Pool(num_cores)
    drop_list = pool.map(create_final_sample, df_split)
    pool.close()
    pool.join()
    drop_list = sum(drop_list, [])

    # Drop bad examples
    for i in drop_list:
        train_df.drop(i, inplace=True)

    # Add weights to datasets
    train_df = loss_weights(train_df)

    # Save the csv files
    train_df.to_csv(os.path.join(training_dir, 'training.csv'))



    # Generate a testing dataset
    testing_dir = os.path.join(dataset_dir, 'testing')    
    if os.path.exists(testing_dir):
        shutil.rmtree(testing_dir)
    os.mkdir(testing_dir)
    for test_dist in test_dists:
        os.mkdir(os.path.join(testing_dir, str(test_dist)))
        for setting in ['cancer_white', 'benign_white', 'cancer_black', 'benign_black']:
            os.mkdir(os.path.join(testing_dir, str(test_dist), setting))

    # Create dataset for each test distribution
    for test_dist in test_dists:
        # Generate training dataset csv
        drop_list = []
        test_df_dist = test_df.copy()
        for index, row in test_df_dist.iterrows():
            is_cancer = row['dx'] == 'bcc' or row['dx'] == 'akiec' or row['dx'] == 'mel'

            if is_cancer:
                white_skin = np.random.randint(100) / 100 < test_dist
            else:
                white_skin = np.random.randint(100) / 100 >= test_dist
        
            image_id = row['image_id']
            image_id_number = int(image_id[5:])
            if image_id_number < 29306:
                old_img_location = os.path.join(SKIN_CANCER_RAW_DATA_DIR, 'HAM10000_images_part_1', image_id + '.jpg')
            else:
                old_img_location = os.path.join(SKIN_CANCER_RAW_DATA_DIR, 'HAM10000_images_part_2', image_id + '.jpg')
            segmenation_image_id = image_id + '_segmentation.png'
            segmenation_location = os.path.join(SKIN_CANCER_RAW_DATA_DIR, 'HAM10000_segmentations', segmenation_image_id)

            if is_cancer and white_skin:
                new_location = os.path.join(testing_dir, str(test_dist), 'cancer_white', image_id + '.png')
                mediator_img_location = os.path.join(testing_dir, str(test_dist), 'cancer_white', segmenation_image_id)
            elif is_cancer and not white_skin:
                new_location = os.path.join(testing_dir, str(test_dist), 'cancer_black', image_id + '.png')
                mediator_img_location = os.path.join(testing_dir, str(test_dist), 'cancer_black', segmenation_image_id)
            elif not is_cancer and white_skin:
                new_location = os.path.join(testing_dir, str(test_dist), 'benign_white', image_id + '.png')
                mediator_img_location = os.path.join(testing_dir, str(test_dist), 'benign_white', segmenation_image_id)
            elif not is_cancer and not white_skin:
                new_location = os.path.join(testing_dir, str(test_dist), 'benign_black', image_id + '.png')
                mediator_img_location = os.path.join(testing_dir, str(test_dist), 'benign_black', segmenation_image_id)

            test_df_dist.loc[index, ['old_img_location']] = old_img_location
            test_df_dist.loc[index, ['new_img_location']] = new_location
            test_df_dist.loc[index, ['segmentation_location']] = segmenation_location
            test_df_dist.loc[index, ['mediator_img_location']] = mediator_img_location
            test_df_dist.loc[index, ['cancer']] = int(is_cancer)
            test_df_dist.loc[index, ['group']] = int(white_skin) * 2 + int(is_cancer)
            test_df_dist.loc[index, ['aux_label']] = int(white_skin)

        # Create testing dataset
        num_cores = multiprocessing.cpu_count() - 1
        df_split = np.array_split(test_df_dist, num_cores)
        pool = multiprocessing.Pool(num_cores)
        drop_list = pool.map(create_final_sample, df_split)
        pool.close()
        pool.join()
        drop_list = sum(drop_list, [])

        # Drop bad examples
        for i in drop_list:
            test_df_dist.drop(i, inplace=True)

        # Add weights to datasets
        test_df_dist = loss_weights(test_df_dist)

        # Save the csv files
        test_df_dist.to_csv(os.path.join(testing_dir, str(test_dist), 'testing.csv'))


if __name__ == "__main__":
    generate_skin_cancer(0.9, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 4)