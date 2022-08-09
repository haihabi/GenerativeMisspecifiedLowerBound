import os
import pyresearchutils as pru
import numpy as np
import torch
import pickle


def save_dataset2file(in_ds, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(in_ds, f)


def load_dataset2file(file_path):
    with open(file_path, "rb") as f:
        ds = pickle.load(f)
    return ds


def dataset_vs_testset_checking(data_model, in_dataset, parameter_name, range_iteration=20):
    pru.logger.info("Starting Dataset check")
    testing_set = data_model.parameter_range(range_iteration, theta_scale_min=None, theta_scale_max=None)
    for theta in testing_set[parameter_name].cpu().detach().numpy():
        label_array = np.stack(in_dataset.label)
        error = np.abs(theta.reshape([1, -1]) - label_array).sum(axis=-1)
        if np.any(error == 0):
            pru.logger.critical(f"{theta} is in the training set")
    pru.logger.info("Finished dataset check")


def save_or_load_model(dm, base_dataset_folder, disable_load, additional_save=None):
    model_dataset_file_path = os.path.join(base_dataset_folder, "models")
    os.makedirs(model_dataset_file_path, exist_ok=True)
    if dm.model_exist(model_dataset_file_path) and not disable_load:
        dm.load_data_model(model_dataset_file_path)
        pru.logger.info(f"Load Model:{model_dataset_file_path}")
    else:
        dm.save_data_model(model_dataset_file_path)
        pru.logger.info(f"Save Model:{model_dataset_file_path}")
    if additional_save is not None:
        dm.save_data_model(additional_save)


def generate_and_save_or_load_dataset(in_data_model,
                                      base_dataset_folder,
                                      batch_size,
                                      dataset_size,
                                      val_dataset_size,
                                      transform=None,
                                      force_data_generation=False):
    os.makedirs(base_dataset_folder, exist_ok=True)
    training_dataset_file_path = os.path.join(base_dataset_folder,
                                              f"training_{in_data_model.name}_{dataset_size}_dataset.pickle")
    validation_dataset_file_path = os.path.join(base_dataset_folder,
                                                f"validation_{in_data_model.name}_{val_dataset_size}_dataset.pickle")
    if os.path.isfile(training_dataset_file_path) and os.path.isfile(
            validation_dataset_file_path) and not force_data_generation:
        training_data = load_dataset2file(training_dataset_file_path)
        validation_data = load_dataset2file(validation_dataset_file_path)
        pru.logger.info("Loading Dataset Files")
    else:
        pru.logger.info("Start Training Dataset Generation")
        training_data = in_data_model.build_dataset(dataset_size, None)
        pru.logger.info("Start Validation Dataset Generation")
        validation_data = in_data_model.build_dataset(val_dataset_size, None)
        save_dataset2file(training_data, training_dataset_file_path)
        save_dataset2file(validation_data, validation_dataset_file_path)
        pru.logger.info("Saving Dataset Files")
    training_data.set_transform(transform)
    validation_data.set_transform(transform)

    training_dataset_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size,
                                                          shuffle=True, num_workers=0, pin_memory=False)
    validation_dataset_loader = torch.utils.data.DataLoader(validation_data, batch_size=batch_size,
                                                            shuffle=False, num_workers=0, pin_memory=False)
    return training_dataset_loader, validation_dataset_loader
