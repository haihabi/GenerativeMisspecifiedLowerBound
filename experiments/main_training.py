import pyresearchutils  as pru
import measurements_distributions
import os
import torch
from experiments import constants
import flow_models
from tqdm import tqdm
import wandb


def config() -> pru.ConfigReader:
    _cr = pru.initialized_config_reader(default_base_log_folder=os.path.join("./temp", "logs"))
    ###############################################
    # Training
    ###############################################
    _cr.add_parameter("n_epochs", type=int, default=10)
    _cr.add_parameter("lr", type=float, default=1.6e-3)
    _cr.add_parameter("grad_norm_clipping", type=float, default=0.1)
    _cr.add_parameter("weight_decay", type=float, default=1e-4)

    _cr.add_parameter("random_padding", type=str, default="false")
    _cr.add_parameter("padding_size", type=int, default=0)
    # _cr.add_parameter("quantization_preprocessing", type=str, default="true")
    # _cr.add_parameter("variational_dequantization", type=str, default="false")

    ###############################################
    # Signal Model Parameter
    ###############################################
    _cr.add_parameter("d_x", type=int, default=16)
    _cr.add_parameter("d_p", type=int, default=1)
    _cr.add_parameter("norm_min", type=float, default=0.01)
    _cr.add_parameter("norm_max", type=float, default=10.0)
    # _cr.add_parameter("threshold", type=float, default=1.0)
    # _cr.add_parameter("snr_min", type=float, default=2.0)
    # _cr.add_parameter("snr_max", type=float, default=2.0)
    # _cr.add_parameter('quantization', type=str, default="true")
    # _cr.add_parameter('base_model_folder', type=str, default="./temp")
    ###############################################
    # Dataset Parameters
    ###############################################
    _cr.add_parameter('base_dataset_folder', type=str, default="./temp/datasets")
    _cr.add_parameter('batch_size', type=int, default=512)
    _cr.add_parameter('dataset_size', type=int, default=200000)
    _cr.add_parameter('val_dataset_size', type=int, default=20000)
    _cr.add_parameter('force_data_generation', type=str, default="false")
    ###############################################
    # Flow Model
    ###############################################
    # _cr.add_parameter("n_blocks", type=int, default=3)
    # _cr.add_parameter("coupling_type", type=str, default="Affine", enum=flow_model.CouplingType)

    return _cr


def flow_training_loop(in_step_per_epoch,
                       in_train_loader,
                       in_cnf,
                       in_opt,
                       in_ma):
    in_cnf.train()
    with tqdm(total=in_step_per_epoch) as progress_bar:
        for i, (gamma, param) in enumerate(in_train_loader):
            param = {k: pru.torch.update_device(v) for k, v in param.items()}
            gamma = pru.torch.update_device(gamma)
            in_opt.zero_grad()
            loss = in_cnf.nll_mean(gamma, **param)
            loss.backward()
            in_opt.step()
            in_ma.log(loss=loss.item())

            progress_bar.set_postfix(nll=loss.item())
            progress_bar.update(1.0)


def validation_run(in_ma, in_validation_loader, in_cnf):
    in_cnf.eval()
    for gamma, param in in_validation_loader:
        param = {k: pru.torch.update_device(v) for k, v in param.items()}
        gamma = pru.torch.update_device(gamma)
        in_ma.log(validation_nll=in_cnf.nll_mean(gamma, **param).item())


def run_main(in_run_parameters):
    data_model = measurements_distributions.LinearModel(in_run_parameters.d_x, in_run_parameters.d_p,
                                                        in_run_parameters.norm_min,
                                                        in_run_parameters.norm_max)
    train_loader, val_loader = measurements_distributions.generate_and_save_or_load_dataset(data_model,
                                                                                            in_run_parameters.base_dataset_folder,
                                                                                            in_run_parameters.batch_size,
                                                                                            in_run_parameters.dataset_size,
                                                                                            in_run_parameters.val_dataset_size,
                                                                                            force_data_generation=in_run_parameters.force_data_generation)
    cnf = flow_models.generate_cnf_model(in_run_parameters.d_x, in_run_parameters.d_p, [constants.THETA])
    m_step = len(train_loader)
    opt = torch.optim.Adam(cnf.parameters(), lr=in_run_parameters.lr, weight_decay=in_run_parameters.weight_decay)
    ma = pru.MetricAveraging()
    for e in range(in_run_parameters.n_epochs):
        flow_training_loop(m_step, train_loader, cnf, opt, ma)
        validation_run(ma, val_loader, cnf)
        results_log = ma.result
        pru.logger.info(f"Finished epoch:{e} training with results:{ma.results_str()}")
        ma.clear()
        if ma.is_best("validation_nll"):
            pru.logger.info(":) :) :) New Best !!!!")
            torch.save(cnf.state_dict(), os.path.join(wandb.run.dir, "flow_best.pt"))

    torch.save(cnf.state_dict(), os.path.join(wandb.run.dir, "flow_last.pt"))
    data_model.save_data_model(wandb.run.dir)


if __name__ == '__main__':
    cr = config()
    run_parameters, run_log_folder = pru.initialized_log(constants.PROJECT, cr, enable_wandb=True)
    run_main(run_parameters)

    pass
