import pyresearchutils  as pru
from experiments import measurements_distributions
import os
import torch
from experiments import constants
from experiments import flow_models
from tqdm import tqdm
import wandb


def config() -> pru.ConfigReader:
    _cr = pru.initialized_config_reader(default_base_log_folder=os.path.join("./temp", "logs"))
    ###############################################
    # Training
    ###############################################
    _cr.add_parameter("base_epochs", type=int, default=360)
    _cr.add_parameter('base_dataset_size', type=int, default=200000)
    _cr.add_parameter("n_validation_points", type=int, default=180)
    _cr.add_parameter("lr", type=float, default=2e-4)
    _cr.add_parameter("grad_norm_clipping", type=float, default=0.0)
    _cr.add_parameter("weight_decay", type=float, default=0.0)

    _cr.add_parameter("random_padding", type=str, default="false")
    _cr.add_parameter("padding_size", type=int, default=0)
    ###############################################
    # CNF Parameters
    ###############################################
    _cr.add_parameter("n_blocks", type=int, default=1)
    _cr.add_parameter("n_layer_inject", type=int, default=1)
    _cr.add_parameter("n_hidden_inject", type=int, default=16)
    _cr.add_parameter("inject_scale", type=str, default="false")
    _cr.add_parameter("inject_bias", type=str, default="false")
    ###############################################
    # Signal Model Parameter
    ###############################################
    _cr.add_parameter("model_name", type=str, default="LinearGaussian", enum=measurements_distributions.ModelName)
    _cr.add_parameter("d_x", type=int, default=8)
    _cr.add_parameter("d_p", type=int, default=2)
    _cr.add_parameter("norm_min", type=float, default=0.1)
    _cr.add_parameter("norm_max", type=float, default=10.0)
    _cr.add_parameter("min_limit", type=float, default=-5.0)
    _cr.add_parameter("max_limit", type=float, default=5.0)
    ###############################################
    # Dataset Parameters
    ###############################################
    _cr.add_parameter('base_dataset_folder', type=str, default="./temp/datasets")
    _cr.add_parameter('batch_size', type=int, default=512)
    _cr.add_parameter('dataset_size', type=int, default=200000)  # 200000
    _cr.add_parameter('val_dataset_size', type=int, default=20000)
    _cr.add_parameter('force_data_generation', type=str, default="false")

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
    pru.set_seed(0)
    data_model = measurements_distributions.get_measurement_distribution(
        in_run_parameters.model_name,
        d_x=in_run_parameters.d_x,
        d_p=in_run_parameters.d_p,
        norm_min=in_run_parameters.norm_min,
        norm_max=in_run_parameters.norm_max,
        a_limit=in_run_parameters.min_limit,
        b_limit=in_run_parameters.max_limit)
    measurements_distributions.save_or_load_model(data_model, in_run_parameters.base_dataset_folder)
    train_loader, val_loader = measurements_distributions.generate_and_save_or_load_dataset(data_model,
                                                                                            in_run_parameters.base_dataset_folder,
                                                                                            in_run_parameters.batch_size,
                                                                                            in_run_parameters.dataset_size,
                                                                                            in_run_parameters.val_dataset_size,
                                                                                            force_data_generation=in_run_parameters.force_data_generation)
    n_epochs = in_run_parameters.base_epochs * int(in_run_parameters.base_dataset_size / in_run_parameters.dataset_size)
    pru.logger.info(f"Number of epochs:{n_epochs}")
    cnf = flow_models.generate_cnf_model(in_run_parameters.d_x,
                                         in_run_parameters.d_p,
                                         [constants.THETA],
                                         n_blocks=in_run_parameters.n_blocks,
                                         n_layer_inject=in_run_parameters.n_layer_inject,
                                         n_hidden_inject=in_run_parameters.n_hidden_inject,
                                         inject_scale=in_run_parameters.inject_scale,
                                         inject_bias=in_run_parameters.inject_bias)
    m_step = len(train_loader)
    opt = torch.optim.Adam(cnf.parameters(), lr=in_run_parameters.lr, weight_decay=in_run_parameters.weight_decay)
    ma = pru.MetricAveraging()
    validation_mod = int(n_epochs / in_run_parameters.n_validation_points)
    for e in range(n_epochs):
        flow_training_loop(m_step, train_loader, cnf, opt, ma)
        if e % validation_mod == 0:
            validation_run(ma, val_loader, cnf)
        results_log = ma.result
        pru.logger.info(f"Finished epoch:{e} of {n_epochs} training with results:{ma.results_str()}")
        ma.clear()
        wandb.log(results_log)
        if e % validation_mod == 0 and ma.is_best("validation_nll"):
            pru.logger.info(":) :) :) New Best !!!!")
            torch.save(cnf.state_dict(), os.path.join(wandb.run.dir, "flow_best.pt"))

    torch.save(cnf.state_dict(), os.path.join(wandb.run.dir, "flow_last.pt"))
    data_model.save_data_model(wandb.run.dir)


if __name__ == '__main__':
    cr = config()
    run_parameters, run_log_folder = pru.initialized_log(constants.PROJECT, cr, enable_wandb=True)
    run_main(run_parameters)

    pass
