import pyresearchutils  as pru
import measurements_distributions
from experiments import constants


def config() -> pru.ConfigReader:
    _cr = pru.initialized_config_reader(default_base_log_folder=os.path.join("./temp", "logs"))
    ###############################################
    # Training
    ###############################################
    _cr.add_parameter("n_epochs", type=int, default=180)
    _cr.add_parameter("lr", type=float, default=1.6e-3)
    _cr.add_parameter("grad_norm_clipping", type=float, default=0.1)
    _cr.add_parameter("weight_decay", type=float, default=1e-4)

    _cr.add_parameter("random_padding", type=str, default="false")
    _cr.add_parameter("padding_size", type=int, default=0)
    _cr.add_parameter("quantization_preprocessing", type=str, default="true")
    _cr.add_parameter("variational_dequantization", type=str, default="false")

    ###############################################
    # Signal Model Parameter
    ###############################################
    _cr.add_parameter("dim", type=int, default=16)
    _cr.add_parameter("bit_width", type=int, default=1)
    _cr.add_parameter("threshold", type=float, default=1.0)
    _cr.add_parameter("snr_min", type=float, default=2.0)
    _cr.add_parameter("snr_max", type=float, default=2.0)
    _cr.add_parameter('quantization', type=str, default="true")
    _cr.add_parameter('base_model_folder', type=str, default="./temp")
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
    _cr.add_parameter("n_blocks", type=int, default=3)
    _cr.add_parameter("coupling_type", type=str, default="Affine", enum=flow_model.CouplingType)

    return _cr


def run_main(run_parameters):
    measurements_distributions.LinearModel(d_x, d_p, )


if __name__ == '__main__':
    cr = config()
    run_parameters, run_log_folder = pru.initialized_log(constants.PROJECT, cr, enable_wandb=True)
    run_main(run_parameters)

    pass
