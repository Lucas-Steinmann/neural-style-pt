import os
import os.path as osp
import shutil
from typing import Dict

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from multiscale_strategies import MULTISCALE_STRATEGIES, MultiScaleParamStrategy
from neural_style import neural_style


def gen_step_param_strategies(config: DictConfig) -> Dict[str, MultiScaleParamStrategy]:
    strategies = {}
    for key, val in config.items():
        strategies[key] = MULTISCALE_STRATEGIES.build(val)
    return strategies


def output_image_name(step: int) -> str:
    return f'step_{step}.png'


CONTENT_IMAGE_NAME = "content.png"
STYLE_IMAGE_NAME = "style.png"


@hydra.main(version_base=None, config_path='.', config_name='multiscale_config.yaml')
def multiscale_transfer(cfg: DictConfig):
    print(f"Working directory: {os.getcwd()}")
    steps = cfg.multiscale_steps
    # copy input images to working directory
    orig_content_image = osp.join(get_original_cwd(), cfg.content_image)
    orig_style_image = osp.join(get_original_cwd(), cfg.style_image)
    shutil.copy2(orig_content_image, CONTENT_IMAGE_NAME)
    shutil.copy2(orig_style_image, STYLE_IMAGE_NAME)

    # generate strategies from config
    param_strategies = gen_step_param_strategies(cfg.neural_style)
    for step in range(steps):
        # generate the new parameters
        step_params = DictConfig({key: strategy.compute() for key, strategy in param_strategies.items()})
        # model file is given relative to project root. hotfix the path
        step_params.model_file = osp.join(get_original_cwd(), step_params.model_file)
        # in the first step use content image, otherwise use output from last step
        step_params.content_image = CONTENT_IMAGE_NAME if step == 0 else output_image_name(step - 1)
        # style image is constant
        step_params.style_image = STYLE_IMAGE_NAME
        # use content image as init image
        step_params.init = 'image'
        step_params.init_image = step_params.content_image
        # automatically generate output image name
        step_params.output_image = output_image_name(step)
        # run the step
        neural_style(step_params)
        # inform parameter strategies that the step is finished
        for strategy in param_strategies.values():
            strategy.step()


if __name__ == "__main__":
    multiscale_transfer()
