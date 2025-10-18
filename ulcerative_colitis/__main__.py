from random import randint

import hydra
import pyvips  # noqa: F401
from lightning import seed_everything
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf
from rationai.mlkit import Trainer, autolog

from ulcerative_colitis.data import DataModule

# from ulcerative_colitis.ulcerative_colitis_attention_mil_multiclass import (
#     UlcerativeColitisModelAttentionMILMulticlass,
# )
# from ulcerative_colitis.ulcerative_colitis_attention_mil import (
#     UlcerativeColitisModelAttentionMIL,
# )
from ulcerative_colitis.ulcerative_colitis_slide_embeddings import (
    UlcerativeColitisModelSlideEmbeddings,
)


OmegaConf.register_new_resolver(
    "random_seed", lambda: randint(0, 2**31), use_cache=True
)


@hydra.main(config_path="../configs", config_name="default", version_base=None)
@autolog
def main(config: DictConfig, logger: Logger | None) -> None:
    seed_everything(config.seed, workers=True)

    data = hydra.utils.instantiate(
        config.data,
        _recursive_=False,  # to avoid instantiating all the datasets
        _target_=DataModule,
    )
    model = hydra.utils.instantiate(
        config.model,
        _target_=UlcerativeColitisModelSlideEmbeddings,
    )

    trainer = hydra.utils.instantiate(config.trainer, _target_=Trainer, logger=logger)
    getattr(trainer, config.mode)(model, datamodule=data, ckpt_path=config.checkpoint)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
