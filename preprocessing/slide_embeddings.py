import gigapath.slide_encoder
import torch


def load_slide_encoder() -> torch.nn.Module:
    # TODO - fork and fix repo
    return gigapath.slide_encoder.create_model(
        "hf_hub:prov-gigapath/prov-gigapath",
        "gigapath_slide_enc12l768d",
        1536,
        global_pool=True,
    )


def load_dataset() -> None:
    pass


def main() -> None:
    _slide_encoder = load_slide_encoder()


if __name__ == "__main__":
    main()
