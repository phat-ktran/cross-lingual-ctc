import copy

__all__ = ["build_model"]


def build_model(config):
    config = copy.deepcopy(config)

    from .rec_svtrnet import SVTRNet
    from .rec_crnn import CRNN

    support_dict = ["SVTRNet", "CRNN"]

    module_name = config.pop("name")
    assert module_name in support_dict, Exception(
        "when model typs is {}, model only support {}".format(support_dict)
    )
    module_class = eval(module_name)(**config)
    return module_class
