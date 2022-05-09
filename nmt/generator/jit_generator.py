import torch


class JITGenerator(object):
    def __init__(self, model_path, device) -> None:
        self.model = torch.jit.load(model_path, map_location=device)
        self.model.to(device)
        self.device = device

    def decode(self, src_input):
        src_input = src_input.to(self.device)
        output = self.model(src_input)
        return output.tolist()