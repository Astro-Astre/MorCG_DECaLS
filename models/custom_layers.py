from torch import Tensor, nn, functional

# this would be more elegant, but sadly using a stack with multiple sequential models
# (as opposed to some layers, then a sequential model)
# seems to silently break loading weights

class PermaDropout(nn.modules.dropout._DropoutNd):
    # https://pytorch.org/docs/stable/_modules/torch/nn/modules/dropout.html#Dropout
    def forward(self, input: Tensor) -> Tensor:
            return nn.functional.dropout(input, self.p, True, self.inplace)  # simply replaced self.training with True
