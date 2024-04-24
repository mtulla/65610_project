import brevitas.nn as qnn
import torch.nn as nn
import torch
from concrete.ml.torch.compile import compile_brevitas_qat_model

N_FEAT = 12
n_bits = 3

class QATSimpleNet(nn.Module):
    def __init__(self, n_hidden):
        super().__init__()

        self.quant_inp = qnn.QuantIdentity(bit_width=n_bits, return_quant_tensor=True)
        self.fc1 = qnn.QuantLinear(N_FEAT, n_hidden, True, weight_bit_width=n_bits, bias_quant=None)
        self.quant2 = qnn.QuantIdentity(bit_width=n_bits, return_quant_tensor=True)
        self.fc2 = qnn.QuantLinear(n_hidden, n_hidden, True, weight_bit_width=n_bits, bias_quant=None)
        self.quant3 = qnn.QuantIdentity(bit_width=n_bits, return_quant_tensor=True)
        self.fc3 = qnn.QuantLinear(n_hidden, 2, True, weight_bit_width=n_bits, bias_quant=None)

    def forward(self, x):
        x = self.quant_inp(x)
        x = self.quant2(torch.relu(self.fc1(x)))
        x = self.quant3(torch.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


print(1)
torch_input = torch.randn(100, N_FEAT)
print(2)
torch_model = QATSimpleNet(30)
print(3)
quantized_module = compile_brevitas_qat_model(
    torch_model, # our model
    torch_input, # a representative input-set to be used for both quantization and compilation
)
print(4)

x_test = numpy.array([numpy.random.randn(N_FEAT)])

y_pred = quantized_module.forward(x_test, fhe="execute")