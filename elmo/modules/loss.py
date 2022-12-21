import mindspore
import mindspore.nn as nn
import mindspore.ops as P
import numpy as np
from mindspore import Tensor, Parameter, ms_function
# from elmo.ops.sampled_softmax_loss import SampledSoftmaxLoss
from elmo.ops.SampledSoftmaxLoss import  SampledSoftmaxLoss
from mindspore.common.initializer import initializer, Normal, Zero

class LossCell(nn.Cell):
    def __init__(
            self, 
            hidden_size,
            vocab_size, 
            sample_softmax, 
            num_sampled, 
            num_true=1,
            seed=0,
            training=True):
        super().__init__()
        self.training = training
        self.sample_softmax = sample_softmax
        self.hidden_size = hidden_size

        self.weight = Parameter(initializer(Normal(1.0 / np.sqrt(hidden_size)), (vocab_size, hidden_size), mindspore.float32))
        self.bias = Parameter(initializer(Zero(), (vocab_size), mindspore.float32))

        self.sampled_softmax_loss = SampledSoftmaxLoss(num_sampled, vocab_size, num_true, seed=seed)
        self.sparse_softmax_cross_entropy_with_logits = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
        self.matmul = nn.MatMul(False, True)
        self.reduce_mean = P.ReduceMean()
    
    #@ms_function
    def construct(self, lstm_outputs, next_ids):
        total_loss = []
        for lstm_output, next_token_id in zip(lstm_outputs, next_ids):
            next_token_id_flat = next_token_id.view((-1, 1))
            if self.training and self.sample_softmax:
                lstm_output = lstm_output.view((-1, self.hidden_size))
                loss = self.sampled_softmax_loss(self.weight, self.bias, next_token_id_flat, lstm_output)
            else:
                next_token_id_flat = P.Squeeze(1)(next_token_id_flat)
                output_scores = self.matmul(lstm_output, self.weight) + self.bias
                output_scores = output_scores.view((-1, output_scores.shape[-1]))
                loss = self.sparse_softmax_cross_entropy_with_logits(output_scores, next_token_id_flat)
            total_loss.append(self.reduce_mean(loss))
        
        return 0.5 * (total_loss[0] + total_loss[1]) * 20