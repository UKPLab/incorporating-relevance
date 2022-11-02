import torch
from sklearn.metrics import accuracy_score


def binary_accuracy_from_logits(y_true, logits):
    y_true = y_true.detach().cpu()
    logits = logits.detach().cpu()
    probs = torch.sigmoid(logits)
    accuracy = accuracy_score(y_true.view(-1).long(), probs.view(-1) >= 0.5)
    return accuracy


def batch_iter(
    batch_size: int,
    input_ids,
    token_type_ids,
    attention_mask,
    targets,
    shuffle: bool = False,
):
    if shuffle:
        p = torch.randperm(n=input_ids.size(0), device=input_ids.device)
        input_ids = input_ids[p]
        token_type_ids = token_type_ids[p]
        attention_mask = attention_mask[p]
        targets = targets[p]

    for batch_i in range(0, input_ids.size(0), batch_size):
        batch_input_ids = input_ids[batch_i : batch_i + batch_size]
        batch_token_type_ids = token_type_ids[batch_i : batch_i + batch_size]
        batch_attention_mask = attention_mask[batch_i : batch_i + batch_size]
        batch_targets = targets[batch_i : batch_i + batch_size]

        yield batch_i, (
            batch_input_ids,
            batch_token_type_ids,
            batch_attention_mask,
        ), batch_targets


class InnerLRScheduler:
    def __init__(self, lr, num_warmup_steps, num_training_steps):
        self.lr = lr
        self.current_step = 0
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps

    def step(self):
        self.current_step += 1

    def get_lr(self):
        # https://huggingface.co/transformers/_modules/transformers/optimization.html#get_linear_schedule_with_warmup
        if self.current_step < self.num_warmup_steps:
            lmdb = float(self.current_step) / float(max(1, self.num_warmup_steps))
        else:
            lmdb = max(
                0.0,
                float(self.num_training_steps - self.current_step)
                / float(max(1, self.num_training_steps - self.num_warmup_steps)),
            )
        return lmdb * self.lr
