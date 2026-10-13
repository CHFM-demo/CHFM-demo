import logging

import torch
from transformers.data.data_collator import *


logger = logging.getLogger(__name__)

SUPPORTED_DECODER_MODELS = ['codegen', 'bloomz', 'gpt-neox', 'llama']
SUPPORTED_SEQ2SEQ_MODELS = ['t5', 'flan-t5']


def check_model(model_name, supported_models):
    for sup_model in supported_models:
        if sup_model.lower() in model_name.lower():
            return True

    return False


@dataclass
class DataCollatorForUIE:
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_source_length: Optional[int] = None
    max_target_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    add_task_name: bool = False
    add_dataset_name: bool = False
    common_dataset_name: str = None
    text_only: bool = False
    num_examples: int = 0
    input_record_file: str = None

    def __call__(self, batch, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors

        if getattr(self.model.config, "model_type", None) == "llama":
            return self.llama_call(batch, return_tensors)

        model_name = (getattr(self.model.config, "model_type", "") + " "
                      + self.model.config._name_or_path)
        # print(model_name)
        if check_model(model_name, SUPPORTED_DECODER_MODELS):
            model_inputs = self.decoder_call(batch, return_tensors)
        elif check_model(model_name, SUPPORTED_SEQ2SEQ_MODELS):
            model_inputs = self.seq2seq_call(batch, return_tensors)
        else:
            raise ValueError('Unsupport model {}!'.format(model_name))

        return model_inputs

    def get_instruction(self, instance):
        # "instructions \n options \n {0} \n Answer: "
        instruction = instance['Instance']["instruction"]
        content = instance['Instance']['sentence']

        # add task/ds prefix
        prefix = ''
        if self.add_task_name:
            prefix += "Task:" + instance['Task'] + '\n'
        if self.add_dataset_name:
            ds_name = self.common_dataset_name if self.common_dataset_name else instance['Dataset']
            prefix = prefix + "Dataset:"
            prefix = prefix + ds_name + '\n' if prefix else instance['Dataset'] + '\n'
        if prefix:
            instruction = prefix + instruction

        # TODO, support few shot
        # add few shot samples
        samples = ''
        if len(instance['Samples']) > 0:
            raise Exception('Few shot is coming soon...')
        if samples:
            content = samples + content
        # TODO, fix bug
        try:
            instruction = instruction.format(content)
        finally:
            return instruction


    def seq2seq_call(self, batch, return_tensors):
        sources = []
        labels = []

        for instance in batch:
            label = instance['Instance']['label']
            labels.append(label)
            instruction = self.get_instruction(instance)

            source = instruction
            tokenized_source = self.tokenizer(source)["input_ids"]
            if len(tokenized_source) <= self.max_source_length:
                sources.append(source)
            else:
                sources.append(self.tokenizer.decode(tokenized_source[:self.max_source_length], skip_special_tokens=True))

        # TODO, support online demo
        if self.text_only:
            model_inputs = {"inputs": sources, "labels": labels}
        else:
            model_inputs = self.tokenizer(
                sources,
                max_length=self.max_source_length,
                padding=self.padding,
                return_tensors=return_tensors,
                truncation=True,
                pad_to_multiple_of=self.pad_to_multiple_of
            )
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(
                    labels,
                    max_length=self.max_target_length,
                    padding=self.padding,
                    return_tensors=return_tensors,
                    truncation=True,
                    pad_to_multiple_of=self.pad_to_multiple_of
                )
            label_mask = labels["attention_mask"].bool()
            model_inputs["labels"] = labels["input_ids"].masked_fill(~label_mask, self.label_pad_token_id)

            # prepare decoder_input_ids
            if self.model is not None:
                decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=model_inputs["labels"])
                model_inputs["decoder_input_ids"] = decoder_input_ids

            self._save_samples(model_inputs, sources, labels)

        return model_inputs

    def llama_call(self, batch, return_tensors):
        """Tokenize prompt/answer separately so truncation cannot discard the answer."""
        if self.max_source_length < 1 or self.max_target_length < 1:
            raise ValueError("Llama source and target lengths must be positive")
        self.tokenizer.padding_side = "left"
        sequences, targets, sources, answers = [], [], [], []
        for instance in batch:
            prompt = self.get_instruction(instance)
            answer = instance["Instance"]["label"]
            sources.append(prompt)
            answers.append(answer)
            prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
            bos = [self.tokenizer.bos_token_id] if self.tokenizer.bos_token_id is not None else []
            # Preserve the end of the instruction (including Answer:) for generation.
            budget = self.max_source_length - len(bos)
            prompt_ids = bos + (prompt_ids[-budget:] if budget > 0 else [])
            answer_ids = []
            if instance["subset"] == "train":
                answer_ids = self.tokenizer(answer, add_special_tokens=False)["input_ids"]
                answer_ids = answer_ids[:max(0, self.max_target_length - 1)]
                answer_ids.append(self.tokenizer.eos_token_id)
            sequences.append({"input_ids": prompt_ids + answer_ids})
            targets.append([-100] * len(prompt_ids) + answer_ids)
        if self.text_only:
            return {"inputs": sources, "labels": answers}
        model_inputs = self.tokenizer.pad(
            sequences, padding=self.padding, pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors)
        width = model_inputs["input_ids"].shape[1]
        model_inputs["labels"] = torch.tensor(
            [[-100] * (width - len(row)) + row for row in targets], dtype=torch.long)
        # The custom Llama forward uses a mask on prediction positions (before shifting).
        loss_mask = torch.zeros_like(model_inputs["input_ids"], dtype=torch.float)
        loss_mask[:, :-1] = (model_inputs["labels"][:, 1:] != -100).float()
        model_inputs["loss_mask"] = loss_mask
        self._save_samples(model_inputs, sources, answers)
        return model_inputs

    def decoder_call(self, batch, return_tensors):
        self.tokenizer.padding_side = 'left'
        sources = []
        label_lens = []
        labels = []
        max_len = -1
        if batch[0]['subset'] == "train":
            limit_input_len = self.max_source_length + self.max_target_length
        else:
            limit_input_len = self.max_source_length

        for instance in batch:
            label = instance['Instance']['label']
            labels.append(label)
            instruction = self.get_instruction(instance)

            # add bos and eos
            task_input = self.tokenizer.bos_token + instruction
            label = label + self.tokenizer.eos_token

            tokenized_input = self.tokenizer(task_input)["input_ids"]
            tokenized_label = self.tokenizer(label)["input_ids"]

            # (input) for inference, (input + label) for training
            if instance['subset'] in ['dev', 'test']:
                label_lens.append(0)
                if len(tokenized_input) <= limit_input_len:
                    max_len = max(len(tokenized_input), max_len)
                    sources.append(task_input)
                else:
                    max_len = limit_input_len
                    input_wo_label = self.tokenizer.decode(
                        tokenized_input[: limit_input_len],
                        skip_special_tokens=False
                    )
                    sources.append(input_wo_label)
            else:
                if len(tokenized_input) + len(tokenized_label) <= limit_input_len:
                    max_len = max(len(tokenized_input) + len(tokenized_label), max_len)
                    label_lens.append(len(tokenized_label))
                    sources.append(task_input + label)
                else:
                    max_len = self.max_source_length
                    input_w_label = self.tokenizer.decode(
                        (tokenized_input + tokenized_label)[: limit_input_len],
                        skip_special_tokens=False
                    )
                    sources.append(input_w_label)
                    label_lens.append(max(0, limit_input_len - len(tokenized_input)))

        # TODO, support online demo
        if self.text_only:
            model_inputs = {"inputs": sources, 'labels': labels}
        else:
            model_inputs = self.tokenizer(
                sources,
                max_length=self.max_source_length,
                padding=self.padding,
                return_tensors=return_tensors,
                truncation=True,
                pad_to_multiple_of=self.pad_to_multiple_of
            )

            label_mask = model_inputs["attention_mask"].bool()
            model_inputs["labels"] = model_inputs['input_ids'].masked_fill(~label_mask, self.label_pad_token_id)

            # loss mask
            max_len = min(max_len, limit_input_len)
            loss_mask = torch.ones((label_mask.shape))
            for k, label_len in enumerate(label_lens):
                loss_mask[k, : max_len - label_len - 1] = 0
            model_inputs['loss_mask'] = loss_mask.masked_fill(~label_mask, 0)

            self._save_samples(model_inputs, sources, labels)

        return model_inputs

    def _save_samples(self, model_inputs, sources, labels):
        if not self.input_record_file:
            return

        loss_label = []
        if hasattr(model_inputs, 'loss_mask'):
            for loss, id in zip(model_inputs.loss_mask, model_inputs.input_ids):
                loss_label.append(self.tokenizer.decode((loss * id).view(-1).int()))

            with open(self.input_record_file, 'a+', encoding='utf-8') as f:
                for text, label, mask_label in zip(sources, labels, loss_label):
                    f.write(text+'\n')
                    f.write(label + '\n')
                    f.write(mask_label+'\n\n')
        else:
            with open(self.input_record_file, 'a+', encoding='utf-8') as f:
                for text, label in zip(sources, labels['input_ids']):
                    f.write(text + '\n')
                    f.write(self.tokenizer.decode(label, clean_up_tokenization_spaces=False) + '\n')