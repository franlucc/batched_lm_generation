"""
Generates completions from a base model. This code borrows heavily from
MultiPL-E:

https://github.com/nuprl/MultiPL-E/blob/main/automodel.py

"""
from .base import GeneratorBase, partial_arg_parser, stop_at_stop_token
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List
import torch
import itertools
import json


class AutoModelGenerator(GeneratorBase):
    model_name: str
    model_kwargs: dict
    tokenizer: AutoTokenizer
    model: AutoModelForCausalLM

    def __init__(self, model_name: str, model_kwargs, **super_args):
        super().__init__(**super_args)
        self.model_name = model_name
        self.model_kwargs = model_kwargs

    def init_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, padding_side="left"
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        assert (
            self.tokenizer.pad_token is not None
        ), "tokenizer has neither pad_token nor eos_token"

        self.__all_special_token_ids = self.tokenizer.all_special_ids

        assert (
            len(self.__all_special_token_ids) >= 1
        ), "tokenizer.all_special_ids() is empty"
        assert (
            self.tokenizer.pad_token_id in self.__all_special_token_ids
        ), "pad_token_id not in all_special_ids"
        assert (
            self.tokenizer.eos_token_id in self.__all_special_token_ids
        ), "eos_token_id not in all_special_ids"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            **self.model_kwargs,
            device_map="cuda"
        )
        self.model.eval()

    def __completion_tensors(
        self,
        prompts: List[str],
        max_length: int,
        temperature: float,
        top_p: float,
    ) -> torch.Tensor:
        self.model.eval()  # Not essential, but just in case.

        inputs = self.tokenizer(
            prompts,
            padding=True,
            return_tensors="pt",
            return_token_type_ids=False,
            truncation=True,
            max_length=max_length - 1,
        ).to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                do_sample=True,
                use_cache=True,
                top_p=top_p,
                temperature=temperature,
                max_length=max_length,
                pad_token_id=self.tokenizer.pad_token_id
            )
        return output

    def __is_normal_token_id(self, token_id: int) -> bool:
        return token_id not in self.__all_special_token_ids

    def __is_pad_or_bos_token_id(self, token_id: int) -> bool:
        if token_id == self.tokenizer.pad_token_id:
            return True
        if (
            self.tokenizer.bos_token_id is not None
            and token_id == self.tokenizer.bos_token_id
        ):
            return True
        return False

    def __remove_padding_and_stop_at_special_tokens(
        self, token_id_list: List[int]
    ) -> List[int]:
        # Removes all the pad tokens or BOS tokens on the left-hand side using the
        # pad token ID. This is more robust than looking for the string representation of
        # the pad token. Thus the prompt can begin with the literal string
        # "<|endoftext|>" (which is a common representation of the pad token).
        left_padding_removed = itertools.dropwhile(
            self.__is_pad_or_bos_token_id, token_id_list
        )
        # Returns all tokens to the left of the first special token. This has
        # the effect of removing all right-hand padding. Moreover, it also
        # stops generation at other special tokens. For example, consider
        # StarCoder 2, where a completion may reach the end of a file and then
        # continue onto a second file: A<file_sep>B. The code below removes
        # <file_sep>B and only produces A.
        right_specials_removed = itertools.takewhile(
            self.__is_normal_token_id, left_padding_removed
        )
        return list(right_specials_removed)

    def decode_single_output(self, output_tensor, prompt):
        output_token_ids = self.__remove_padding_and_stop_at_special_tokens(
            output_tensor.tolist()
        )
        detok_hypo_str = self.tokenizer.decode(
            output_token_ids,
            clean_up_tokenization_spaces=False,
            skip_special_tokens=False,
        )
        # Skip the prompt (which may even have stop_tokens)
        return detok_hypo_str[len(prompt) :]

    def batch_generate(
        self,
        prompts: List[str],
        top_p: float,
        temperature: float,
        max_tokens: int,
        stop: List[str],
    ) -> List[str]:
        prompts = [prompt.strip() for prompt in prompts]
        output_tensors = self.__completion_tensors(
            prompts,
            max_tokens,
            temperature,
            top_p,
        )
        return [
            stop_at_stop_token(
                self.decode_single_output(output_tensor, prompt),
                stop,
            )
            for (prompt, output_tensor) in zip(prompts, output_tensors)
        ]


def main():
    parser = partial_arg_parser()
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--flash-attention2", action="store_true")
    parser.add_argument(
        "--stop", type=str, required=True, help="JSON list of stop tokens"
    )
    args = parser.parse_args()
    args.stop = json.loads(args.stop)

    model_kwargs = {}
    if args.flash_attention2:
        model_kwargs["attn_implementation"] = "flash_attention2"

    super_args = {
        k: v
        for (k, v) in vars(args).items()
        if k not in ["model_name", "flash_attention2"]
    }

    generator = AutoModelGenerator(args.model_name, model_kwargs, **super_args)
    generator.generate_all()


if __name__ == "__main__":
    main()
