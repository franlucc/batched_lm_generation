"""
Generates code completions from a chat model.

Given PROMPT, the conversation is set up as:

    System: You are an expert programmer.

    User: Complete the following function:

    ```
    PROMPT
    ```

    Assistant: Sure, here is the code:

    ```

Notice that we force the model to begin its response with "Sure, here is
the code" followed by a code block. This forces a terse response. We return
contents of the complete code block, which is likely to repeat the PROMPT
itself.
"""

from .base import GeneratorBase, partial_arg_parser, stop_at_stop_token
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria
from typing import List
import torch


def prompt_to_messages(prompt: str) -> List[dict]:
    return [
        {"role": "system", "content": "You are an expert programmer."},
        {"role": "user", "content": f"Complete the following function:\n\n```\n{prompt}\n```"},
    ]


class CustomStoppingCriteria(StoppingCriteria):
    def __init__(self, end_markdown_tokens: torch.Tensor, eos_token_id: int):
        self.__stopped_indices__ = None
        self.__end_markdown_tokens = end_markdown_tokens
        self.__end_markdown_tokens_lens = self.__end_markdown_tokens.shape[0]
        self.__eos_token_id = eos_token_id

    def __call__(
        self, input_ids: torch.LongTensor, score: torch.FloatTensor, **kwargs
    ) -> bool:
        if self.__stopped_indices__ is None:
            self.__stopped_indices__ = torch.zeros_like(
                input_ids[:, -1], dtype=torch.bool, device=input_ids.device
            )

        is_eos = input_ids[:, -1] == self.__eos_token_id
        is_end_markdown = torch.all(
            input_ids[:, -self.__end_markdown_tokens_lens :]
            == self.__end_markdown_tokens,
            dim=1,
        )
        new_stopped_indices = torch.logical_or(is_eos, is_end_markdown)
        self.__stopped_indices__ = torch.logical_or(
            self.__stopped_indices__, new_stopped_indices
        )
        return self.__stopped_indices__.all()


class ChatCoder(GeneratorBase):
    model_name: str
    model_kwargs: dict
    tokenizer: AutoTokenizer
    model: AutoModelForCausalLM

    def __init__(self, model_name: str, **super_args):
        super().__init__(**super_args)
        self.model_name = model_name

    def init_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="cuda",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, padding_side="left"
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.__end_markdown_tokens = self.tokenizer.encode(
            "\n```", add_special_tokens=False, return_tensors="pt"
        )[0].to(self.model.device)

    def batch_generate(
        self,
        prompts: List[str],
        top_p: float,
        temperature: float,
        max_tokens: int,
        stop: List[str],
    ) -> List[str]:
        assert len(stop) == 0, "stop tokens not supported with chat models"

        chat_prompts = self.tokenizer.apply_chat_template(
            [prompt_to_messages(p) for p in prompts],
            add_generation_prompt=True,
            tokenize=False,
        )
        chat_prompts = [f"{p}Sure, here is the code.\n\n```" for p in chat_prompts]
        inputs = self.tokenizer(chat_prompts, return_tensors="pt", padding=True).to(
            self.model.device
        )
        stopping_criteria = CustomStoppingCriteria(
            self.__end_markdown_tokens, self.tokenizer.eos_token_id
        )
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_length=max_tokens,
                top_p=top_p,
                do_sample=True,
                temperature=temperature,
                stopping_criteria=[stopping_criteria],
                pad_token_id=self.tokenizer.eos_token_id,
            )
        output_ids = output_ids[:, inputs.input_ids.shape[1] :]

        output_texts = self.tokenizer.batch_decode(
            output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        # The prompt ends with ```. The completion either begins with \n or
        # LANG\n. This strips that out.
        output_texts = [t.split("\n", maxsplit=1)[1] for t in output_texts]
        output_texts = [stop_at_stop_token(t, ["\n```"]) for t in output_texts]
        return output_texts


def main():
    parser = partial_arg_parser()
    parser.add_argument("--model-name", type=str, required=True)
    args = parser.parse_args()

    EXTRA_ARGS = ["model_name"]
    super_args = {k: v for (k, v) in vars(args).items() if k not in EXTRA_ARGS}

    generator = ChatCoder(model_name=args.model_name, stop=[], **super_args)
    generator.generate_all()


if __name__ == "__main__":
    main()
