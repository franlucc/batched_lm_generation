
from .base import GeneratorBase, partial_arg_parser
from PIL import Image
from io import BytesIO
from typing import List, Tuple
import base64
from openai import OpenAI
import os

def encode_image(image):
   with BytesIO() as output:
    image.save(output, format="PNG")
    return base64.b64encode(output.getvalue()).decode('utf-8')

class GPTModel(GeneratorBase):
    model_name: str
    model_kwargs: dict
    client: OpenAI

    def __init__(self, model_name: str, **super_args):
      super().__init__(**super_args)
      self.model_name = model_name
      self.client = None

    def init_model(self):
      key = os.getenv("OPENAI_API_KEY")
      self.client = OpenAI(api_key=key)

    # Each prompt is a tuple with a text prompt and an image.
    def batch_generate(
        self,
        prompts: List[Tuple[str, Image.Image]],
        top_p: float,
        temperature: float,
        max_tokens: int,
        stop: List[str],
    ) -> List[str]:
      generated_texts = []
      for item in prompts:
        text = item[0]
        image = item[1]
        base64_image = encode_image(image)
        response = self.client.chat.completions.create(
        model="gpt-4o",
        messages=[
          {"role": "system", "content": "Reply with the format: Statement {Option} is correct."},
          {
            "role": "user",
            "content": [
              {"type": "text", "text": text},
              {
                "type": "image_url",
                "image_url": {
                  "url": f"data:image/jpeg;base64,{base64_image}",
                },
              },
            ],
          }
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        n=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=stop
        )
        answer = response.choices[0].message.content
        generated_texts.append(answer)
      return generated_texts
        
    
def main(): 
    parser = partial_arg_parser()
    parser.add_argument("--model-name", type=str, required=True)
    args = parser.parse_args()

    EXTRA_ARGS = ["model_name"]
    super_args = {k: v for (k, v) in vars(args).items() if k not in EXTRA_ARGS}

    generator = GPTModel(model_name=args.model_name, stop=["\n\n\n"], **super_args)
    generator.generate_all()


if __name__ == "__main__":
    main()
