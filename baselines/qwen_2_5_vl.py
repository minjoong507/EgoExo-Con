import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from utils.vision_process import process_vision_info
from utils.cons_utils import *
from prompts.qwen import prompt



class Qwen2_5_VL_Options(BaseOptions):
    def __init__(self):
        super(Qwen2_5_VL_Options, self).__init__()
        BaseOptions.initialize(self)

class Qwen2_5_VL:
    def __init__(self, args):
        self.args = args
        self.model_path = args.model_path
        self.debug = args.debug
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.set_default_device(self.device)

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            attn_implementation=None,
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.max_new_tokens = 256
        self.system = prompt["system"]

    def extract_time(self, paragraph):
        pattern = r'(?:(?:from\s+|in\s+|between\s+|at\s+|—|to\s+|and\s+|–)\s*)?(\d+(?:\.\d+)?)\s*(?:to|and|–|—)\s*(?:(?:from\s+|in\s+|between\s+|at\s+|—|to\s+|and\s+|–)\s*)?(\d+(?:\.\d+)?)?\s*seconds?'
        match = re.search(pattern, paragraph)

        if match:
            start_timestamp = float(match.group(1))
            # If only the start timestamp is found, treat it as the end time.
            if match.group(2) is None:
                return [0, start_timestamp]
            else:
                return [start_timestamp, float(match.group(2))]

        else:
            pattern = r'\d+\.\d+|\d+'

            # Find all matches in the input string
            matches = re.findall(pattern, paragraph)

            # Convert matched strings to float
            float_numbers = [float(match) for match in matches]
            if len(float_numbers) == 0:
                return [0, 0]
            elif len(float_numbers) == 1:
                return [0, float_numbers[0]]
            else:
                return float_numbers[:2]


    def run(self, task, video_paths, queries, timestamps=None, lengths=None):
        questions, choices = generate_questions(task, prompt, queries, timestamps)
        messages = [
            [
                {
                    "role": "system",
                    "content": self.system,
                 },
                {
                    "role": "user",
                    "content": [
                        {"video": video_path, "max_pixels": 16*28*28, "fps": 1.0},
                        {"type": "text", "text": question},
                    ],
                },
            ]
            for video_path, question in zip(video_paths, questions)
        ]

        self.processor.padding_side = 'left'
        texts = [self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(text=texts,
                                images=image_inputs,
                                videos=video_inputs,
                                padding=True,
                                return_tensors="pt")
        inputs = inputs.to(self.device)



        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_beams=1,
                use_cache=True,
            )
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        responses = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        if self.debug:
            print(f"Q: {questions[0]}: A: {responses[0]}")

        if task in ["grounding"]:
            timestamps = []
            for response in responses:
                try:
                    timestamps.append(self.extract_time(response))
                except:
                    timestamps.append([0, 0])
            return [{"q": question, "a": response, "t": timestamp} for question, response, timestamp in
                    zip(questions, responses, timestamps)]

        elif task in ["verif"]:
            return [{"q": question, "a": response, "c": choice} for question, response, choice in
                    zip(questions, responses, choices)]

        else:
            raise NotImplementedError(f"Task {task} is not yet implemented.")

