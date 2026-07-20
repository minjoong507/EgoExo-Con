import copy
import json
import re
import numpy as np
import argparse
import pandas as pd
import random
import os
import logging


# ANSI escape codes for colors
class Formatter(logging.Formatter):
    COLOR_CODES = {
        'DEBUG': '\033[94m',    # Blue
        'INFO': '\033[92m',     # Green
        'WARNING': '\033[91m',  # Red
        'ERROR': '\033[93m',  # Yellow
        'CRITICAL': '\033[95m', # Magenta
    }
    RESET_CODE = '\033[0m'  # Reset color

    def format(self, record):
        log_color = self.COLOR_CODES.get(record.levelname, self.RESET_CODE)
        message = super().format(record)
        return f"{log_color}{message}{self.RESET_CODE}"


def load_logger(name):
    # Custom logger setup
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Console handler with color formatter
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # Define formatter with color
    formatter = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(console_handler)
    return logger


class BaseOptions(object):
    def __init__(self):
        self.parser = None
        self.initialized = False
        self.opt = None


    def initialize(self):
        self.initialized = True
        parser = argparse.ArgumentParser()
        parser.add_argument("--model_type", type=str,
                            default="qwen2_5",
                            choices=['qwen2_5'],
                            help="Video large language model to evaluate.")
        parser.add_argument("--task",
                            type=str,
                            default="verif", choices=['verif', 'grounding'],
                            help="Type of task.")
        parser.add_argument("--split_list", type=list, default=["CharadesEgo", "EgoExo4D", "Lemma"], help="List of splits that we support.")
        parser.add_argument('--description', action="store_true",
                            help="Prompt the model to generate a video description before performing target tasks.")
        parser.add_argument('--fine_tuned', action="store_true")
        parser.add_argument('--split', type=str, default="all", choices=["all", "CharadesEgo", "EgoExo4D", "Lemma"], help="We can choose the split to use. If it set as 'all', we evaluate a model for whole set.")
        parser.add_argument('--iou_thd', type=float, default=0.5)
        parser.add_argument('--resume', action="store_true")
        parser.add_argument("--video_root", type=str, default="video_datasets/", help="path to video files")
        parser.add_argument("--model_path", type=str, default=None, help="path to model")
        parser.add_argument("--output_dir", type=str, default=None, help="path to output files")
        parser.add_argument("--exp_id", type=str, default=None, help="ID of this run.")
        parser.add_argument("--seed", type=int, default=1000)
        parser.add_argument("--bsz", type=int, default=1)
        parser.add_argument('--eval_by_split', action="store_true", help="report performance by split")
        parser.add_argument('--no_shuffle', action="store_true", help="report performance by split")
        parser.add_argument('--debug', action="store_true", help="Debug mode.")
        self.parser = parser

    def parse(self):
        if not self.initialized:
            self.initialize()

        opt = self.parser.parse_args()
        self.opt = opt

        return opt

def generate_questions(task, prompt, queries, timestamps=None):
    questions, choices = [], []

    for query, timestamp in zip(queries, timestamps):
        choice = random.choice(["pos", "neg"])
        st, ed = timestamp[0], timestamp[1]

        if task == "grounding":
            question = prompt[task].format(event=query)

        elif task == "verif":
            add_detail = prompt.get("add_detail", "")
            question = random.choice(prompt[choice]).format(event=query, st=st, ed=ed)
            question += f" {add_detail}"

        else:
            raise NotImplementedError(f"Not implemented task: {task}")

        questions.append(question)
        choices.append(choice)

    return questions, choices



def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(l.replace("'","").strip("\n")) for l in f.readlines()]



def save_jsonl(data, filename):
    """data is a list"""
    with open(filename, "w") as f:
        f.write("\n".join([json.dumps(e) for e in data]))


def save_json(data, filename, save_pretty=False, sort_keys=False):
    with open(filename, "w") as f:
        if save_pretty:
            f.write(json.dumps(data, indent=4, sort_keys=sort_keys))
        else:
            json.dump(data, f)


def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)


def get_iou(A, B):
    try:
        max0 = max((A[0]), (B[0]))
        min0 = min((A[0]), (B[0]))
        max1 = max((A[1]), (B[1]))
        min1 = min((A[1]), (B[1]))

        return round(max(min1 - max0, 0) / (max1 - min0), 2)

    except:
        return 0


def dict_to_markdown(d, max_str_len=120):
    # convert list into its str representation
    d = {k: v.__repr__() if isinstance(v, list) else v for k, v in d.items()}
    # truncate string that is longer than max_str_len
    if max_str_len is not None:
        d = {k: v[-max_str_len:] if isinstance(v, str) else v for k, v in d.items()}
    return pd.DataFrame(d, index=[0]).transpose().to_markdown()


def display(args):
    opt = args if isinstance(args, dict) else vars(args)
    print(dict_to_markdown(opt, max_str_len=120))