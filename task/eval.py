from utils.cons_utils import *
import os
import ast
import glob
from easydict import EasyDict as edict
from tqdm import tqdm
from openai import OpenAI

# The API key is read from the OPENAI_API_KEY environment variable.
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

logger = load_logger("Evaluation")
GPT_MODEL = "gpt-4o-mini"
OPT_FILE_NAME = "opt.json"
PREDICTION_FILE_NAME = "predictions.jsonl"
EVALUATION_FILE_NAME = "eval_results.json"


def extract_json_dict(text: str):
    """
    Extracts a valid JSON dictionary from a string by detecting the first '{' and last '}'.
    """
    start = text.find('{')
    end = text.rfind('}') + 1

    if start == -1 or end == -1:
        raise ValueError("No valid JSON boundaries found.")

    json_str = text[start:end]

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON decoding failed: {e}")


def _validate_answer(question, answer, expected):
    """
        The implementation to determine whether the answer is correct.
        For efficiency, we first check if the answer contains 'yes' or 'no' keywords.
        If not, we employ GPT to assess the answer.
    """

    # If the answer only contains a single word, either "yes" or "no",
    if len(answer.split()) == 1 and answer.split()[0].lower() in ["yes", "no", "yes.", "no.", "yes,", "no,"]:
        answer = answer.split()[0].lower()
        if expected == "pos":
            if "yes" in answer:
                return 1
            else:
                return 0

        elif expected == "neg":
            if "no" in answer:
                return 1
            else:
                return 0

    else:
        completion = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content":
                        """
                        You are an intelligent evaluator tasked with determining the correctness of an answer in response to a given question. 
                        Based on the expected response type (positive or negative), you need to decide whether the provided answer aligns correctly with both the content and timestamps in the question.
                        
                        INSTRUCTIONS:
                        1. Input Information:
                           - Question: A question about a specific moment or event in the video.
                           - Expected Answer:
                             - "pos" (Positive): The expected answer should agree with the content of the question. It may include phrases like "Yes," or similar content that confirms the question's scenario.
                             - "neg" (Negative): The expected answer should disagree with the content of the question. It may include phrases like "No," or similar content that refutes the question's scenario.
                           - Provided Answer: The actual answer given to the question. You must evaluate whether it is correct based on the question and the expected answer.
                        
                        2. Evaluation Process:
                           - If the expected answer type is "pos," the provided answer should be a positive affirmation of the question.
                           - If the expected answer type is "neg," the provided answer should refute or disagree with the question.
                           
                        3. Output format:
                        The output should be formatted as a dictionary format with the key "result", and the value being either 1 (correct) or 0 (incorrect).
                        
                        EXAMPLES:
                        Input:
                        - Question: Does the event 'A young woman is standing and speaking with her friends' happen in the video?
                        - Expected Answer: neg
                        - Provided Answer: No, we can not see a young woman standing in the video.
                        
                        Output:
                        {
                          "result": 1
                        }
                        
                        Input:
                        - Question: Is the event 'Is the man reading a book under the tree' included in the video?
                        - Expected Answer: pos
                        - Provided Answer: Yes the event is included in the video.
                        
                        Output:
                        {
                          "result": 1
                        }
                        
                        Input:
                        - Question: Does the event 'A man play soccer' happens  in the video?
                        - Expected Answer: pos
                        - Provided Answer: The man not is playing soccer in the video.
                        
                        Output:
                        {
                          "result": 0
                        }
                    
                        
                        Input:
                        - Question: Does the event 'Person opens the window a few more times.' not happen in the video? Please answer with 'Yes' or 'No'.
                        - Expected Answer: pos
                        - Provided Answer: The event 'Person opens the window a few more times' does not happen .
                        
                        Output:
                        {
                          "result": 1
                        }
                        
                        Now it's your turn.
                        """
                },
                {
                    "role": "user",
                    "content":
                        f"""
                        Input:
                        - Question: {question}
                        - Expected Answer: {expected}
                        - Provided Answer: {answer}
                        
                        PLEASE DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. ONLY PROVIDE A DICTIONARY FORMAT.
                        Output:
                        """
                }
            ]
        )
        response_message = completion.choices[0].message.content
        try:
            response = ast.literal_eval(response_message)['result']
            return response

        except Exception as e:
            logger.warning(f"Error occurs: {e} with message: {response_message}, and reprocess it")
            return extract_json_dict(response_message)


def validate_answer_for_egoexo(qa_pair, query_type, debug=False):
    """
    Judge a single ego/exo answer.

    If the answer is not a single 'yes' or 'no' word, GPT-4o-mini is used to evaluate it.
    A response of 1 means the answer is correct; 0 means it is incorrect.
    """

    prompt_choice, question, answer = qa_pair["c"], qa_pair["q"], qa_pair["a"]
    assert prompt_choice in ["pos", "neg"], f"We do not support the prompt template for the {prompt_choice}"

    if query_type == "pos":
        expected_answer = "pos" if prompt_choice == "pos" else "neg"

    else:
        expected_answer = "neg" if prompt_choice == "pos" else "pos"
    response = _validate_answer(question=question, answer=answer, expected=expected_answer)

    qa_pair["j"] = "T" if response == 1 else "F"
    case = "Correct" if response == 1 else "Incorrect"

    if debug:
        print(f"Question: {question}, Answer: {answer}, Expected: {expected_answer} | {case}")

    return response, qa_pair

# The codes for calculating scores



def calculate_r1(iou_values, thresholds=[0.3, 0.5, 0.7], n_preds=None):
    if n_preds is None:
        n_preds = len(iou_values)
    r1_scores = {}
    for threshold in thresholds:
        r1 = sum(1 for iou in iou_values if iou >= threshold) / n_preds if n_preds > 0 else 0
        r1_scores[f'IoU={threshold}'] = float("{:.2f}".format(r1 * 100))
    miou = round(sum(iou_values) / n_preds * 100, 3) if n_preds > 0 else 0
    r1_scores['mIoU'] = miou

    return r1_scores


def compute_verif_scores(performance, length):
    performance["ego"] = round(performance["ego"] / (length * 2) * 100, 2)
    performance["exo"] = round(performance["exo"] / (length * 2) * 100, 2)
    performance["both_correct"] = round(performance["both_correct"] / (length * 2) * 100, 2)
    performance["copy"] = "& ".join([f"{v}" for k, v in performance.items() if k in ["ego", "exo", "both_correct"]])
    return performance

def _eval_verif_consistency(results, args):
    performance = {
        "exo": 0,
        "ego": 0,
        "both_correct": 0,
    }

    for result in tqdm(results, desc=f"Evaluating {args.task} predictions..", total=len(results)):
        data = result['verif']

        for q_type in ["a", "m"]:  # a: aligned / m: misaligned
            query_type = "pos" if q_type == "a" else "neg"

            qa_pair_for_ego_video = data[q_type]["ego"]
            qa_pair_for_exo_video = data[q_type]["exo"]

            if not isinstance(qa_pair_for_ego_video, dict) or not isinstance(qa_pair_for_exo_video, dict):
                logger.info(f"Invalid output: {data}")
                continue

            result_for_ego_video, qa_pair_for_ego_video = validate_answer_for_egoexo(qa_pair_for_ego_video, query_type, debug=args.debug)
            result_for_exo_video, qa_pair_for_exo_video = validate_answer_for_egoexo(qa_pair_for_exo_video, query_type, debug=args.debug)

            data[q_type]["ego"]["j"] = qa_pair_for_ego_video['j']
            data[q_type]["exo"]["j"] = qa_pair_for_exo_video['j']

            performance["ego"] += result_for_ego_video
            performance["exo"] += result_for_exo_video

            if result_for_ego_video + result_for_exo_video == 2:
                performance["both_correct"] += 1

    return performance

def eval_egoexo_verif_consistency(results, args, verbos=False):
    if isinstance(results, list):
        pass
    elif isinstance(results, str):
        results = load_jsonl(results)
    else:
        raise TypeError("Input file must be a list or a string")

    if args.eval_by_split:
        performance_by_split = {}
        len_by_split = {}
        performance_all = {
        "exo": 0,
        "ego": 0,
        "both_correct": 0,
        }

        for split_name in args.split_list:
            result_by_split = [_result for _result in results if _result['meta']['source'] == split_name]
            if len(result_by_split) == 0: continue

            len_by_split[split_name] = len(result_by_split)
            performance_by_split[split_name] = _eval_verif_consistency(result_by_split, args)

            for key in ["exo", "ego", "both_correct"]:
                performance_all[key] += performance_by_split[split_name][key]

        performance_by_split['all'] = performance_all

        for split_name in performance_by_split.keys():
            length = len_by_split[split_name] if split_name != "all" else len(results)
            performance_by_split[split_name] = compute_verif_scores(performance_by_split[split_name], length)

        if verbos:
            print("Performance")
            print(performance_by_split)
            print('------------------------')

        display(performance_by_split['all'])
        try:
            save_jsonl(results, os.path.join(args.test_path, f"{args.task}_judgements.jsonl"))
        except AttributeError:
            save_jsonl(results, os.path.join(args.output_dir, f"{args.task}_judgements.jsonl"))

        return performance_by_split


    else:
        performance = _eval_verif_consistency(results, args)
        performance = compute_verif_scores(performance, len(results))

        if verbos:
            print("Performance")
            print(performance)

        display(performance)

        try:
            save_jsonl(results, os.path.join(args.test_path, f"{args.task}_judgements.jsonl"))
        except AttributeError:
            save_jsonl(results, os.path.join(args.output_dir, f"{args.task}_judgements.jsonl"))

        return performance


def compute_r1_iou_scores(performance, iou_thds=[0.3, 0.5, 0.7], con_iou_thd=0.5):
    N = len(performance["ego"])
    r1_iou_scores = {}

    r1_iou_scores["exo"] = calculate_r1(performance["exo"])
    r1_iou_scores["ego"] = calculate_r1(performance["ego"])

    ego_exo = []
    for iou_thd in iou_thds:
        ego_exo.append([1 if performance["exo"][i] >= iou_thd and performance["ego"][i] >= iou_thd
                        else 0 for i in range(N)])
    r1_iou_scores["consistent"] = {k: (sum(v) / N * 100) for k, v in zip([f"IoU={iou_thd}" for iou_thd in iou_thds], ego_exo)}

    if con_iou_thd:
        r1_iou_scores['final'] = {
            "exo": r1_iou_scores["exo"][f"IoU={con_iou_thd}"],
            "ego": r1_iou_scores["ego"][f"IoU={con_iou_thd}"],
            "consistent": r1_iou_scores["consistent"][f"IoU={con_iou_thd}"],
        }

    return r1_iou_scores


def eval_egoexo_grounding_consistency(results, args, verbos=False, iou_thd=0.5):
    if isinstance(results, list):
        pass
    elif isinstance(results, str):
        results = load_jsonl(results)
    else:
        raise TypeError("Input file must be a list or a string")

    print(f"Total {len(results)} predictions.")

    if args.eval_by_split:
        performance_by_split = {}
        r1_score_by_split = {}
        len_by_split = {}

        performance_all = {
            "exo": [],
            "ego": [],
            "both_correct": [],
        }

        for split_name in args.split_list:
            result_by_split = [_result for _result in results if _result['meta']['source'] == split_name]
            if len(result_by_split) == 0: continue

            len_by_split[split_name] = len(result_by_split)
            performance_by_split[split_name] = {
                "exo": [result['grounding']['exo_iou'] for result in result_by_split],
                "ego": [result['grounding']['ego_iou'] for result in result_by_split],
            }

            for key in ["exo", "ego"]:
                performance_all[key].extend(performance_by_split[split_name][key])

        performance_by_split['all'] = performance_all

        for split_name in performance_by_split.keys():
            r1_score_by_split[split_name] = compute_r1_iou_scores(performance_by_split[split_name], con_iou_thd=iou_thd)

        if verbos:
            print("Scores")
            print(r1_score_by_split)
            print(r1_score_by_split['all']['final'])

        return r1_score_by_split

    else:
        performance = {
            "exo": [result['grounding']['exo_iou'] for result in results],
            "ego": [result['grounding']['ego_iou'] for result in results],
        }
        r1_iou_scores = compute_r1_iou_scores(performance, con_iou_thd=iou_thd)

        if verbos:
            print("Scores")
            print(r1_iou_scores)

        return r1_iou_scores



if __name__ == "__main__":
    import argparse
    args = argparse.ArgumentParser(description="Evaluate the Video-LLM's prediction results")
    args.add_argument("--iou_thd", type=float, default=0.5,
                      help="Considering the model's prediction which has a higher IoU than the threshold.")
    args.add_argument("--test_path", type=str, required=True,
                      help="path to the model's predictions")
    args.add_argument("--eval_by_split", action="store_true")
    args.add_argument('--debug', action="store_true",
                      help="run in the debug mode.")
    args = args.parse_args()

    opt = None

    opt_path = os.path.join(args.test_path, OPT_FILE_NAME)
    if os.path.exists(opt_path):
        opt = edict(load_json(opt_path))
    else:
        raise FileNotFoundError(opt_path)

    opt.test_path = args.test_path
    path_to_predictions = f"{opt.task}_{PREDICTION_FILE_NAME}"
    file_path = os.path.join(args.test_path, path_to_predictions)
    opt_path = os.path.join(args.test_path, OPT_FILE_NAME)

    if args.eval_by_split:
        opt.eval_by_split = True

    if not os.path.exists(file_path):
        _file_path = glob.glob(os.path.join(args.test_path, f"*{PREDICTION_FILE_NAME}"))[0]
        if os.path.exists(_file_path):
            logger.info(f"Replace the file path {file_path} to {_file_path}")
            file_path = _file_path
        else:
            raise FileNotFoundError(f"File {file_path} does not exist!")

    logger.info(f"Loading predictions from {file_path}")

    if opt:
        logger.info(f"Configurations of {file_path}")
        display(opt)

    EVAL_FUNC = {
        'verif': eval_egoexo_verif_consistency,
        'grounding': eval_egoexo_grounding_consistency,
    }

    result = EVAL_FUNC[opt.task](file_path, args=opt)
    logger.info("Done.")

    for k, v in result.items():
        if k in ["short", "medium", "long"]:
            print(f"{k} : {v['copy']}")

        else:
            print(f"{k}: {v}")

    if not args.debug:
        if PREDICTION_FILE_NAME in args.test_path:
            args.test_path = args.test_path.replace(PREDICTION_FILE_NAME, "")
        save_json(result, os.path.join(args.test_path, f"{opt.task}_{EVALUATION_FILE_NAME}"), save_pretty=True)