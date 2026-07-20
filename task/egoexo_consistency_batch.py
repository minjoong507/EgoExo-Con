import os
import time
import math
import random
from tqdm import tqdm
from easydict import EasyDict as edict
from utils.cons_utils import (save_jsonl, save_json, load_json, get_iou,
                              load_logger, display, load_jsonl)
from task.eval import eval_egoexo_verif_consistency, eval_egoexo_grounding_consistency

eval_func = {
    'verif': eval_egoexo_verif_consistency,
    'grounding': eval_egoexo_grounding_consistency,
}

logger = load_logger("[EgoExo-Consistency Evaluation]")
OPT_FILE_NAME = "opt.json"
PREDICTION_FILE_NAME = "predictions.jsonl"
EVALUATION_FILE_NAME = "eval_results.json"


def _run_verification(model, exo_video_paths, ego_video_paths, temporal_queries, misaligned_queries, timestamps, lengths=None):
    responses_for_exo_video = model.run(
        task="verif",
        video_paths=exo_video_paths,
        queries=temporal_queries,
        timestamps=timestamps,
        lengths=lengths,
    )

    responses_for_ego_video = model.run(
        task="verif",
        video_paths=ego_video_paths,
        queries=temporal_queries,
        timestamps=timestamps,
        lengths=lengths,
    )

    responses_mis_for_exo_video = model.run(
        task="verif",
        video_paths=exo_video_paths,
        queries=misaligned_queries,
        timestamps=timestamps,
        lengths=lengths,
    )

    responses_mis_for_ego_video = model.run(
        task="verif",
        video_paths=ego_video_paths,
        queries=misaligned_queries,
        timestamps=timestamps,
        lengths=lengths,
    )

    return [
        edict(
        a = edict(exo=ans_for_exo_video,ego=ans_for_ego_video,),
        m = edict(exo=mis_ans_for_exo_video,ego=mis_ans_for_ego_video)
        )
        for ans_for_exo_video, ans_for_ego_video, mis_ans_for_exo_video, mis_ans_for_ego_video in zip(responses_for_exo_video, responses_for_ego_video, responses_mis_for_exo_video, responses_mis_for_ego_video)
    ]

def _run_grounding(model, exo_video_paths, ego_video_paths, temporal_queries, timestamps, lengths=None):
    exo_pred_moments = model.run(
        task="grounding",
        video_paths=exo_video_paths,
        queries=temporal_queries,
        timestamps=timestamps,
        lengths=lengths,
    )

    ego_pred_moments = model.run(
        task="grounding",
        video_paths=ego_video_paths,
        queries=temporal_queries,
        timestamps=timestamps,
        lengths=lengths,
    )

    try:
        exo_ious = [get_iou(gt_moment, exo_pred_moment['t']) for gt_moment, exo_pred_moment in zip(timestamps, exo_pred_moments)]
    except:
        exo_ious = [0 for i in range(len(timestamps))]
    try:
        ego_ious = [get_iou(gt_moment, ego_pred_moment['t']) for gt_moment, ego_pred_moment in zip(timestamps, ego_pred_moments)]
    except:
        ego_ious = [0 for i in range(len(timestamps))]

    # Save the results.
    return [
        edict(
            ego=ego_pred_moment,
            exo=exo_pred_moment,
            ego_iou=ego_iou,
            exo_iou=exo_iou,
        ) for ego_pred_moment, exo_pred_moment, ego_iou, exo_iou in zip(ego_pred_moments, exo_pred_moments, ego_ious, exo_ious)
    ]



def main(args, model):
    path_to_predictions = f"{args.task}_{PREDICTION_FILE_NAME}"
    path_to_eval_results = f"{args.task}_{EVALUATION_FILE_NAME}"

    test_data = load_json(args.test_path)
    logger.info(f"Loaded {len(test_data)} samples from {args.test_path}")
    logger.info(f"Total unique video: {len(set([d['exo_video_path'] for d in test_data]))} | query: {len(test_data)}")

    if not args.no_shuffle:
        random.shuffle(test_data)

    results = []

    if args.resume:
        results = load_jsonl(os.path.join(args.output_dir, f"{args.task}_predictions.jsonl"))
        _meta_list1 = [result['meta']['vid'] for result in results]
        _meta_list2 = [result['meta']['query'] for result in results]
        _test_data = [data for data in test_data if data['video_id'] not in _meta_list1 and data['temporal_query'] not in _meta_list2]
        print(f"Loaded {len(results)} predictions and test on {len(test_data)} => {len(_test_data)} samples")
        test_data = _test_data

    epochs = math.ceil(len(test_data) / args.bsz)

    for epoch in tqdm(range(epochs), total=epochs, desc=f"Evaluating Ego- & Exo- Consistency"):
        sid = epoch * args.bsz
        eid = min((epoch + 1) * args.bsz, len(test_data))

        batch = test_data[sid:eid]

        ego_centric_videos = [data['ego_video_path'] for data in batch]
        exo_centric_videos = [data['exo_video_path'] for data in batch]
        sources = [data['source'] for data in batch]
        temporal_queries = [data['temporal_query'] for data in batch]
        misaligned_queries = [data['misaligned_query'] for data in batch]
        lengths = [data['length'] for data in batch]
        timestamps = [data['timestamp'] for data in batch]

        exo_video_paths = [os.path.join(args.video_root, source,f"{vid}.mp4") for vid, source in zip(exo_centric_videos, sources)]
        ego_video_paths = [os.path.join(args.video_root, source, f"{vid}EGO.mp4") for vid, source in zip(ego_centric_videos, sources)]

        meta = [
            edict(
                vid=vid,
                query=temporal_query,
                misaligned_query=misaligned_query,
                timestamp=timestamp,
                length=length,
                source=source,
            ) for vid, temporal_query, misaligned_query, timestamp, length, source in zip(exo_centric_videos, temporal_queries, misaligned_queries, timestamps, lengths, sources)
        ]


        if "grounding" in args.task:
            batch_grounding_result = _run_grounding(
                model=model,
                exo_video_paths=exo_video_paths,
                ego_video_paths=ego_video_paths,
                temporal_queries=temporal_queries,
                timestamps=timestamps,
                lengths=lengths,
            )
            results.extend(
                [edict(meta=_meta, grounding=grounding_result) for _meta, grounding_result in zip(meta, batch_grounding_result)]
            )

        elif "verif" in args.task:
            batch_verif_result = _run_verification(
                model=model,
                exo_video_paths=exo_video_paths,
                ego_video_paths=ego_video_paths,
                temporal_queries=temporal_queries,
                misaligned_queries=misaligned_queries,
                timestamps=timestamps,
                lengths=lengths,
            )
            results.extend(
                [edict(meta=_meta, verif=verif_result) for _meta, verif_result in zip(meta, batch_verif_result)]
            )

        else:
            raise NotImplementedError(f"{args.task} is not implemented.")

        if args.debug and epoch == 1:
            break

        if epoch % 10 == 0:
            logger.info(f"{len(results)} predictions are saved")
            save_jsonl(results, os.path.join(args.output_dir, path_to_predictions))

    logger.info(f"{len(results)} predictions for {args.task} will be saved at {path_to_predictions}")
    logger.info("============ Save Performance ============")
    save_jsonl(results, os.path.join(args.output_dir, path_to_predictions))

    eval_results = eval_func[args.task](results, args=args, verbos=True)
    save_json(eval_results, os.path.join(args.output_dir, path_to_eval_results), save_pretty=True)
    logger.info("============ Done. ============")


def run_egoexo_consistency(model, args):
    if args.resume and args.output_dir:
        if not os.path.exists(args.output_dir):
            raise FileNotFoundError(f"Failed to find Output directory {args.output_dir}")
        logger.info(f"Overwriting output dir from {args.output_dir}")

    else:
        cur_time = time.strftime("%Y_%m_%d_%H_%M_%S")
        folder_name = f"{args.exp_id}_{cur_time}" if args.exp_id else cur_time
        output_dir = os.path.join(f"{args.task}_results", args.model_type, folder_name)

        if args.debug:
            output_dir = f"debug_results/{args.model_type}_{cur_time}"

        os.makedirs(output_dir, exist_ok=True)
        args.output_dir = output_dir

    # display and save results
    display(args)
    save_json(vars(args), os.path.join(args.output_dir, OPT_FILE_NAME), save_pretty=True)

    main(args, model)
