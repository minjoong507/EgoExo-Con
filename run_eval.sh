# Evaluate saved predictions, e.g.:
#   bash run_eval.sh --test_path <output_dir> --eval_by_split

which_python=$(which python)
echo "which python: ${which_python}"
export PYTHONPATH=${PYTHONPATH}:.
echo "PYTHONPATH: ${PYTHONPATH}"

python task/eval.py ${@:1}
