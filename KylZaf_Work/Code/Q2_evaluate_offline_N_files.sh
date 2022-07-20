#! /bin/bash
# (bash) .\Q2_evaluate_offline_N_files.sh <audio files for evaluation directory> <file number>


# directory with audio files for evaluation
search_dir=$1
# number of random audio files for evaluation
N=$2

# select the N random files and move them to search_dir
python functions.py $N
python -c 'import functions; functions.move_N_files_forEval('$N')';

files=()
for entry in "$search_dir"/*; do
  files+=( "$entry" )
done

# run the evaluation for all files to create the logfiles and scores_csv for the final step
for f in "${files[@]}"; do
  python Q2_evaluate_offline_EvalNFiles.py $f
done


python -c 'import functions; functions.moveBack_files_forEval()';

python -c 'import functions; functions.classif_report()';
