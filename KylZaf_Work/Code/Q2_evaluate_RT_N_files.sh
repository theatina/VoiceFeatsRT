#! /bin/bash
# Syntax: (bash) .\Q2_evaluate_RT_N_files.sh <audio files for evaluation directory> <file number>
# _______________________________________________________________________________________________


# directory with audio files for evaluation
eval_dir=$1
# number of random audio files for evaluation
N=$2

# select the N random files and move them to the evaluation directory
python functions.py $N
python -c 'import functions; functions.move_N_files_forEval('$N')';

# store audio file paths from the evaluation directory to run the evaluation .py file N times ( 1 for each )
files=()
for entry in "$eval_dir"/*; do
  files+=( "$entry" )
done

# progress bar
model_eval=$'\nModel evaluation in progress '
function printProgressBar() {
    local progressBar="."
    printf "%s" "${progressBar}"
}
printf "%s" "${model_eval}"

# run the evaluation for all files to create the logfiles and scores_csv for the final step
for f in "${files[@]}"; do
  python Q2_evaluate_RT_EvalNFiles.py $f
  ((cnt++))
  printProgressBar 
done

# move the files from the evaluation directory to the initial audio files directory 
python -c 'import functions; functions.moveBack_files_forEval()'
# create the final classification report from .csv file with the audio file windows' emotion prediction
python -c 'import functions; functions.classif_report()'

EvalStatus="Completed"
printf " [%s]\n\n" "${EvalStatus}"