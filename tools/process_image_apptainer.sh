#!/bin/bash

echo "start processing: $3"
start_time=$(date +%s)

container=$1
model_dir=$2
image=$3
out_dir=$4

iname=`basename $image`
idir=`dirname $image`
ibase=`basename $image .nii.gz`

out_liver="/opt/output/${ibase}_liver.nii.gz"
out_spleen="/opt/output/${ibase}_spleen.nii.gz"
out_slab="/opt/output/${ibase}_slab.nii.gz"
out_fats="/opt/output/${ibase}_fats.nii.gz"
out_muscles="/opt/output/${ibase}_muscles.nii.gz"
out_imf="/opt/output/${ibase}_imf.nii.gz"

opts="--nv --bind ${idir}:/opt/input:ro --bind ${out_dir}:/opt/output:rw --bind ${model_dir}:/opt/models:ro"
apt="apptainer run ${opts} ${container}"

liver="${apt} python /opt/jabba_ai_core/src/jabba_ai_core/ab_liver.py -v -i /opt/input/${iname} -o ${out_liver} -m /opt/models/liver_deeplab_model"
echo $liver
$liver

spleen="${apt} python /opt/jabba_ai_core/src/jabba_ai_core/ab_spleen.py -v -i /opt/input/${iname} -o ${out_spleen} -m /opt/models/spleen_deeplab_model"
echo $spleen
$spleen

abd="${apt} python /opt/jabba_ai_core/src/jabba_ai_core/ab_slab.py -v -i /opt/input/${iname} -o ${out_slab} -m /opt/models/slice_locator_run_3_model.30-0.05-0.979.h5"
echo $abd
$abd

ret=`$abd`
echo $ret
zmin="$(echo "$ret" | grep slab_start_index | cut -d ':' -f2)"
zmax="$(echo "$ret" | grep slab_end_index | cut -d ':' -f2)"
echo "Abodmen: $zmin -> $zmax"

fats="${apt} python /opt/jabba_ai_core/src/jabba_ai_core/ab_fats.py -i /opt/input/${iname} -o ${out_fats} -z $zmin $zmax -m /opt/models/visceral_run_7_model.326-0.03-0.98.h5"
echo $fats
$fats

muscles="${apt} python /opt/jabba_ai_core/src/jabba_ai_core/ab_muscles.py -i /opt/input/${iname} -o ${out_muscles} ${out_imf} -z $zmin $zmax -m /opt/models/model_scripted_unet_depth4_res4.pt"
echo $muscles
$muscles

end_time=$(date +%s)
run_seconds=$((end_time - start_time))
runtime_hours=$(echo "scale=2; $run_seconds / 3600" | bc)

readonly SECONDS_PER_HOUR=3600
readonly SECONDS_PER_MINUTE=60

hours=$((${run_seconds} / ${SECONDS_PER_HOUR}))
seconds=$((${run_seconds} % ${SECONDS_PER_HOUR}))
minutes=$((${run_seconds} / ${SECONDS_PER_MINUTE}))
seconds=$((${seconds} % ${SECONDS_PER_MINUTE}))

run_time=$(printf "%02d:%02d:%02d" ${hours} ${minutes} ${seconds})
echo "Run time: $run_time"

echo "done"

