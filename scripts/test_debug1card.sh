cd /inspire/hdd/project/wuliqifa/chenxinyan-240108120066/songbur/newpas/OpenDWM

source /inspire/hdd/global_user/chenxinyan-240108120066/miniconda3/bin/activate ggearth

export CUDA_VISIBLE_DEVICES=0

export PYTHONPATH=/inspire/hdd/project/wuliqifa/chenxinyan-240108120066/songbur/newpas/OpenDWM/src:\
/inspire/hdd/project/wuliqifa/chenxinyan-240108120066/songbur/newpas/OpenDWM/externals/waymo-open-dataset/src:\
/inspire/hdd/project/wuliqifa/chenxinyan-240108120066/songbur/newpas/OpenDWM/externals/TATS/tats/fvd \

torchrun --nproc_per_node=1 src/dwm/preview.py \
  -c /inspire/hdd/project/wuliqifa/chenxinyan-240108120066/songbur/newpas/OpenDWM/configs/debug/rolling_ref+cleanN.json \
  -o output/rolling_ref+cleanN4+2fps2



