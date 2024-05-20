conda create python=3.9 -n dlp-lab5 -y
conda activate dlp-lab5

pip install numpy tqdm pyyaml pytz tensorboard pandas scipy
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118



# tensorboard
tensorboard --logdir="/home_nfs/ericyangchen/DLP/lab5/src/outputs" --port 9090



# training & inpainting
python training_transformer.py 


output_path="outputs/cosine-sweet-6-total-12"
load_transformer_ckpt_path="outputs/best/ckpt/epoch_50.pt"
python inpainting.py \
    --load-transformer-ckpt-path $load_transformer_ckpt_path \
    --output-path $output_path \
    --mask-func cosine \
    --sweet-spot 6 \
    --total-iter 12


# Calculate FID
experiment_name="cosine-sweet-6-total-12"

cd faster-pytorch-fid
inpainting_result_path="../outputs/$experiment_name/test_results"
python fid_score_gpu.py --device cuda --predicted-path $inpainting_result_path
cd ..
