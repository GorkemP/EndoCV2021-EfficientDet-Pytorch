import os

os.system(
        "/opt/conda/bin/python train.py -c 0 -p birdview_vehicles --lr 5e-3 --batch_size 8 --load_weights weights/efficientdet-d0.pth  --num_epochs 5 --save_interval 200")
