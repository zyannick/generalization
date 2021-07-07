python3 main.py --inference --model FlowNet2 --save_flow --inference_dataset MpiSintelClean --inference_dataset_root ./MpiSintelClean/ --resume ./FlowNet2_checkpoint.pth.tar


python3 infer.py --inference --model FlowNet2 --save_flow --inference_dataset ImagesFromFolder --inference_dataset_root ./files/ --resume ./FlowNet2_checkpoint.pth.tar