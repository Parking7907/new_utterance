CUDA_VISIBLE_DEVICES=1 python ./trainSpeakerNet.py --eval --model ResNetSE34F2 --log_input True --trainfunc aamsoftmax --margin 0.3 --save_path exps/test --eval_frames 400 --initial_model ./checkpoints/F2_7_290.model --nOut 512