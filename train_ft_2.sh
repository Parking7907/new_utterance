CUDA_VISIBLE_DEVICES=1 python ./trainSpeakerNet.py --model ResNetSE34F2_ft --log_input True --encoder_type SAP --trainfunc amsoftmax --save_path exps/exp_F2_ft_1 --nClasses 5994 --batch_size 400 --scale 30 --margin 0.3 --nOut 256