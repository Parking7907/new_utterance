CUDA_VISIBLE_DEVICES=1 python ./trainSpeakerNet.py --model ResNetSE34F2 --log_input True --encoder_type SAP --trainfunc aamsoftmax --save_path exps/exp_F2_2 --nClasses 5994 --batch_size 800 --scale 30 --margin 0.3 --nOut 256  --scheduler cosineannealing --optimizer sgd