
python main.py --old_loss True --use_teacher False --model_num 1 --batch_size 128 --model wideresnet --epochs 200
cp ./ckpt/model1_model_best.pth.tar ./ckpt/wideresnet
python main.py --old_loss True --use_teacher False --model_num 1 --batch_size 128 --model densenet --epochs 200
cp ./ckpt/model1_model_best.pth.tar ./ckpt/densenet
python main.py --old_loss True --use_teacher False --model_num 1 --batch_size 128 --model googlenet --epochs 200
cp ./ckpt/model1_model_best.pth.tar ./ckpt/googlenet

