# image demon
python tools/demo.py image -f exps/default/yolox_s.py -c ./pretrained/yolox_s.pth --path assets/dog.jpg --conf 0.25 --nms 0.45 --tsize 640 --save_result --device gpu

# video demon
python tools/demo.py video -f exps/default/yolox_m.py -c ./pretrained/yolox_m.pth --path ./videos/turn1.mp4 --conf 0.25 --nms 0.45 --tsize 640 --save_result --device gpu
python tools/demo_vehicle_track.py video -f exps/example/yolox_m_vehicle_det.py -c ./pretrained/yolox_m.pth --path ./videos/test_13.mp4 --conf 0.25 --nms 0.45 --tsize 640 --save_result --device gpu

# training custom dataset(evaluation failed)
python ./tools/train.py -f exps/example/yolox_voc/yolox_voc_s_vehicle.py -d 2 -b 16 -o -c ./pretrained/yolox_s.pth --cache
python ./tools/train.py -f exps/example/yolox_voc/yolox_nano_vehicle.py -d 2 -b 32 -o -c ./pretrained/yolox_nano.pth --cache


