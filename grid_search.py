import os
import re
import shutil
import time

best_result = 0
best_line = ''
best_parameters = {}
path_1 = './output/predictions_.json'
path_2 = './output/pytorch_model.bin'
save_path = './best_model'


# for lr in [4e-5]:
for lr in [3e-5, 4e-5, 5e-5]:
    for bs in [8, 9, 10]:
        line = os.popen('python main.py --train_file dataset/train.json --predict_file dataset/valid.json --model_type '
                    'spanbert --model_name_or_path bert-base-cased --output_dir output/ --version_2_with_negative '
                    '--do_train --do_eval --overwrite_output --save_steps 0 --learning_rate %f '
                    '--per_gpu_train_batch_size %d' % (lr, bs)).read()
#    time.sleep(150)
    print("line-----------------: ", line)
    exact = re.findall(r"(?<='exact': )\d+\.?\d*", line)[0]
    exact = float(exact)
    print("exact: ", exact)
    f1 = re.findall(r"(?<='f1': )\d+\.?\d*", line)[0]
    f1 = float(f1)
    print("f1: ", f1)
    result = 0.5*exact+0.5*f1
    print("result: ", result)
    if result > best_result:  # Find the best performing parameter
        best_result = result
        shutil.copy(path_1, save_path)  # Save best predictions_.json
        shutil.copy(path_2, save_path)  # Save best pytorch_model.bin
        best_parameters = {'lr': lr, 'bs': bs}

print("Best result:{:.2f}".format(best_result))
print("Best parameters:{}".format(best_parameters))
