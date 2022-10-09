rem 1 激活虚拟环境
call D:\applications\Anaconda3\envs\torch_gpu\Scripts\activate

rem 2 跳转到该项目的根目录下

cd /d C:\Users\Super-IdoI\Desktop\dataset-ecir\Fintech-Key-Phrase

rem 3 执行python文件
python flask_released_api.py

rem 4 执行后暂停 # 不太懂
pause