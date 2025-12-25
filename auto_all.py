import subprocess
import os


# python run.py --hef doc/yolov11n13cw.hef --input 1img_video.mp4 --labels-json labels13cw.json 

# ['yolov11s.hef', 'yolov11n13cw.hef', 'yolov8s.hef', 'yolov11m.hef', 'yolov10x.hef', 'yolov10b.hef', 'yolov8n.hef', 'yolov10n13cw.hef', 'yolov11l.hef', 'yolov10s.hef', 'yolov11x.hef']



vd_path = "./imgs/v/"
inputs = os.listdir(vd_path)
max_seconds = 1
models = ['yolov11s.hef', 'yolov11n13cw.hef', 'yolov11m.hef', 'yolov10x.hef', 'yolov10b.hef', 'yolov8n.hef', 'yolov10n13cw.hef', 'yolov11l.hef', 'yolov10s.hef', 'yolov11x.hef']

for i in inputs:
	in_path = vd_path+i
	try:
		subprocess.run(["python", "run.py", "--hef", f"doc/{models[10]}", "--input", in_path, "--labels-json", "labels13cw.json"], timeout = max_seconds, check=True)
	except subprocess.TimeoutExpired:
		print(f">> TIMEOUT: The script took longer than {max_seconds}s and was killed.")
	except subprocess.CalledProcessError:
		print(">> ERROR: The script crashed (syntax error or bug).")
	print("="*30)
