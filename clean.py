import os


#models = ['yolov11s.hef', 'yolov11n13cw.hef', 'yolov8s.hef', 'yolov11m.hef', 'yolov10x.hef', 'yolov10b.hef', 'yolov8n.hef', 'yolov10n13cw.hef', 'yolov11l.hef', 'yolov10s.hef', 'yolov11x.hef']

models = ['yolov11s', 'yolov11n13cw', 'yolov11m', 'yolov10x', 'yolov10b', 'yolov8n', 'yolov10n13cw', 'yolov11l', 'yolov10s', 'yolov11x']

for model in models:
	path = f"./{model}/"
	imgs = os.listdir(path)
	count = 0

	for i in os.listdir(path):
		if i.split(".")[0][-2:] != "01":
			os.remove(path+i)
			print(i,"Removed!")
		else:
			count += 1
			#print(i)
	print("="*30)
	print("Done! ",model , count)
