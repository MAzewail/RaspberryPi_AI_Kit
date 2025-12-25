import os


#models = ['yolov11s.hef', 'yolov11n13cw.hef', 'yolov8s.hef', 'yolov11m.hef', 'yolov10x.hef', 'yolov10b.hef', 'yolov8n.hef', 'yolov10n13cw.hef', 'yolov11l.hef', 'yolov10s.hef', 'yolov11x.hef']

models = ['yolov11s', 'yolov11n13cw', 'yolov11m', 'yolov10x', 'yolov10b', 'yolov8n', 'yolov10n13cw', 'yolov11l', 'yolov10s', 'yolov11x']

for model in models:
	path = f"./{model}/"
	imgs = os.listdir(path)
	count = 0
	inc = 0
	im_names = []

	for i in os.listdir(path):
		im_names.append(int(i.split("_")[1]))
	im_names = set(im_names)
	for n,name in enumerate(im_names):
		if name != n+inc:
			#os.remove(path+i)
			print(name)
			inc += 1
#	else:
#		count += 1
#		#print(i)
	print("="*30)
	print("Done! ",model , count)
