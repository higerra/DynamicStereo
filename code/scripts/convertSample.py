import subprocess

startid = 1
endid = 12

source_path = "/home/yanhang/Documents/research/DynamicStereo/data/rfTrain"
target_path = "/home/yanhang/Documents/research/DynamicStereo/data/rfTrain2"

size = (640, 360)

list_path = "{}/list.txt".format(target_path)
list_file = open(list_path, 'w+')

for id in range(startid, endid+1):
    print "rescaling sample{}.mp4 to {} by {}".format(id, size[0], size[1])
    command = "ffmpeg -i {}/sample{}.mp4 -an -vf scale={}:{} -b:v 40000K {}/sample_rescale{}.mp4".format(source_path, id, size[0], size[1], target_path, id)
    print command
    subprocess.call(command, shell=True)

    print "rescaling gt{}.png to {} by {}".format(id, size[0], size[1])
    command = "convert {}/gt{}.png -resize {}x{} {}/gt_rescale{}.png".format(source_path, id, size[0], size[1], target_path, id)
    print command
    subprocess.call(command, shell=True)

    list_file.write("sample_rescale{}.mp4 gt_rescale{}.png\n".format(id, id))

list_file.close()
