import subprocess

datasets = ['casino2', 'casino3', 'casino4', 'casino5', 'casino6', 'casino7', 'casino8', 'statue1', 'vegas29', 'vegas32', 'vegas33']
testFrame = ['00100', '00200']

video_segment_path = '/home/yanhang/Documents/research/external_code/video_segment/seg_tree_sample/build/seg_tree_sample'

for data in datasets:
    for tf in testFrame:
        command = 'ffmpeg -i ../data/data_{}/midres/prewarp/prewarpb{}_%05d.jpg -vcodec h264 -qp 0 data/data_{}/midres/prewarp/prewarpb{}.mp4'.format(data, tf, data, tf)
        print command
        subprocess.check_call(command, shell=True)

        command = '{} ../data/data_{}/midres/prewarp/prewarpb{}.mp4 --write_to_file --render_and_save'.format(video_segment_path, data, tf)
        print command
        subprocess.check_call(command, shell=True)
