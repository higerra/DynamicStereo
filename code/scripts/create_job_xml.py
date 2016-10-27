import argparse
from xml.etree.ElementTree import Element, SubElement, ElementTree
import json
import os.path

parser = argparse.ArgumentParser()
parser.add_argument('list')
parser.add_argument('--data_root', default='~/hang/data/working/')
parser.add_argument('--binPath', default='~/hang/bin/')
parser.add_argument('--priority', default='Highest')
parser.add_argument('--output', default='z:/jobs/job.xml')

args = parser.parse_args()

stereo_exec = '~/hang/bin/DynamicStereo'
render_exec = '~/hang/bin/SegmentAndRender'
data_root = '~/hang/data/working/'
data_rootw = 'Z:/data/working/'

listfile = open(args.list)
datasets = listfile.readlines()

JobAttribute = {'Version': '3.000', 'IsExclusive': 'false',
                'Priority': args.priority, 'xmlns': 'http://schemas.microsoft.com/HPCS2008R2/scheduler/'}
job_node = Element('Job', JobAttribute)

SubElement(job_node, 'Dependencies')
task_node = SubElement(job_node, 'Tasks')

for data in datasets:
    name = data.split()[0]
    data_path = data_root + name + '/'
    data_pathw = data_rootw + '/'
    if os.path.isfile(data_pathw + '/conf.json'):
        with open(data_pathw+'conf.json') as f:
            conf = json.load(f)
    else:
        with open(data_rootw + '/default.json') as f:
            conf = json.load(f)

    global_stereo = ''
    global_render = '--classifierPath=/root/hang/data/visualword/model_new.rf --codebookPath=/root/hang/data/visualword/metainfo_new_cluster00050.yml'

    if 'global_stereo' in conf:
        global_stereo += ' ' + conf['global_stereo']
    if 'global_render' in conf:
        global_render += ' ' + conf['global_render']
    
    for frame in conf['frames']:
        tf = frame['frameid']
        cmd = 'export LD_LIBRARY_PATH=~/hang/lib;'
        cmd += '{} {} --testFrame={}'.format(stereo_exec, data_path, tf)
        
        if 'weight_smooth' in frame:
            cmd += ' weight_smoot={}'.format(frame['weight_smooth'])
        
        cmd += ' ' + global_stereo + ';'

        cmd += '{} {} --testFrame={}'.format(render_exec, data_path, tf)
        cmd += ' ' + global_render

        TaskAttribute = {'CommandLine': cmd,  'MinCores': '4', 'MaxCores': '8'}
        SubElement(task_node, 'Task', TaskAttribute)

tree = ElementTree(job_node)
tree.write(args.output)


