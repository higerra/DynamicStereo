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

segmentation_exec = '~/hang/bin/VideoSegmentation'

data_root = '~/hang/data/traindata/samples/'
data_rootw = 'Z:/data/traindata/samples/'

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

    cmd = '{} {}/{}'.format(segmentation_exec, data_root, name)

    TaskAttribute = {'CommandLine': cmd,  'MinCores': '4', 'MaxCores': '8'}
    SubElement(task_node, 'Task', TaskAttribute)

tree = ElementTree(job_node)
tree.write(args.output)


