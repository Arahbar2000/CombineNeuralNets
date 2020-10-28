import os
import argparse
import numpy as np
import pandas as pd
import skimage.io
import skimage.segmentation
import skimage.measure
import skimage.morphology
import cv2
import compute_metrics as comp
import matplotlib.pyplot as plt
import shutil
import seaborn as sns
from random_walker import random_walker
from ensemble import Ensemble
from optimize import Optimize
import statistics


class Config:
    def __init__(self, output, root, annot, file, counter=None, erosions=5, beta=30, instance=False):
        self.root = root
        self.annot = annot
        self.segmentations = [x for x in os.listdir(root) if not x.startswith('.')]
        # f = open('/mnt/resources/work/CombineNeuralNets/' + file)
        f = open('/Users/arianrahbar/ResearchEllison/CombineNeuralNets/' + file)
        self.files = f.readlines()
        self.instance = instance
        self.filename = file.split('.')[0]
        self.erosions = erosions
        self.beta = beta
        self.counter = counter
        self.output = output
        f.close()


def parse_arguments():
    description = 'Gets the root directory'
    parser = argparse.ArgumentParser(description=description)
    required = parser.add_argument_group("required arguments")

    required.add_argument('-r', '--root', type=str, required=True)
    required.add_argument('-a', '--annot', type=str, required=True)
    required.add_argument('-f', '--file', type=str, required=True)
    required.add_argument('-o', '--output', type=str, required=True)
    args = parser.parse_args()

    return {
        'root': args.root,
        'annot': args.annot,
        'file': args.file,
        'output': args.output
    }


def process_all_images(config):
    filenames = sorted(config.files)
    tables = {}
    segmentations = {}
    ensembles = {}
    for segmentation in config.segmentations:
        segmentations[segmentation] = {}
    ensembles['walker_binary'] = {}
    ensembles['walker_label'] = {}
    ensembles['union'] = {}
    methods = {'unet': {}, 'walker_binary': {}, 'walker_label': {}, 'opt': {}}
    for key in ['jac', 'af1', 'merge_rate', 'split_rate']:
        tables[key] = pd.DataFrame(columns=list(methods.keys()), copy=True)
    os.makedirs(config.output, exist_ok=True)
    root_dir = os.path.join(config.output, config.filename)
    if os.path.exists(root_dir):
        shutil.rmtree(root_dir)
    os.makedirs(root_dir, exist_ok=True)
    counter = 0
    for file in filenames:
        if counter % 50 == 0:
            print(counter)
        if counter == config.counter:
            break
        annot_path = os.path.join(config.annot, file.strip())
        annot = skimage.io.imread(annot_path, as_gray=True)
        for segmentation in segmentations.keys():
            path = os.path.join(config.root, segmentation, file.strip())
            segmentations[segmentation]['orig'] = skimage.io.imread(path ,as_gray=True)
            segmentations[segmentation]['results'] = comp.get_per_image_metrics(
                annot, segmentations[segmentation]['orig'], False)
            segmentations[segmentation]['mask'] = np.where(segmentations[segmentation]['orig'] > 0, 255, 0)
        for ensemble in ensembles.keys():
            ensembles[ensemble]['orig'] = Ensemble([segmentations[segmentation]['orig']
                                                    for segmentation in segmentations.keys()],
                                                   config.erosions, config.beta).ensemble(ensemble)
            ensembles[ensemble]['results'] = comp.get_per_image_metrics(annot, ensembles[ensemble]['orig'], False)
            ensembles[ensemble]['mask'] = np.where(ensembles[ensemble]['orig'] > 0, 255, 0)
        opt_ensemble = Optimize(segmentations, ensembles).optimize()

        for key in tables.keys():
            results = {}
            # for segmentation in segmentations.keys():
            #     results[segmentation] = segmentations[segmentation]['results'][key]
            avg = statistics.mean([segmentations[segmentation]['results'][key] for segmentation in segmentations])
            results['unet'] = avg
            for ensemble in ensembles.keys():
                if ensemble != 'union':
                    results[ensemble] = ensembles[ensemble]['results'][key]
            results['opt'] = ensembles[opt_ensemble]['results'][key]
            tables[key] = tables[key].append(results, ignore_index=True)
        counter += 1

    os.makedirs(os.path.join(root_dir, 'stats'), exist_ok=True)
    output_charts(tables, list(methods.keys()),
                  os.path.join(root_dir, 'stats'), config)


def output_charts(tables, methods, root_dir, config):
    bins = np.linspace(0, 1, 100)
    reference = {
        'jac': 'Jaccard Score',
        'af1': 'F1 Score',
        'merge_rate': 'Merge Rate',
        'split_rate': 'Split Rate'
    }
    for key in tables.keys():
        for method in methods:
            sns.distplot(tables[key][method].tolist(), hist=False, kde=True,
                         kde_kws={'shade': True, 'linewidth': 3}, bins=bins, label=method)
        plt.legend(loc='upper left') if (key == 'jac' or key == 'f1') else plt.legend()
        plt.title(f"{reference[key]} {config.filename.capitalize()} (erosions={config.erosions}, beta={config.beta})")
        plt.xlabel(reference[key])
        plt.savefig(os.path.join(root_dir, f'{reference[key].split()[0]}.png'))
        plt.clf()
        avg = tables[key].mean()
        std = tables[key].std()
        avg.plot.bar(yerr=std.tolist())
        plt.ylim(bottom=0)
        plt.xticks(rotation='horizontal')
        plt.title(
            f'{reference[key]} Averages {config.filename.capitalize()} (erosions={config.erosions}, beta={config.beta})')
        plt.xlabel('Method')
        plt.ylabel(f'{reference[key]}')
        plt.savefig(os.path.join(root_dir, f'{reference[key].split()[0]}_averages.png'))
        plt.clf()


if __name__ == '__main__':
    options = parse_arguments()

    CONFIG = Config(output=options['output'],
                    root=options['root'],
                    annot=options['annot'],
                    file=options['file'],
                    instance=False,
                    erosions=6,
                    beta=30,
                    counter=None)
    process_all_images(CONFIG)
