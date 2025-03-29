import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple
from config import TrainingConfig
import logging
import torch

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class ExperimentManager:
    def __init__(self, config: TrainingConfig, output_dir: str = 'experiments'):
        self.config = config
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = Path(output_dir) / f"{self.timestamp}_{config.subject}_{config.encoder_version}_{config.emb_type}_{config.paradigm}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Experiment Save Dir: {self.output_dir}")
        
        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(vars(config), f, indent=2)
            
        self.results = {
            'repeats': [],
            'ratios': [],
            'train_loss': [],
            'test_loss': [],
            'test_euclidean_dist': [],
            'test_cosine_sim': []
        }
        
        self.loss_histories = {}
        
    def get_history_key(self, repeats: int, ratio: float) -> tuple:
        return (repeats, ratio)
        
    def log_loss(self, epoch: int, train_loss: float, contrast_loss: float,
                 test_loss: float, test_euclidean_dist: float, test_cosine_sim: float,
                 repeats: int, ratio: float):
        key = self.get_history_key(repeats, ratio)
        
        if key not in self.loss_histories:
            self.loss_histories[key] = {
                'epoch': [],
                'train_loss': [],
                'contrast_loss': [],
                'test_loss': [],
                'test_euclidean_dist': [],
                'test_cosine_sim': []
            }
        
        history = self.loss_histories[key]
        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['contrast_loss'].append(contrast_loss)
        history['test_loss'].append(test_loss)
        history['test_euclidean_dist'].append(test_euclidean_dist)
        history['test_cosine_sim'].append(test_cosine_sim)
        
        df = pd.DataFrame(history)
        loss_history_dir = self.output_dir / 'loss_history'
        if not loss_history_dir.exists():
            loss_history_dir.mkdir(parents=True)
        df.to_csv(loss_history_dir / f'loss_history_r{repeats}_s{ratio:.1f}.csv', index=False)
        
        self._plot_loss_curves(repeats, ratio)
    
    def _plot_loss_curves(self, repeats: int, ratio: float):
        history = self.loss_histories[self.get_history_key(repeats, ratio)]
        epochs = range(len(history['train_loss']))
        
        plt.figure(figsize=(12, 12))
        
        plt.subplot(3, 1, 1)
        plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
        plt.plot(epochs, history['test_loss'], 'y-', label='Fixed Test Loss')
        plt.title(f'Loss Curves (repeats={repeats}, ratio={ratio:.1f})')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(3, 1, 2)
        plt.plot(epochs, history['test_euclidean_dist'], 'y-', label='Fixed Test Distance')
        plt.title('Euclidean Distance')
        plt.xlabel('Epoch')
        plt.ylabel('Distance')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(3, 1, 3)
        plt.plot(epochs, history['test_cosine_sim'], 'y-', label='Fixed Test Similarity')
        plt.title('Cosine Similarity')
        plt.xlabel('Epoch')
        plt.ylabel('Similarity')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        loss_curves_dir = self.output_dir / 'loss_curves'
        if not loss_curves_dir.exists():
            loss_curves_dir.mkdir(parents=True)
        plt.savefig(loss_curves_dir / f'loss_curves_r{repeats}_s{ratio:.1f}.png')
        plt.close()
        
    def log_result(self, repeats: int, ratio: float, metrics: Dict[str, float]):
        required_keys = ['repeats', 'ratios', 'train_loss',
                        'test_loss', 'test_euclidean_dist', 'test_cosine_sim']
        for key in required_keys:
            if key not in self.results:
                self.results[key] = []
        
        self.results['repeats'].append(repeats)
        self.results['ratios'].append(ratio)
        
        self.results['train_loss'].append(metrics['train_loss'])
        self.results['test_loss'].append(metrics['test_loss'])
        self.results['test_euclidean_dist'].append(metrics['test_euclidean_dist'])
        self.results['test_cosine_sim'].append(metrics['test_cosine_sim'])
        
        lengths = [len(arr) for arr in self.results.values()]
        if len(set(lengths)) > 1:
            logger.error(f"Inconsistent array lengths: {self.results}")
            raise ValueError(f"Inconsistent array lengths: {lengths}")
            
        df = pd.DataFrame(self.results)
        df.to_csv(self.output_dir / 'results.csv', index=False)
        
    def visualize_results(self):
        df = pd.DataFrame(self.results)
        
        self._plot_heatmap(
            df,
            metrics=[
                ('test_loss', 'Fixed Test Loss', 'YlOrRd'),
                ('test_euclidean_dist', 'Fixed Test Distance', 'YlGnBu'),
                ('test_cosine_sim', 'Fixed Test Cosine Similarity', 'RdYlGn')
            ],
            output_file=f'{self.config.subject}_{self.config.encoder_version}_{self.config.emb_type}_test_results_heatmap.png'
        )
    
        self._plot_heatmap(
            df,
            metrics=[
                ('test_cosine_sim', 'Fixed Test Cosine Similarity', 'RdYlGn')
            ],
            output_file=f'{self.config.subject}_{self.config.encoder_version}_{self.config.emb_type}_test_cosine_similarity_heatmap.png'
        )
    
    def _plot_heatmap(self, df: pd.DataFrame, metrics: List[Tuple[str, str, str]], output_file: str):
        pivot_tables = {}
        for metric, _, _ in metrics:
            pivot_tables[metric] = pd.pivot_table(
                df,
                values=metric,
                index='repeats',
                columns='ratios'
            )
        
        plot_num = len(metrics)
        fig, axes = plt.subplots(1, plot_num, figsize=(20, 6))
        
        for idx, (metric, title, cmap) in enumerate(metrics):
            pivot_data = pivot_tables[metric]
            sns.heatmap(
                pivot_data,
                ax=axes[idx],
                cmap=cmap,
                annot=True,
                fmt='.3f'
            )
            axes[idx].set_title(title)
            axes[idx].set_xlabel('Sentence Ratio')
            axes[idx].set_ylabel('Number of Repeats')
            
            x_ticks = axes[idx].get_xticks()
            x_labels = [f'{float(pivot_data.columns[int(x)]):.1f}' if x >= 0 and x < len(pivot_data.columns) else '' for x in x_ticks]
            axes[idx].set_xticklabels(x_labels)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / output_file, dpi=600, bbox_inches='tight')
        plt.close()
        
    def visualize_embeddings(self, eeg_embeddings, text_embeddings, categories, epoch, repeats, ratio):
        pass

    def log_category_result(self, num_categories: int, metrics: Dict[str, float]):
        if not hasattr(self, 'category_results'):
            self.category_results = {
                'num_categories': [],
                'train_loss': [],
                'test_loss': [],
                'test_euclidean_dist': [],
                'test_cosine_sim': []
            }
        
        self.category_results['num_categories'].append(num_categories)
        self.category_results['train_loss'].append(metrics['train_loss'])
        self.category_results['test_loss'].append(metrics['test_loss'])
        self.category_results['test_euclidean_dist'].append(metrics['test_euclidean_dist'])
        self.category_results['test_cosine_sim'].append(metrics['test_cosine_sim'])
        
        df = pd.DataFrame(self.category_results)
        df.to_csv(self.output_dir / 'category_results.csv', index=False)

    def visualize_category_results(self, results: List[Dict]):
        plt.figure(figsize=(20, 10))
        
        num_categories = self.category_results['num_categories']
        
        plt.subplot(3, 1, 1)
        plt.plot(num_categories, self.category_results['test_loss'], 'r-o', label='Test Loss')
        plt.title('Loss vs Number of Categories')
        plt.xlabel('Number of Categories')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        
        plt.subplot(3, 1, 2)
        plt.plot(num_categories, self.category_results['test_euclidean_dist'], 'b-o', label='Test Euclidean Distance')
        plt.title('Euclidean Distance vs Number of Categories')
        plt.xlabel('Number of Categories')
        plt.ylabel('Distance')
        plt.grid(True)
        plt.legend()
        
        plt.subplot(3, 1, 3)
        plt.plot(num_categories, self.category_results['test_cosine_sim'], 'g-o', label='Test Cosine Similarity')
        plt.title('Cosine Similarity vs Number of Categories')
        plt.xlabel('Number of Categories')
        plt.ylabel('Similarity')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'category_performance_curves.png', dpi=600, bbox_inches='tight')
        plt.close() 