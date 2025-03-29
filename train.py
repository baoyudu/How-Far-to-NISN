import torch
import numpy as np
from torch.utils.data import DataLoader
from eeg_encoder import EEGEncoder
from eeg_encoder_2 import EEGEncoder2
from eeg_encoder_3 import EEGEncoder3
from losses import ContrastiveLoss, TripletLoss
from data_loader import EEGDataLoader
from config import TrainingConfig
from experiment import ExperimentManager
from typing import Optional, List
import random
import argparse

def evaluate_model(model, test_loader, contrastive_loss, device, config):
    model.eval()
    total_loss = 0
    total_contrast_loss = 0
    total_euclidean_dist = 0
    total_cosine_sim = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in test_loader:
            eeg_data = batch['eeg_data'].to(device)
            text_embeddings = batch['embeddings'][config.lang][config.emb_type].to(device)
            categories = batch['category'].to(device)
            
            eeg_embeddings = model(eeg_data)
            
            loss = contrastive_loss(eeg_embeddings, text_embeddings)
            
            eeg_embeddings_norm = torch.nn.functional.normalize(eeg_embeddings, dim=1)
            text_embeddings_norm = torch.nn.functional.normalize(text_embeddings, dim=1)
            euclidean_distances = torch.norm(eeg_embeddings_norm - text_embeddings_norm, dim=1)
            avg_euclidean_dist = euclidean_distances.mean().item()
            
            cosine_similarities = torch.nn.functional.cosine_similarity(eeg_embeddings, text_embeddings)
            avg_cosine_sim = cosine_similarities.mean().item()
            
            total_loss += loss.item()
            total_contrast_loss += loss.item()
            total_euclidean_dist += avg_euclidean_dist * len(categories)
            total_cosine_sim += avg_cosine_sim * len(categories)
            total_samples += len(categories)
    
    avg_loss = total_loss / len(test_loader)
    avg_contrast_loss = total_contrast_loss / len(test_loader)
    avg_euclidean_dist = total_euclidean_dist / total_samples
    avg_cosine_sim = total_cosine_sim / total_samples
    
    return {
        'test_loss': avg_loss,
        'test_contrast_loss': avg_contrast_loss,
        'test_euclidean_dist': avg_euclidean_dist,
        'test_cosine_sim': avg_cosine_sim
    }

def collate_fn(batch, max_seq_len: Optional[int] = None, pad_sequences: bool = True):
    seq_lengths = [item['eeg_data'].shape[1] for item in batch]
    
    if max_seq_len is not None:
        target_len = max_seq_len
    else:
        target_len = max(seq_lengths) if pad_sequences else min(seq_lengths)
    
    processed_batch = []
    for item in batch:
        curr_len = item['eeg_data'].shape[1]
        if curr_len < target_len:
            padding = torch.zeros(item['eeg_data'].shape[0], target_len - curr_len)
            eeg_data = torch.cat([item['eeg_data'], padding], dim=1)
        else:
            eeg_data = item['eeg_data'][:, :target_len]
            
        processed_batch.append({
            'eeg_data': eeg_data,
            'embeddings': item['embeddings'],
            'category': item['category'] if isinstance(item['category'], torch.Tensor) 
                       else torch.tensor(item['category'])
        })
    
    return {
        'eeg_data': torch.stack([item['eeg_data'] for item in processed_batch]),
        'embeddings': {
            lang: {
                emb_type: torch.stack([item['embeddings'][lang][emb_type] for item in processed_batch])
                for emb_type in batch[0]['embeddings'][lang]
            }
            for lang in batch[0]['embeddings']
        },
        'category': torch.stack([item['category'] for item in processed_batch])
    }

def print_embedding_norms(eeg_embeddings, text_embeddings, n_samples=5):
    print("\nVerifying embedding normalization:")
    print("-" * 50)
    
    n_samples = min(n_samples, len(eeg_embeddings))
    
    for i in range(n_samples):
        eeg_norm = torch.norm(eeg_embeddings[i], p=2).item()
        text_norm = torch.norm(text_embeddings[i], p=2).item()
        print(f"Sample {i + 1}:")
        print(f"  EEG Embedding L2 norm: {eeg_norm:.6f}")
        print(f"  Text Embedding L2 norm: {text_norm:.6f}")
    print("-" * 50)

def train_one_configuration(
    config: TrainingConfig,
    n_repeats: int,
    sentence_ratio: float,
    experiment: ExperimentManager,
    data_loader: EEGDataLoader,
    mode: str = 'grid'
):
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print("\n" + "="*50)
    print(f"Training Configuration:")
    print(f"- Sentence ratio: {sentence_ratio:.2f}")
    print(f"- Number of repeats: {n_repeats}")
    print(f"- Categories: {config.categories}")
    print("="*50 + "\n")
    
    data_splits = data_loader.get_all_splits(
        paradigm=config.paradigm,
        subject=config.subject,
        categories=config.categories,
        sentence_ratio=sentence_ratio,
        train_repeats=n_repeats,
        lang=config.lang,
        emb_type=config.emb_type,
        mode=mode
    )
    
    max_seq_len = None
    if config.max_seq_len is not None and config.pad_sequences:
        max_seq_len = config.max_seq_len
    elif config.pad_sequences:
        max_seq_len = max(
            max(item['eeg_data'].shape[1] for item in data_splits['train']),
            max(item['eeg_data'].shape[1] for item in data_splits['test'])
        )
        if config.debug_data_loading:
            print(f"Maximum sequence length across all datasets: {max_seq_len}")
    
    collate_with_config = lambda batch: collate_fn(
        batch, 
        max_seq_len=max_seq_len, 
        pad_sequences=config.pad_sequences
    )
    
    train_loader = DataLoader(
        data_splits['train'], 
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_with_config
    )
    test_loader = DataLoader(
        data_splits['test'],
        batch_size=config.batch_size,
        collate_fn=collate_with_config
    )
    
    if config.debug_data_loading:
        print(f"\nDataset sizes:")
        print(f"- Train set: {len(data_splits['train'])} samples")
        print(f"- Test set: {len(data_splits['test'])} samples\n")
    
    if config.encoder_version == 1:
        print("encoder1 is deprecated, please use encoder2 or encoder3")
        exit()
        model = EEGEncoder(
            embedding_dim=config.embedding_dim,
            dropout_rate=config.eegcnn_dropout
        ).to(config.device)
    elif config.encoder_version == 2:
        model = EEGEncoder2(
            embedding_dim=config.embedding_dim,
            dropout_rate=config.eegcnn_dropout,
            projection_dropout=config.projection_dropout
        ).to(config.device)
    else:
        model = EEGEncoder3(
            embedding_dim=config.embedding_dim,
            projection_dropout=config.projection_dropout,
            debug_output=config.debug_model_structure
        ).to(config.device)
    
    if config.debug_model_structure:
        print("\nModel Architecture:")
        print("="*50)
        print(model)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("\nModel Parameters:")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print("="*50 + "\n")
    
    if config.debug_model_structure and config.encoder_version == 2:
        print("Layer output shapes:")
        print("-"*50)
        sample_batch = next(iter(train_loader))
        sample_eeg = sample_batch['eeg_data'][:1].to(config.device)
        
        with torch.no_grad():
            print(f"Input shape: {sample_eeg.shape}")
            
            features = model.eegcnn(sample_eeg)
            print(f"EEGcnn output shape: {features.shape}")
            
            features_flat = features.view(features.size(0), -1)
            print(f"Flattened features shape: {features_flat.shape}")
            
            output = model(sample_eeg)
            print(f"Final output shape: {output.shape}")
        print("-"*50 + "\n")
    
    contrastive_loss = ContrastiveLoss(config.temperature).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
        min_lr=config.scheduler_min_lr,
        verbose=True
    )
    
    best_test_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(config.num_epochs):
        model.train()
        avg_loss = 0.0
        avg_contrast_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            eeg_data = batch['eeg_data'].to(config.device)
            text_embeddings = batch['embeddings'][config.lang][config.emb_type].to(config.device)
            categories = batch['category'].to(config.device)
            
            optimizer.zero_grad()
            
            eeg_embeddings = model(eeg_data)
            
            if config.debug_embedding_norms and batch_idx % config.debug_embedding_norm_freq == 0:
                print_embedding_norms(eeg_embeddings, text_embeddings)
            
            loss = contrastive_loss(eeg_embeddings, text_embeddings)
            
            loss.backward()
            
            if config.debug_gradient_flow:
                for name, param in model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        print(f"{name}: grad norm = {param.grad.norm().item():.6f}")
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            
            optimizer.step()
            
            avg_loss += loss.item()
            avg_contrast_loss += loss.item()
            
            if (batch_idx + 1) % config.log_every_n_batches == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_loader)}")
                print(f"Loss: {loss.item():.4f}")
                
                if config.debug_memory_usage:
                    print("\nMemory Usage:")
                    print(f"Allocated: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
                    print(f"Cached: {torch.cuda.memory_reserved()/1024**2:.1f}MB")
        
        avg_loss /= len(train_loader)
        avg_contrast_loss /= len(train_loader)
        
        test_metrics = evaluate_model(model, test_loader, contrastive_loss, config.device, config)
        
        scheduler.step(test_metrics['test_loss'])
        
        experiment.log_loss(
            epoch=epoch,
            train_loss=avg_loss,
            contrast_loss=avg_contrast_loss,
            test_loss=test_metrics['test_loss'],
            test_euclidean_dist=test_metrics['test_euclidean_dist'],
            test_cosine_sim=test_metrics['test_cosine_sim'],
            repeats=n_repeats,
            ratio=sentence_ratio
        )
        
        if test_metrics['test_loss'] < best_test_loss - config.min_delta:
            best_test_loss = test_metrics['test_loss']
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            best_model_dir = experiment.output_dir / 'best_model'
            if not best_model_dir.exists():
                best_model_dir.mkdir(parents=True)
            torch.save(best_model_state, 
                      best_model_dir / f'best_model_r{n_repeats}_s{sentence_ratio:.1f}.pt')
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    final_test_metrics = evaluate_model(model, test_loader, contrastive_loss, config.device, config)
    
    final_metrics = {
        'train_loss': avg_loss,
        'test_loss': final_test_metrics['test_loss'],
        'test_euclidean_dist': final_test_metrics['test_euclidean_dist'],
        'test_cosine_sim': final_test_metrics['test_cosine_sim']
    }
    
    if config.debug_output:
        print("\nFinal Test Set Performance:")
        print(f"Test Loss: {final_test_metrics['test_loss']:.4f}")
        print(f"Test Euclidean Distance: {final_test_metrics['test_euclidean_dist']:.4f}")
        print(f"Test Cosine Similarity: {final_test_metrics['test_cosine_sim']:.4f}")
    
    return final_metrics

def grid_search_training(config: TrainingConfig):
    print(f"Experiment Manager: {config.note}")
    
    data_loader = EEGDataLoader(config.data_path)
    experiment = ExperimentManager(config)
    
    repeats_range = range(config.repeat_range[0], 
                         config.repeat_range[1] + 1,
                         config.repeat_step)
    ratios_range = np.arange(config.ratio_range[0],
                            config.ratio_range[1] + 0.01,
                            config.ratio_step)
    
    for n_repeats in repeats_range:
        for ratio in ratios_range:
            print(f"\nTraining with {n_repeats} repeats and {ratio:.1f} ratio")
            metrics = train_one_configuration(
                config=config, 
                n_repeats=n_repeats, 
                sentence_ratio=ratio, 
                experiment=experiment,
                data_loader=data_loader,
                mode='grid'
            )
            experiment.log_result(n_repeats, ratio, metrics)
    
    experiment.visualize_results()

def category_linear_search_training(config: TrainingConfig):
    print(f"Category Linear Search Training: {config.note}")
    
    data_loader = EEGDataLoader(config.data_path)
    experiment = ExperimentManager(config)
    
    all_categories = config.categories.copy()
    all_categories.sort()
    
    results = []
    
    for num_categories in range(1, len(all_categories) + 1):
        current_categories = all_categories[:num_categories]
        print(f"\nTraining with {num_categories} categories: {current_categories}")
        
        config.categories = current_categories
        
        metrics = train_one_configuration(
            config=config,
            n_repeats=3,
            sentence_ratio=1.0,
            experiment=experiment,
            data_loader=data_loader,
            mode='category'
        )
        
        results.append({
            'num_categories': num_categories,
            'categories': current_categories,
            'metrics': metrics
        })
        
        experiment.log_category_result(num_categories, metrics)
    
    experiment.visualize_category_results(results)
    
    return results

def get_config_grid(emb_type='jina', exp_id=None):
    embedding_dim = 768 if emb_type == 'labse' else 1024
    
    return TrainingConfig(
        data_path="data/target.h5",
        paradigm="para02",
        subject="sub01",
        categories=[7, 38],
        encoder_version=2,
        pad_sequences=True,
        eegcnn_dropout=0.1,
        projection_dropout=0.1,
        random_seed=44,
        seed=44,
        gradient_clip=1.0,
        patience=10,
        min_delta=1e-4,
        log_every_n_batches=10,
        scheduler_factor=0.5,
        scheduler_patience=3,
        scheduler_min_lr=1e-5,
        emb_type=emb_type,
        embedding_dim=embedding_dim,
        debug_output=True,
        debug_model_structure=True,
        repeat_range=(2, 14),
        ratio_range=(0.2, 1.0),
        repeat_step=1,
        ratio_step=0.1,
        note=f'Grid Search Training_{exp_id}' if exp_id else 'Grid Search Training'
    )

def get_config_category_expansion(emb_type='jina', exp_id=None):
    embedding_dim = 768 if emb_type == 'labse' else 1024
    
    return TrainingConfig(
        data_path="data/target.h5",
        paradigm="para01",
        subject="sub02", 
        categories=[38, 4, 0, 7, 25, 29, 8, 22],
        encoder_version=2,
        pad_sequences=True,
        eegcnn_dropout=0.1,
        projection_dropout=0.1,
        random_seed=44,
        seed=44,
        gradient_clip=1.0,
        patience=10,
        min_delta=1e-4,
        log_every_n_batches=10,
        scheduler_factor=0.5,
        scheduler_patience=3,
        scheduler_min_lr=1e-5,
        emb_type=emb_type,
        embedding_dim=embedding_dim,
        debug_output=True,
        debug_model_structure=False,
        debug_data_loading=True,
        note=f'Category Linear Search Training_{exp_id}' if exp_id else 'Category Linear Search Training'
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='EEG Training Program')
    parser.add_argument('--mode', type=str, choices=['grid', 'category'], 
                      default='grid', help='Training mode: grid (Grid Search) or category (Category Linear Search)')
    parser.add_argument('--paradigm', type=str, default='para02', help='Paradigm name')
    parser.add_argument('--subject', type=str, default='sub01', help='Subject ID')
    parser.add_argument('--encoder', type=int, default=2, choices=[1, 2, 3], help='Encoder version')
    parser.add_argument('--emb_type', type=str, default='jina', choices=['jina', 'labse'], help='Embedding type')
    parser.add_argument('--exp_id', type=str, default=None, help='Experiment ID for distinguishing different experiments')
    args = parser.parse_args()

    if args.mode == 'grid':
        config = get_config_grid(args.emb_type, args.exp_id)
        config.paradigm = args.paradigm
        config.subject = args.subject
        config.encoder_version = args.encoder
        config.note = f'Grid Search Training_{args.exp_id}' if args.exp_id else f'Grid Search Training ({args.emb_type}, encoder{args.encoder})'
        
        print("\nExecuting Grid Search Training...")
        grid_search_training(config)
    else:
        config = get_config_category_expansion(args.emb_type, args.exp_id)
        config.paradigm = args.paradigm
        config.subject = args.subject
        config.encoder_version = args.encoder
        config.note = f'Category Linear Search Training_{args.exp_id}' if args.exp_id else f'Category Linear Search Training ({args.emb_type}, encoder{args.encoder})'
        
        print("\nExecuting Category Linear Search Training...")
        category_linear_search_training(config) 