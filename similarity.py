import h5py
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import logging
from typing import Dict, List

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_embeddings(h5_path: str, paradigm: str = 'para01', subject: str = 'sub01', 
                   lang: str = None, emb_type: str = 'jina') -> Dict[str, np.ndarray]:
    languages = ['zh', 'en', 'fr', 'es'] if lang is None else [lang]
    embeddings_dict = defaultdict(list)
    
    with h5py.File(h5_path, 'r') as f:
        paradigm_group = f[paradigm]
        subject_group = paradigm_group[subject]
        
        for category in subject_group.keys():
            category_group = subject_group[category]
            logger.info(f"Processing category: {category}")
            
            for sentence in category_group.keys():
                sentence_group = category_group[sentence]
                
                if 'embeddings' in sentence_group:
                    emb_group = sentence_group['embeddings']
                    
                    for lang_code in languages:
                        if lang_code in emb_group:
                            lang_group = emb_group[lang_code]
                            if emb_type in lang_group:
                                embedding = lang_group[emb_type][()]
                                embeddings_dict[lang_code].append(embedding)
    
    for lang_code in embeddings_dict:
        if embeddings_dict[lang_code]:
            embeddings_dict[lang_code] = np.vstack(embeddings_dict[lang_code])
            logger.info(f"Loaded {len(embeddings_dict[lang_code])} embeddings for language {lang_code}")
    
    return embeddings_dict

def calculate_cosine_similarities(embeddings_dict: Dict[str, np.ndarray]) -> tuple:
    languages = list(embeddings_dict.keys())
    n_langs = len(languages)
    
    similarity_matrix = np.zeros((n_langs, n_langs))
    
    for i, lang1 in enumerate(languages):
        for j, lang2 in enumerate(languages):
            if i == j:
                similarity_matrix[i,j] = 1.0
            else:
                emb1 = embeddings_dict[lang1]
                emb2 = embeddings_dict[lang2]
                if len(emb1) != len(emb2):
                    logger.warning(f"Number of embeddings mismatch: {lang1}({len(emb1)}) vs {lang2}({len(emb2)})")
                    continue
                
                similarities = []
                for idx in range(len(emb1)):
                    sim = cosine_similarity(emb1[idx:idx+1], emb2[idx:idx+1])[0,0]
                    similarities.append(sim)
                
                avg_sim = np.mean(similarities)
                similarity_matrix[i,j] = avg_sim
                logger.info(f"Average similarity between {lang1} and {lang2}: {avg_sim:.3f}")
    
    return similarity_matrix, languages

def plot_heatmap(similarity_matrix: np.ndarray, languages: List[str], 
                output_path: str = 'embedding_similarity_heatmap.png') -> None:
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, 
                xticklabels=languages,
                yticklabels=languages,
                annot=True,
                fmt='.3f',
                cmap='YlOrRd')
    
    plt.title('Cross-lingual Embedding Similarity')
    plt.tight_layout()
    plt.savefig(output_path, dpi=600)
    plt.close()
    logger.info(f"Heatmap saved to {output_path}")

def main():
    h5_path = "data/embedding_data/target_embedding.h5"
    paradigm = "para02"
    subject = "sub01"
    emb_type = "labse"
    output_path = f"embedding_similarity_heatmap_{emb_type}.png"
    
    logger.info("Loading embeddings...")
    embeddings_dict = load_embeddings(h5_path, paradigm, subject, emb_type=emb_type)
    
    if not embeddings_dict:
        logger.error("No embeddings were loaded!")
        return
    
    logger.info("Calculating similarities...")
    similarity_matrix, languages = calculate_cosine_similarities(embeddings_dict)
    
    logger.info("Plotting heatmap...")
    plot_heatmap(similarity_matrix, languages, output_path)

if __name__ == "__main__":
    main()
