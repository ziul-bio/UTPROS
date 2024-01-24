#!/usr/bin/env python3 -u

import os
import torch

def save_embeddings_per_layer(embed_path, new_path):
    count = 0
    for file in os.listdir(embed_path):
        try:
            if file.endswith('.pt'):
                count += 1
                print(f'Splitting file number {count}: {file}')

                file_path = os.path.join(embed_path, file)
                tensor = torch.load(file_path)
                num_layers = len(tensor['mean_representations'])

                for layer in range(num_layers):
                    # Create a directory for the current layer if it doesn't exist
                    layer_dir = os.path.join(new_path, str(layer))
                    if not os.path.exists(layer_dir):
                        os.makedirs(layer_dir)

                    # Save tensor to the layer-specific directory
                    embed = {}
                    save_path = os.path.join(layer_dir, file)
                    embed['label'] = tensor['label']
                    embed['mean_representations'] = tensor['mean_representations'][layer]
                    torch.save(embed, save_path)

        except Exception as e:
            print(f"Error processing file {file}: {e}")

if __name__ == "__main__":
    embed_path = "/stor/work/Wilke/luiz/tail_stability/embeddings/tail_stability_esm2_15B_mean/"
    new_path = "/stor/work/Wilke/luiz/tail_stability/embeddings/esm2_15B_embeds_per_layer/"

    save_embeddings_per_layer(embed_path, new_path)

