#!/usr/bin/env python3 -u

# how to run:
# python scripts/split_layers.py -I embeddings/data_set_AllLayers_mean -O embeddings/data_set_AllLayers_splited


import os
import torch
import argparse

def save_embeddings_per_layer(embed_path, new_path):
    count = 0
    for file in os.listdir(embed_path):
        try:
            if file.endswith('.pt'):
                count += 1
                file_path = os.path.join(embed_path, file)
                tensor = torch.load(file_path)
                num_layers = len(tensor['mean_representations'])

                if count % 1000 == 0:
                    print(f'Processed {count} files')

                for layer in range(num_layers):
                    layer_dir = os.path.join(new_path, str(layer))
                    # Create a directory for the current layer if it doesn't exist
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

    parser = argparse.ArgumentParser(description='Split embeddings per layer')
    parser.add_argument('-I', '--embed_path', type=str, help='Path to the embeddings to split')
    parser.add_argument('-O', '--out_path', type=str, help='Path to save the split embeddings')

    args = parser.parse_args()
    embed_path = args.embed_path
    out_path = args.out_path
    os.makedirs(out_path, exist_ok=True)

    save_embeddings_per_layer(embed_path, out_path)

