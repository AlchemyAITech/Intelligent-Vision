import json

notebook_path = "sam3_image_predictor_example.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        new_source = []
        for line in source:
            # Fix using_colab
            if 'using_colab = True' in line:
                line = line.replace('using_colab = True', 'using_colab = False')
            
            # Remove condition for imports to ensure they run locally
            if 'if using_colab:' in line:
                continue # Skip the if statement
            
            # Fix device and autocast
            if 'torch.autocast("cuda"' in line:
                new_source.append('device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")\n')
                new_source.append('print(f"Using device: {device}")\n')
                new_source.append('if device == "cuda":\n')
                line = '    ' + line
            
            # Fix BPE path
            if 'bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"' in line:
                line = line.replace('/assets/', '/sam3/assets/')
            
            # Fix model build call with device
            if 'model = build_sam3_image_model(bpe_path=bpe_path)' in line:
                line = line.replace('build_sam3_image_model(bpe_path=bpe_path)', 'build_sam3_image_model(bpe_path=bpe_path, device=device)')

            new_source.append(line)
        cell['source'] = new_source

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook patched successfully.")
