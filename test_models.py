import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

log_file = "test_output.txt"

with open(log_file, "w", encoding="utf-8") as f:
    try:
        f.write("1: importing torch...\n"); f.flush()
        import torch
        f.write("2: torch OK\n"); f.flush()
        
        f.write("3: importing DistilBertModel...\n"); f.flush()
        from transformers import DistilBertModel
        f.write("4: DistilBertModel OK\n"); f.flush()
        
        f.write("5: importing BertModel...\n"); f.flush()
        from transformers import BertModel
        f.write("6: BertModel OK\n"); f.flush()
        
        f.write("7: importing torchvision vit_b_16...\n"); f.flush()
        import torchvision.models as models
        vit = models.vit_b_16
        f.write("8: ViT OK\n"); f.flush()
        
        f.write("\n=== ALL IMPORTS OK ===\n")
    except Exception as e:
        import traceback
        f.write(f"\nERROR: {e}\n")
        f.write(traceback.format_exc())

print("Done")
