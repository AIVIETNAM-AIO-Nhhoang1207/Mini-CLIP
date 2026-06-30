"""
Chạy benchmark tất cả models và tổng hợp kết quả.
Usage: python -m scripts.run_all_benchmarks
"""
import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Thêm project root vào path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark.utils import get_device
from benchmark.benchmark_mini_clip import benchmark_mini_clip
from benchmark.benchmark_clip import benchmark_clip
from benchmark.benchmark_openclip import benchmark_openclip
from benchmark.benchmark_siglip import benchmark_siglip


def print_comparison_table(all_results):
    """In bảng so sánh kết quả tất cả models."""
    print("\n")
    print("=" * 80)
    print("  BENCHMARK COMPARISON TABLE")
    print("=" * 80)

    # Header
    header = f"{'Model':<40} {'R@1':>6} {'R@5':>6} {'R@10':>6}"
    print(f"\n  Text-to-Image (T2I)")
    print(f"  {'-'*60}")
    print(f"  {header}")
    print(f"  {'-'*60}")
    for r in all_results:
        t2i = r["t2i"]
        print(f"  {r['name']:<40} {t2i['R@1']:>5.2f}% {t2i['R@5']:>5.2f}% {t2i['R@10']:>5.2f}%")

    print(f"\n  Image-to-Text (I2T)")
    print(f"  {'-'*60}")
    print(f"  {header}")
    print(f"  {'-'*60}")
    for r in all_results:
        i2t = r["i2t"]
        print(f"  {r['name']:<40} {i2t['R@1']:>5.2f}% {i2t['R@5']:>5.2f}% {i2t['R@10']:>5.2f}%")

    print(f"\n{'='*80}")


def main():
    device = get_device()
    print(f"Device: {device.type.upper()}")
    print("Running benchmarks for all models...\n")

    all_results = []

    # 1. Mini-CLIP
    print("=" * 60)
    print("  [1/4] Mini-CLIP (ResNet-18 + DistilBERT)")
    print("=" * 60)
    results = benchmark_mini_clip(device=device)
    all_results.append(results)

    # 2. CLIP
    print("\n" + "=" * 60)
    print("  [2/4] CLIP (ViT-B/32, OpenAI)")
    print("=" * 60)
    results = benchmark_clip(device=device)
    all_results.append(results)

    # 3. OpenCLIP
    print("\n" + "=" * 60)
    print("  [3/4] OpenCLIP (ViT-B-32, LAION-2B)")
    print("=" * 60)
    results = benchmark_openclip(device=device)
    all_results.append(results)

    # 4. SigLIP
    print("\n" + "=" * 60)
    print("  [4/4] SigLIP (siglip-base-patch16-224)")
    print("=" * 60)
    results = benchmark_siglip(device=device)
    all_results.append(results)

    # Bảng tổng hợp
    print_comparison_table(all_results)


if __name__ == "__main__":
    main()
