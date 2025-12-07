# check_bins.py
import numpy as np

def check_bins(path):
    bins = np.load(path)
    print(f"파일: {path}")
    print(f"- 길이: {len(bins)}")
    print(f"- 데이터 타입: {bins.dtype}")

    # 정수형 여부
    is_int = np.all(np.equal(np.mod(bins, 1), 0))
    print(f"- 정수형으로만 구성?: {is_int}")

    # 값 범위 체크
    print(f"- 최소값: {bins.min()}, 최대값: {bins.max()}")

    # 분포 출력
    print("- 분포 (value: count):")
    for v in range(5):
        print(f"  {v}: {(bins == v).sum()}")

    print("- 첫 20개 샘플:", bins[:20])
    print()

def main():
    check_bins("train_bins.npy")
    check_bins("val_bins.npy")

if __name__ == "__main__":
    main()

