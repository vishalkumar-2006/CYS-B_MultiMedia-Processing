from PIL import Image
import numpy as np

def compute_rate(labels, K):
    prob = np.array([(labels == i).mean() for i in range(K)])
    return -np.log2(np.maximum(prob, 1e-12))

def rd_kmeans(pixels, K, lam, iters=10):
    N = len(pixels)
    centroids = pixels[np.random.choice(N, K, replace=False)]
    labels = np.zeros(N, dtype=int)

    for _ in range(iters):
        rate = compute_rate(labels, K)
        D = np.sum((pixels[:, None] - centroids[None])**2, axis=2)
        labels = np.argmin(D + lam * rate, axis=1)

        for k in range(K):
            pts = pixels[labels == k]
            if len(pts) > 0:
                centroids[k] = pts.mean(axis=0)
    return centroids, labels

def run(path, K, lam):
    arr = np.array(Image.open(path).convert("RGB"))
    H, W = arr.shape[:2]
    pixels = arr.reshape(-1, 3).astype(float)

    cent, lbl = rd_kmeans(pixels, K, lam)
    out = cent[lbl].reshape(H, W, 3).astype(np.uint8)
    Image.fromarray(out).save("q2_cluster.jpg")
    print("clustered successfully")

K = int(input("Enter number of colors: "))
input="input2.jpg"
lam=0.01
run(input,K,lam)

