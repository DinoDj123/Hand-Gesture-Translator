from collections import Counter

import numpy as np

import backend.video_processor as video_processor


def distance(a, b):
    return np.linalg.norm(a - b)


def predict(video, x, y, k=3):
    distances = []

    for i in range(len(x)):
        d = distance(video, x[i])
        distances.append((d, y[i]))

    distances.sort(key=lambda item: item[0])

    # k najbližih
    nearest = distances[:k]

    # majority vote
    labels = [label for _, label in nearest]
    most_common = Counter(labels).most_common(1)[0][0]

    # najbliži pojedinačni
    best_distance, best_label = distances[0]

    return best_label, best_distance


if __name__ == "__main__":
    x, y = video_processor.get_videos_and_labels()

    id = 3

    test_sample = x[id]

    x_train = np.delete(x, id, axis=0)
    y_train = np.delete(y, id, axis=0)

    prediction = predict(test_sample, x_train, y_train, k=1)

    print("Prava labela:", y[id])
    print("Predikcija:", prediction)
