import numpy as np
from rtree import index
import timeit
import matplotlib.pyplot as plt

# 전역 변수로 R-tree 인덱스 생성
p = index.Property()
idx = None


# 함수: R-tree 생성
def create_rtree(dataset):
    global idx
    p = index.Property()
    idx = index.Index(properties=p)

    for i, point in enumerate(dataset):
        x1, y1 = point
        x2, y2 = x1 + 1, y1 + 1  # 각 점을 나타내는 MBR 생성
        idx.insert(i, (x1, y1, x2, y2), obj=point)


# 함수: R-tree 검색
def search_rtree(query_points):
    global idx
    closest_points = []
    for point in query_points:
        closest_point = idx.nearest((point[0], point[1], point[0] + 1, point[1] + 1), 1, objects=True)
        closest_points.append((point, next(closest_point).object))

    return closest_points


# 데이터세트별 평균 검색 속도 계산
def calculate_average_search_speeds(datasets, query_points):
    average_search_speeds = []

    for dataset in datasets:
        create_rtree(dataset)  # R-tree 생성
        time_taken = timeit.timeit('search_rtree(query_points)', globals=globals(), number=10)
        average_search_speeds.append(time_taken / 10)  # 10번 실행의 평균 시간

    return average_search_speeds


# 메인 코드
dataset_sizes = [100000, 200000, 300000, 400000, 500000]
datasets = []

# 각 데이터세트에 대한 랜덤한 위치 생성
for size in dataset_sizes:
    x_coords = np.random.randint(0, 10000, size)
    y_coords = np.random.randint(0, 10000, size)
    dataset = np.vstack((x_coords, y_coords)).T
    datasets.append(dataset)

# 3. 랜덤 위치에 대한 검색 수행
query_points = np.random.randint(0, 10000, size=(100, 2))

# 4. R트리 검색 및 결과 출력
create_rtree(datasets[0])
closest_points = search_rtree(query_points)
for i, (query_point, closest_point) in enumerate(closest_points):
    print(f"Query Point {i + 1}: {query_point}, Closest Point: {closest_point}")

# 4-2. 데이터셋별 평균 검색 속도 비교 (그래프)
average_search_speeds = calculate_average_search_speeds(datasets, query_points)

plt.plot(dataset_sizes, average_search_speeds, marker='o')
plt.xlabel('Dataset Size')
plt.ylabel('Average Search Speed (seconds)')
plt.title('Average Search Speed vs. Dataset Size')
plt.show()
