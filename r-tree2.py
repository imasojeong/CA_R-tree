import sys
import random
import time
import matplotlib.pyplot as plt


class RTreeNode:
    def __init__(self, is_leaf=True):
        self.is_leaf = is_leaf
        self.children = []
        self.bounding_box = None


class RTree:
    def __init__(self, max_entries=10000):
        self.root = RTreeNode()
        self.max_entries = max_entries

    # 1. (10000,10000)의 2차원 평면에 10000개, 20000개, 30000개, 40000개, 50000개의 위치를 랜덤하게 생성
    def add_dataset(self, dataset_sizes):
        for data_size in dataset_sizes:
            for i in range(data_size):
                x1 = random.randint(0, 10000)
                x2 = random.randint(0, 10000)
                y1 = random.randint(0, 10000)
                y2 = random.randint(0, 10000)
                data = f"Point({x1},{y1}) - ({x2},{y2})"
                bounding_box = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
                self.insert(data, bounding_box)
            print(data_size)

    def insert(self, data, bounding_box):
        entry = (data, bounding_box)
        node = self.choose_leaf(self.root, entry)
        node.children.append(entry)
        if len(node.children) > self.max_entries:
            self.split_node(node)

    def choose_leaf(self, node, entry):
        if node.is_leaf:
            return node
        min_increase = sys.maxsize
        selected_child = None
        for child in node.children:
            child_bb = child[1]
            enlarged_bb = self.enlarge_bb(child_bb, entry[1])
            increase = self.calculate_increase(child_bb, enlarged_bb)
            if increase < min_increase:
                min_increase = increase
                selected_child = child
        return self.choose_leaf(selected_child, entry)

    def enlarge_bb(self, bb1, bb2):
        if bb1 is None:
            return bb2
        return [min(bb1[0], bb2[0]), min(bb1[1], bb2[1]),
                max(bb1[2], bb2[2]), max(bb1[3], bb2[3])]

    def calculate_increase(self, bb1, bb2):
        e1 = self.calculate_area(bb1)
        e2 = self.calculate_area(bb2)
        enlarged = self.calculate_area(self.enlarge_bb(bb1, bb2))
        return enlarged - e1

    def calculate_area(self, bb):
        return (bb[2] - bb[0]) * (bb[3] - bb[1])

    def split_node(self, node):
        entries = node.children
        node.children = []
        node.bounding_box = None

        if node.is_leaf:
            group1, group2 = self.pick_seeds(entries)
        else:
            group1, group2 = self.pick_seeds(entries)
        node1 = RTreeNode(is_leaf=node.is_leaf)
        node2 = RTreeNode(is_leaf=node.is_leaf)
        node1.children.append(group1)
        node2.children.append(group2)

        while len(entries) > 0:
            if (len(node1.children) + len(entries)) <= (self.max_entries - 1):
                node1.children.extend(entries)
                entries = []
            elif (len(node2.children) + len(entries)) <= (self.max_entries - 1):
                node2.children.extend(entries)
                entries = []
            else:
                group = self.pick_next(entries, node1, node2)
                if group is not None:
                    if len(node1.children) + len(group) <= (self.max_entries - 1):
                        node1.children.extend(group)
                    elif len(node2.children) + len(group) <= (self.max_entries - 1):
                        node2.children.extend(group)
                    else:
                        node1.children.extend(group)
            node1.bounding_box = self.adjust_bounding_box(node1.bounding_box, group1[1])
            node2.bounding_box = self.adjust_bounding_box(node2.bounding_box, group2[1])

        node.children.append(group1)
        node.children.append(group2)

    def pick_seeds(self, entries):
        max_waste = -1
        seed1, seed2 = None, None

        for i in range(len(entries) - 1):
            for j in range(i + 1, len(entries)):
                bb1 = entries[i][1]
                bb2 = entries[j][1]
                waste = self.calculate_area(self.enlarge_bb(bb1, bb2)) - self.calculate_area(bb1) - self.calculate_area(bb2)
                if waste > max_waste:
                    max_waste = waste
                    seed1, seed2 = entries[i], entries[j]

        return seed1, seed2

    def pick_next(self, entries, group1, group2):
        max_difference = -1
        selected_entry = None

        for entry in entries:
            bb1 = group1.bounding_box if group1.bounding_box is not None else entry[1]
            bb2 = group2.bounding_box if group2.bounding_box is not None else entry[1]
            difference = abs(self.calculate_area(self.enlarge_bb(bb1, entry[1])) - self.calculate_area(bb1)) - abs(self.calculate_area(self.enlarge_bb(bb2, entry[1])) - self.calculate_area(bb2))
            if difference > max_difference:
                max_difference = difference
                selected_entry = entry

        return selected_entry

    def adjust_bounding_box(self, old_bb, new_bb):
        if old_bb is None:
            return new_bb

        return [min(old_bb[0], new_bb[0]), min(old_bb[1], new_bb[1]),
                max(old_bb[2], new_bb[2]), max(old_bb[3], new_bb[3])]

    # 3-2. R트리 검색의 결과로 질의 위치와 가장 가까운 위치 정보를 R트리에서 찾아서 반환
    def search(self, query_points):
        results = []
        for query_point in query_points:
            result = self.nearest_neighbor_search(self.root, query_point)
            results.append(result)
        return results

    def nearest_neighbor_search(self, node, query_point):
        if node.is_leaf:
            if not node.children:
                return None
            best_distance = sys.maxsize
            best_entry = None
            for entry in node.children:
                distance = self.calculate_distance(entry[1], query_point)
                if distance < best_distance:
                    best_distance = distance
                    best_entry = entry
            return best_entry
        else:
            best_entry = None
            best_distance = sys.maxsize
            for child in node.children:
                if child[1] is not None and self.point_inside_bb(query_point, child[1]):
                    result = self.nearest_neighbor_search(child, query_point)
                    if result is not None:
                        distance = self.calculate_distance(result[1], query_point)
                        if distance < best_distance:
                            best_distance = distance
                            best_entry = result
            return best_entry

    def point_inside_bb(self, point, bb):
        return bb[0] <= point[0] <= bb[2] and bb[1] <= point[1] <= bb[3]

    def calculate_distance(self, bb, point):
        if self.point_inside_bb(point, bb):
            return 0
        else:
            # Calculate the distance between the query point and the nearest point inside the bounding box
            left = max(bb[0], min(point[0], bb[2]))
            top = max(bb[1], min(point[1], bb[3]))
            return ((point[0] - left) ** 2 + (point[1] - top) ** 2) ** 0.5


if __name__ == '__main__':
    dataset_sizes = [10000, 20000, 30000, 40000, 50000]
    max_entries_per_dataset = [10000, 20000, 30000, 40000, 50000]

    search_times = [0] * len(dataset_sizes)
    query_point_count = 100

    for i, dataset_size in enumerate(dataset_sizes):
        rtree = RTree(max_entries=max_entries_per_dataset[i])
        rtree.add_dataset([dataset_size])

        query_points = [(random.randint(0, 10000), random.randint(0, 10000)) for _ in range(query_point_count)]

        results = rtree.search(query_points)

        start_time = time.time()
        for j, query_point in enumerate(query_points):
            result = results[j]
            if result:
                print(f"Query Point {j + 1}: Nearest neighbor to {query_point} is {result[0]} within the bounding box {result[1]}.")
            else:
                print(f"Query Point {j + 1}: No nearest neighbor found for {query_point}.")
        end_time = time.time()

        search_time = end_time - start_time
        search_times[i] = search_time

        print(f"Dataset Size: {dataset_size}, max_entries: {max_entries_per_dataset[i]}, Average Search Time: {search_times[i]} seconds")

    # 그래프 그리기
    plt.plot(dataset_sizes, search_times, marker='o', linestyle='-')
    plt.title('R-Tree Search Speed vs. Dataset Size')
    plt.xlabel('Dataset Size')
    plt.xticks(dataset_sizes, [f'{size}' for size in dataset_sizes])
    plt.ylabel('Search Time (seconds)')
    plt.grid()
    plt.show()