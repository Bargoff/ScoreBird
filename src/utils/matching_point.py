import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from typing import List, Dict, Tuple, ByteString

from src.utils.point import MatchingPoint, Point


def findTemplateMatchingPoints(image_bgr: np.ndarray,
                               template_filename,
                               threshold: float) -> Dict[Tuple, MatchingPoint]:
    # Find the points on an image that match the template image above some threshold.
    matching_points_dict = {}

    # Convert base image and template to grayscale in order to perform template matching
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    template = cv2.imread(str(template_filename), cv2.IMREAD_GRAYSCALE)

    try:
        # Match the template onto the image into resulting points that can match the template
        res = cv2.matchTemplate(image_gray, template, cv2.TM_CCOEFF_NORMED)

        # Find points (which are the top-leftmost of the template) where the template matches above the threshold
        loc = np.where(res >= threshold)

        # The matching points are sorted by ascending y values (the smallest y value is first)
        matching_points = list(zip(*loc[::-1]))
        matching_value_list = res[res >= threshold]

        for i, point in enumerate(matching_points):
            value = matching_value_list[i]
            matching_points_dict[point] = MatchingPoint(point, value)

        return matching_points_dict

    except cv2.error:
        return {}

def findBestMatchingPoints(matching_points_dict: Dict[Tuple, MatchingPoint]) -> List[Tuple]:
    # Using a matching points dictionary, find the points with the
    # highest matching value within clusters of neighboring points.
    # Often, there will be a few to a dozen points that match a single template above a
    # threshold, but only one of those points in the cluster matches "the best".

    # Use the DBSCAN clustering algorithm to group all matching points into clusters.
    distance = 6
    matching_points_list = [point for point in matching_points_dict]
    dbscan = DBSCAN(eps=distance, min_samples=1).fit(matching_points_list)
    cluster_labels = dbscan.labels_
    num_clusters = max(cluster_labels) + 1

    # Initialize the point clusters list of lists as empty
    point_clusters = []
    for i in range(num_clusters):
        point_clusters.append([])

    # Add the matching points to their point cluster list
    for i, point in enumerate(matching_points_list):
        point_clusters[cluster_labels[i]].append(point)

    # Find the point with the maximum matching value for verification and display purposes in each cluster
    best_matching_points = []
    for i, point_cluster in enumerate(point_clusters):
        max_value = 0
        best_point = None
        for point in point_cluster:
            matching_points_dict[point].setCluster(i)
            point_value = matching_points_dict[point].value
            if point_value > max_value:
                max_value = point_value
                best_point = point
        best_matching_points.append(best_point)

    return best_matching_points
