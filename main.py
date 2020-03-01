import cv2
from math import sqrt


def calc_distance(p1, p2):
    (x1, y1) = p1
    (x2, y2) = p2
    return round(sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))


def select_kp(event, x, y, flags, param):
    global first_point, click_was_done, min_index

    if event == cv2.EVENT_LBUTTONDOWN:
        click_was_done = True
        kp1 = param[0]
        distances = []
        minvalue = 999999
        for i in range(len(kp1)):
            distances.append(calc_distance((x, y), kp1[i].pt))
            if distances[i] < minvalue:
                minvalue = distances[i]
                min_index = i
        if minvalue > 31:
            print("The requested point was not located. Please try again.")
        else:
            first_point = kp1[min_index]
            cv2.imshow('Left_with_kp', Left_with_kp)
            print("the closest point is at {} location {} with distance of {}".format(min_index, kp1[min_index].pt,
                                                                                      minvalue))


if __name__ == '__main__':

    n = 100
    m = 10
    # n = int(input("please insert n:\t"))
    for i in [3]:
        first_point = None
        click_was_done = False
        min_index = 0

        img1 = cv2.imread("images/img{}_L.jpg".format(i), cv2.IMREAD_COLOR)
        img2 = cv2.imread("images/img{}_R.JPG".format(i), cv2.IMREAD_COLOR)

        scale_percent = 20  # percent of original size
        width = int(img1.shape[1] * scale_percent / 100)
        height = int(img1.shape[0] * scale_percent / 100)
        dim = (width, height)

        Left = cv2.resize(img1, dim, interpolation=cv2.INTER_AREA)
        Right = cv2.resize(img2, dim, interpolation=cv2.INTER_AREA)

        orb = cv2.ORB_create(nfeatures=n)
        kp1, des1 = orb.detectAndCompute(Left, None)
        kp2, des2 = orb.detectAndCompute(Right, None)

        Left_with_kp = cv2.drawKeypoints(Left, kp1, None)
        param = [kp1]
        cv2.namedWindow("Left_with_kp")
        cv2.setMouseCallback("Left_with_kp", select_kp, param)
        while True:
            cv2.imshow('Left_with_kp', Left_with_kp)
            cv2.waitKey(1)

            if first_point is not None:
                break
        cv2.destroyAllWindows()

        """Part 2"""
        first_point_des1 = des1[min_index:min_index + 1]
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(first_point_des1, des2, k=m)
        matching_result = cv2.drawMatchesKnn(Left, [first_point], Right, kp2, matches, None, flags=2)

        cv2.imshow("Matching Result", matching_result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        """Part 3"""
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        with open("maximum_matching_cover.txt", "w+") as file:
            toprint = ""
            for m in matches:
                p1 = kp1[m.queryIdx].pt
                p2 = kp2[m.trainIdx].pt
                toprint += "{} <--> {}\n".format(p1, p2)
            file.write(toprint)
        matched_image = cv2.drawMatches(Left, kp1, Right, kp2, matches, None, flags=2)
        cv2.imshow("Maximum Matching", matched_image)
        cv2.waitKey(0)


