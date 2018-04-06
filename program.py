import sys
import cv2
import numpy as np
import os
from scipy.stats.stats import pearsonr


def getLine(p1, p2):
    #     if p1[0]!=p2[0]:
    #         a = (p1[1]-p2[1])/(p1[0]-p2[0])
    #         b = p1[1]-a*p1[0]
    #         return (a,b)
    a = p1[1] - p2[1]
    b = -p1[0] + p2[0]
    c = -a * p1[0] - b * p1[1]

    return (a, b, c)


def getDistanceFromLine(p, l):
    return np.abs(l[0] * p[0] - l[1] * p[1] + l[2]) / np.sqrt(l[0] ** 2 + l[1] ** 2)


#     return np.abs(l[0]*p[0]-p[1]+l[1])/np.sqrt(l[0]**2 + 1)

def getLinesCrossing(a1, b1, c1, a2, b2, c2):
    d = a1 * b2 - a2 * b1
    if d != None:
        x = (b1 * c2 - b2 * c1) / d
        y = -(a1 * c2 - a2 * c1) / d
        return [x, y]


def canBeSide(A, B, defects):
    return True
    return tuple(np.append(A[0], B[0])) not in defects


def canBeSides(points, defects):
    result = True
    for i in range(len(points)):
        result = result and canBeSide(points[i], points[(i + 1) % len(points)], defects)
        if not result: break
    return result


def getPointsToCutOut(points, others):
    p1, p2, p3, p4 = points[:, 0, :]
    l1 = getLine(p1, p2)
    l2 = getLine(p2, p3)
    l3 = getLine(p3, p4)

    maxd = -1
    p5 = None
    others = np.append(others, [[p1], [p4]], axis=0)
    for pi in others[:, 0, :]:
        d = getDistanceFromLine(pi, l2)
        if d >= maxd:
            maxd = d
            p5 = pi

    l4 = (l2[0], l2[1], -l2[0] * p5[0] - l2[1] * p5[1])
    p6 = getLinesCrossing(*l1, *l4)
    p7 = getLinesCrossing(*l3, *l4)
    return np.array([[p6], [p2], [p3], [p7]])  # [[0,0],[0,height],[width,height],[width,0]]


def checkIfCorrect(points, width, height):
    for p in points:
        if p[0] < 0 or p[0] > width or p[1] < 0 or p[1] > height:
            return False
    return True


def setProperRotation(img, threshold=0.2):
    rows, cols = img.shape
    w = int(threshold * cols)
    h = int(threshold * rows)
    avrgs = [img[:h, :].mean()]  # od góry
    avrgs += [img[-h:, :].mean()]  # od dołu
    avrgs += [img[:, :w].mean()]  # od lewej
    avrgs += [img[:, -w:].mean()]  # od prawej

    idx = 1
    maxavrg = -1
    for i, avrg in enumerate(avrgs):
        if avrg > maxavrg:
            maxavrg = avrg
            idx = i

    if idx == 1:
        return img
    elif idx == 2:
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
    elif idx == 3:
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -90, 1)
    elif idx == 0:
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 180, 1)
    return cv2.warpAffine(img, M, (cols, rows))


def func(column):
    last_cell = column[0]
    w = 0
    for cell in column[1:]:
        if cell < last_cell:
            w += 1
        elif cell > last_cell:
            last_cell = cell
    return w


def getPropositions(path):
    image = cv2.imread(path, 0)  # 0 - greyscale
    #     image = image[:, ~np.all(image==0, axis=0)]
    #     image = image[~np.all(image==0, axis=1), :]
    _, contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    chull = cv2.convexHull(cnt)
    epsilon = 0.01 * cv2.arcLength(chull, True)
    approx = cv2.approxPolyDP(chull, epsilon, True)

    defects = cv2.convexityDefects(cnt, cv2.convexHull(cnt, returnPoints=False))
    #     print(np.array([[cnt[s],cnt[e]] for [[s,e,_,_]] in defects]))
    #     plt.figure()
    #     plt.imshow(image)

    selections = []
    height, width = image.shape
    for i in range(len(approx)):
        vertices = np.empty((0, 1, 2))
        others = np.empty((0, 1, 2))
        for j in range(len(approx)):
            k = (i + j) % len(approx)
            if j < 4:
                vertices = np.append(vertices, [approx[k]], axis=0)
            else:
                others = np.append(others, [approx[k]], axis=0)
        if canBeSides(vertices, defects):
            selection = getPointsToCutOut(vertices, others)
            #             selections += [selection]
            if checkIfCorrect(selection[:, 0, :], width, height):
                selections += [selection]

    propositions = []
    negpropositions = []
    width, height = 300, 300
    border = 10
    d = 0
    for sel in selections:
        pts2 = np.float32([[d, d], [d, height - d], [width - d, height - d], [width - d, d]])
        pts1 = np.float32(sel[:, 0, :])

        H = cv2.findHomography(pts1, pts2)[0]

        h1, w1 = image.shape
        h2, w2 = image.shape
        pts1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
        pts2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
        pts2_ = cv2.perspectiveTransform(pts1, H)
        pts = np.concatenate((pts1, pts2_), axis=0)
        [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
        t = [-xmin, -ymin]
        Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])  # translate
        #         print(xmax-xmin, ymax-ymin, (xmax-xmin) * (ymax-ymin))
        if (xmax - xmin) * (ymax - ymin) < 0 or (xmax - xmin) * (ymax - ymin) > 7500 * 7500:
            continue
        dst = cv2.warpPerspective(image, Ht.dot(H), (xmax - xmin, ymax - ymin))
        dst[dst != 0] = dst.max()

        dst = dst[:, ~np.all(dst == 0, axis=0)]  # góra i dół
        dst = dst[~np.all(dst == 0, axis=1), :]  # lewo i prawo
        dst = dst[:, ~np.all(dst != 0, axis=0)]
        dst = dst[~np.all(dst != 0, axis=1), :]

        dst = setProperRotation(dst)
        dst = dst[:-border, border:-border]
        dst = cv2.resize(dst, (width, height))
        dst[dst != 0] = dst.max()

        #         print(path,np.apply_along_axis(func,0,dst).sum())
        if np.apply_along_axis(func, 0, dst).sum() < 120:
            dst = dst[:, ~np.all(dst == 0, axis=0)]
            dst = dst[~np.all(dst == 0, axis=1), :]
            dst = dst[:, ~np.all(dst != 0, axis=0)]
            dst = dst[~np.all(dst != 0, axis=1), :]
            dst = cv2.resize(dst, (width, height))

            neg = dst.max() - dst

            propositions += [dst]
            negpropositions += [neg]
    return propositions, negpropositions


def binarySearch(column):
    n = len(column)
    i = int(n / 2)
    d = int(n / 2)
    #     print(i,d,n)
    #     print(column[i] == column[i+1])
    while column[i] == column[i + 1]:
        d /= 2
        if d < 1:
            d = 1
        d = int(d)
        if column[i] != 0:
            i -= d
        else:
            i += d
        if i < 0:
            return 0
        if i >= n - 1:
            return i
    return i


def findEdge(img):
    edge = []
    img = (img > 150) * 255
    for column in img.T:
        edge.append(binarySearch(column))
    return edge


def compareImagesCorr(imgs):
    edges = []
    for i, imgSet in enumerate(imgs):
        edges.append([])
        for img in imgSet:
            edges[i].append(findEdge(img))
            # if len(findEdge(img)) < 5:
            #     print(i)
        #     print(edges)
        #     print(len(imgs))
    pearson = np.zeros((len(edges), len(edges)))
    for i in range(len(edges)):
        for edge in edges[i]:
            edge = edge[::-1]
            for j in range(0, i):
                for edge2 in edges[j]:
                    pearson[i][j] = max(pearson[i][j], pearsonr(edge, edge2)[0] ** 2)
                pearson[j][i] = pearson[i][j]
    for i in range(len(pearson)):
        pearson[i][i] = -1
    return pearson


def compareImagesDiff(imgs):
    diff = np.zeros((len(imgs), len(imgs)), dtype=np.uint16)
    for i in range(len(imgs)):
        best = 99999999999
        for img in imgs[i]:
            img = img[:, ::-1]
            for j in range(0, i):
                for img2 in imgs[j]:
                    best = min((imgs[i] != imgs[j]).sum(), best)
                diff[i][j] = diff[j][i] = best
    return diff


def matrix2result(m):
    result = []
    for row in m:
        result.append(row.argsort()[::-1])
    return result


if __name__ == "__main__":
    directpath = sys.argv[1]
    n = int(sys.argv[2])
    if os.path.exists(directpath):
        imgs = []
        for i in range(n):
            imgs.append([])
            filename = os.path.join(directpath, str(i) + '.png')
            if os.path.exists(filename) and os.access(filename, os.R_OK):
                props, negprops = getPropositions(filename)
                for prop in props:
                    imgs[i].append(prop)
        m = compareImagesCorr(imgs)
        r = matrix2result(m)
        print(r)
        for line in r:
            print(' '.join(str(x) for x in line))
