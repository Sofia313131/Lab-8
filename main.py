import time
import cv2

def image_processing():
    img = cv2.imread('images/variant-10.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    cv2.imshow('threshold', thresh)


EPSILON = 1000
MAX_DELTA = 100000
VALUES_K = 0.4


def video_processing():
    cap = cv2.VideoCapture(0)
    down_points = (640, 480)  # Размер кадра
    last_center = (320, 240)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, down_points, interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        ret, thresh = cv2.threshold(gray, 115, 255, cv2.THRESH_BINARY_INV)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_pair = None
        best_dist_last = 10**10
        for cn1, c1 in enumerate(contours):
            for cn2, c2 in enumerate(contours):
                if cn1 == cn2:
                    continue
                x1, y1, w1, h1 = cv2.boundingRect(c1)
                x2, y2, w2, h2 = cv2.boundingRect(c2)
                dists = [
                    (xx1 - xx2) ** 2 + (yy1 - yy2) ** 2
                    for xx1, yy1 in ((x1, y1), (x1 + w1, y1), (x1, y1 + h1), (x1 + w1, y1 + h1))
                    for xx2, yy2 in ((x2, y2), (x2 + w2, y2), (x2, y2 + h2), (x2 + w2, y2 + h2))
                ]
                areas = [cv2.contourArea(c1), cv2.contourArea(c2)]
                if min(areas) <= 10:
                    continue
                values_cff = abs(areas[0] - areas[1]) / max(areas)
                dist_last = ((x1 + x2 + w1 + w2) // 2 - last_center[0])**2 + \
                            ((y1 + y2 + h1 + h2) // 2 - last_center[1])**2
                if min(dists) < EPSILON and values_cff < VALUES_K and dist_last < MAX_DELTA:
                    if dist_last < best_dist_last:
                        best_dist_last = dist_last
                        best_pair = [c1, c2]
        # Last center
        cv2.circle(frame, last_center, 5, (0, 0, 120), -1)
        if best_pair is not None:
            # Ref-point parts
            x1, y1, w1, h1 = cv2.boundingRect(best_pair[0])
            x2, y2, w2, h2 = cv2.boundingRect(best_pair[1])
            cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)
            cv2.rectangle(frame, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0), 2)
            # New center
            a = ((x1 + x2 + w1//2 + w2//2) + last_center[0]) // 3
            b = ((y1 + y2 + h1//2 + h2//2) + last_center[1]) // 3
            cv2.circle(frame, (a, b), 5, (0, 0, 255), -1)
            last_center = (a, b)
            if 245 <= a <= 395 and 165 <= b <= 315:
                frame = cv2.flip(frame, 1)
        cv2.rectangle(frame, (245, 165), (395, 315), (255, 0, 0), 2)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(0.1)
    cap.release()


if __name__ == '__main__':
    # image_processing()
    video_processing()

cv2.waitKey(0)
cv2.destroyAllWindows()