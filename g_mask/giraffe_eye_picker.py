import cv2
import json

# 이미지 불러오기
img = cv2.imread('giraffe.jpg')
points = []

# 마우스 콜백 함수
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(img, f"{x},{y}", (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        cv2.imshow("Select Eye Points", img)

# 창 생성 및 콜백 등록
cv2.imshow("Select Eye Points", img)
cv2.setMouseCallback("Select Eye Points", click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 좌표 저장
with open("giraffe_eye_points.json", "w") as f:
    json.dump(points, f)
print("좌표 저장 완료:", points)
