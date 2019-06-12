import cv2
import numpy as np

def load_image(path_name):
  image = cv2.imread(path_name)
  return image

def apply_seeded_segmentation(image, seed_x, seed_y, threshold):
  dx = [1, 1, 0, -1, -1, -1, 0, 1]
  dy = [0, 1, 1, 1, 0, -1, -1, -1]
  queue = [(seed_x, seed_y)]
  count = 1
  running_avg = image[seed_y, seed_x] / count
  visited = np.zeros((image.shape[0], image.shape[1]))
  visited[seed_y, seed_x] = 1
  segmented_image = np.zeros(image.shape)
  while(len(queue))>0:
    current_point_x, current_point_y = queue.pop(0)
    segmented_image[current_point_y, current_point_x] = np.array([255, 255, 255])
    for i in range(8):
      new_x = current_point_x + dx[i]
      new_y = current_point_y + dy[i]
      if new_x>=0 and new_x<image.shape[1] and new_y>=0 and new_y<image.shape[0]:
        if visited[new_y, new_x]!=1:
          intensity_at_point = image[new_y][new_x]
          if np.all(np.abs(running_avg - intensity_at_point)<threshold):
            visited[new_y, new_x] = 1
            queue.append((new_x, new_y))
            count+=1
            running_avg = (running_avg * (count-1) + intensity_at_point) / count

  return segmented_image

image = load_image('Q3-faces/face3.jpg')
segmented_image = apply_seeded_segmentation(image, 290, 220, 77)
cv2.circle(segmented_image, (290, 220), 7, (0,0,255), -1)
cv2.imwrite('hi.jpg', segmented_image)

