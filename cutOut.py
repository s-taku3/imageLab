import cv2
import numpy as np
import matplotlib.pyplot as plt

#[[User specified parameters]]

# ------- color condition ----------
R_low = 0
R_high = 100
G_low = 0
G_high = 100
B_low = 100
B_high = 300
# -----------------------------------

# ------- scale condition -----------
Area_th_min = 1200
Area_th_max = 10000
# ------- filter condition ----------
ksize = 3
neiborhood = np.array([[0,1,0],[1,1,1],[0,1,0]],np.uint8)
# ------- function ------------------
def getRectByPoints(points):
    points = list(map(lambda x: x[0], points))

    points = sorted(points,key=lambda x:x[1])
    top_points = sorted(points[:2], key=lambda x:x[0])
    bottom_points = sorted(points[2:4], key=lambda x:x[0])
    points = top_points + bottom_points

    left = min(points[0][0], points[2][0])
    right = max(points[1][0], points[3][0])
    top = min(points[0][1], points[1][1])
    bottom = max(points[2][1], points[3][1])
    return (top, bottom, left, right)

def getPartImageByRect(rect):
    im = img_c
    return im[rect[0]:rect[1], rect[2]:rect[3]]
# ----------------------------------

# Step 1 ---------------------------
image_dir = './assets/'
filename = "input.png"
input_img = image_dir + filename
img = cv2.imread(input_img)
img_c_origin = cv2.imread(input_img)
height = img.shape[0]
width = img.shape[1]
print(height,width)
#img_c = cv2.resize(img , (int(width*0.5), int(height*0.5)))
""" for i in range(2):
    img_c = cv2.GaussianBlur(img_c,(5,5),0)
 """
img_c = img
B, G, R = cv2.split(img_c)

img_g_th = np.where((G < G_high) & (G > G_low), 1, 0)
img_b_th = np.where((B < B_high) & (B > B_low), 1, 0)
img_r_th = np.where((R < R_high) & (R > R_low), 1, 0)

img_th = img_r_th * img_g_th * img_b_th * 255
img_th = np.uint8(img_th)
#img_th = cv2.erode(img_th,neiborhood,iterations=1)
img_th = cv2.dilate(img_th,neiborhood,iterations=7)
img_th = cv2.ximgproc.thinning(img_th,thinningType = cv2.ximgproc.THINNING_GUOHALL)
img_th = cv2.dilate(img_th,neiborhood,iterations=1)

# Step 2 ---------------------------
contours = cv2.findContours(img_th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]

# Step 3 ---------------------------
Active_contours = []

for cont in contours:
    if cv2.contourArea(cont) > Area_th_min and cv2.contourArea(cont) < Area_th_max:
        Active_contours.append(cont)

# Step 4 ---------------------------
cont_img = cv2.drawContours(img_c_origin, Active_contours, -1, (255,0,0), 3)

cont_img = cv2.cvtColor(cont_img, cv2.COLOR_BGR2RGB)
img_c_origin = cv2.cvtColor(img_c_origin, cv2.COLOR_BGR2RGB)

# Step 5 ---------------------------
contours = cv2.findContours(img_th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
th_area = img_c.shape[0] * img_c.shape[1] / 100
contours_large = list(filter(lambda c:cv2.contourArea(c) > th_area, contours))
print(contours_large)
outputs = []
rects = []
approxes = []

for (i,cnt) in enumerate(contours_large):
    arclen = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02*arclen, True)
    if len(approx) < 4:
        continue

    approxes.append(approx)
    rect = getRectByPoints(approx)
    rects.append(rect)
    outputs.append(getPartImageByRect(rect))
    cv2.imwrite(str(i)+'.png',getPartImageByRect(rect))
    img_th = cv2.rectangle(img_th,(rect[2]-100,rect[1]+100),(rect[3]+100,rect[0]-100),(0,0,0),-1)
    print(rect)

# ------------- show images ------------- 
plt.gray()

plt.subplot(1,2,1)
plt.imshow(img_th, vmin=0, vmax=255, interpolation = 'none')
cv2.imwrite('./output/blue_output.png', img_th)
plt.title('Threshold')

plt.subplot(1,2,2)
plt.imshow(cont_img, interpolation = 'none')
plt.title('Contour')

plt.show()
# ----------------------------------------