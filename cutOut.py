import cv2
import numpy as np
import matplotlib.pyplot as plt
import base64

print('start')
#[[User specified parameters]]

# ------- color condition ----------
""" R_low = 130     #0 or 130
R_high = 300    #100 or 300
G_low = 0       #0 or 100
G_high = 100    #100 or 300
B_low = 0       #0 or 100
B_high = 100    #100 or 300 """
# -------------------------------------------------------------------------------------------------

# ------- scale condition -------------------------------------------------------------------------
Area_th_min = 1200
Area_th_max = 10000
# ------- filter condition ------------------------------------------------------------------------
ksize = 3
neiborhood = np.array([[0,1,0],[1,1,1],[0,1,0]],np.uint8)
# ------- function --------------------------------------------------------------------------------
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
# -------------------------------------------------------------------------------------------------

# Step 1 ------------------------------------------------------------------------------------------
beforeRect = [0,0,0,0]
listR = []
listG = []
listB = []
image_dir = './assets/'
filename = "input5.png"

input_img = image_dir + filename
img = cv2.imread(input_img)
img_c_origin = cv2.imread(input_img)
height = img.shape[0]
width = img.shape[1]
#img_c = cv2.resize(img , (int(width*0.5), int(height*0.5)))
""" for i in range(2):
    img_c = cv2.GaussianBlur(img_c,(5,5),0)
"""
for num in range(4):
    img_c = img
    B, G, R = cv2.split(img_c)
    if(num == 0):
        R_low = 0
        R_high = 100
        G_low = 0
        G_high = 100
        B_low = 0
        B_high = 100
        color = 'Bl'
    
    if(num == 1):
        R_low = 130
        R_high = 300
        G_low = 0
        G_high = 100
        B_low = 0
        B_high = 100
        color = 'R'
    
    if(num == 2):
        R_low = 0
        R_high = 100
        G_low = 100
        G_high = 300
        B_low = 0
        B_high = 100
        color = 'G'
    
    if(num == 3):
        R_low = 0
        R_high = 100
        G_low = 0
        G_high = 100
        B_low = 100
        B_high = 300
        color = 'B'

    img_g_th = np.where((G < G_high) & (G > G_low), 1, 0)
    img_b_th = np.where((B < B_high) & (B > B_low), 1, 0)
    img_r_th = np.where((R < R_high) & (R > R_low), 1, 0)

    img_th = img_r_th * img_g_th * img_b_th * 255
    img_th = np.uint8(img_th)
    #img_th = cv2.erode(img_th,neiborhood,iterations=1)
    img_th = cv2.dilate(img_th,neiborhood,iterations=7)
    #img_th = cv2.ximgproc.thinning(img_th,thinningType = cv2.ximgproc.THINNING_GUOHALL)
    #img_th = cv2.dilate(img_th,neiborhood,iterations=1)
    if(num == 0):
        img_th_Bl = img_th
    
    # Step 2 ----------------------------------------------------------------------------------------------
    contours = cv2.findContours(img_th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]

    # Step 3 ----------------------------------------------------------------------------------------------
    Active_contours = []

    for cont in contours:
        if cv2.contourArea(cont) > Area_th_min and cv2.contourArea(cont) < Area_th_max:
            Active_contours.append(cont)

    # Step 4 ---------------------------------------------------------------------------------------------
    cont_img = cv2.drawContours(img_c_origin, Active_contours, -1, (255,0,0), 3)

    cont_img = cv2.cvtColor(cont_img, cv2.COLOR_BGR2RGB)
    img_c_origin = cv2.cvtColor(img_c_origin, cv2.COLOR_BGR2RGB)

    # Step 5 ---------------------------------------------------------------------------------------------
    contours = cv2.findContours(img_th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    th_area = img_c.shape[0] * img_c.shape[1] / 100
    contours_large = list(filter(lambda c:cv2.contourArea(c) > th_area, contours))
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
        img_th = cv2.rectangle(img_th,(rect[2]-100,rect[1]+100),(rect[3]+100,rect[0]-100),(0,0,0),-1)
        img_th_Bl = cv2.rectangle(img_th_Bl,(rect[2],rect[1]),(rect[3],rect[0]),(0,0,0),-1)
        if((abs(beforeRect[0]-rect[0])<200)and(abs(beforeRect[1]-rect[1])<200)and(abs(beforeRect[2]-rect[2])<200)and(abs(beforeRect[3]-rect[3])<200)):
            continue

        img_output = getPartImageByRect(rect)
        img_name = './output/'+str(color)+str(i)+'.png'
        cv2.imwrite(img_name,img_output)
        target_file = r""+str(img_name)+""
        if(num == 1):
            with open (target_file,'rb') as f:
                data = f.read()
                binarySquareR = base64.b64encode(data)
            listR.append(binarySquareR)
        
        if(num == 2):
            with open (target_file,'rb') as f:
                data = f.read()
                binarySquareG = base64.b64encode(data)
            listG.append(binarySquareG)

        if(num == 3):
            with open (target_file,'rb') as f:
                data = f.read()
                binarySquareB = base64.b64encode(data)
            listB.append(binarySquareB)
        beforeRect = rect
        print(rect)

    cv2.imwrite('./output/'+str(color)+'output.png',img_th)
    if(num == 1):
        target_file = r"./output/Routput.png"
        with open(target_file, 'rb') as f:
            data = f.read()
        binaryR = base64.b64encode(data)
        reddict = {"textR":binaryR, "squareR":listR}
    if(num == 2):
        target_file = r"./output/Goutput.png"
        with open(target_file, 'rb') as f:
            data = f.read()
        binaryG = base64.b64encode(data)
        greendict = {"textG":binaryG, "squareG":listG}
    if(num == 3):
        target_file = r"./output/Boutput.png"
        with open(target_file, 'rb') as f:
            data = f.read()
        binaryB = base64.b64encode(data)
        bluedict = {"textB":binaryB, "squareB":listB}
    
    
cv2.imwrite('./output/Bloutput.png',img_th_Bl)
target_file = r"./output/Bloutput.png"
with open(target_file, 'rb') as f:
    data = f.read()
binaryBl = base64.b64encode(data)
Blackdict = {"textBl":binaryBl}
    
""" 
encode_file=r"encodeR.txt"
with open(encode_file,"wb") as f:
    f.write(binaryR)

encode_file=r"encodeG.txt"
with open(encode_file,"wb") as f:
    f.write(binaryG)

encode_file=r"encodeB.txt"
with open(encode_file,"wb") as f:
    f.write(binaryB)
"""
imgdict = {"Black":Blackdict,"red":reddict, "green":greendict, "blue":bluedict}
#print(imgdict)

print('end')

# ------------- show images ---------------------------------------------------------------------------
plt.gray()

plt.subplot(1,2,1)
plt.imshow(img_th_Bl, vmin=0, vmax=255, interpolation = 'none')
cv2.imwrite('./output/blue_output.png', img_th)
plt.title('Threshold')

plt.subplot(1,2,2)
plt.imshow(cont_img, interpolation = 'none')
plt.title('Contour')

plt.show()
# ------------------------------------------------------------------------------------------------------