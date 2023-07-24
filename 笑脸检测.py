import cv2
import numpy as np
import tkinter
import tkinter.filedialog
from PIL import Image,ImageTk


#创建窗口
root = tkinter.Tk()
root.geometry("1600x500")
root.title("JiafengCV   1.0")


#创建标签
label0 = tkinter.Label(root,text="截图的结果如下 ")
label0.place(x=1,y=1)

label1 = tkinter.Label(root,text=" ")
label2 = tkinter.Label(root,text=" ")
label3 = tkinter.Label(root,text=" ")
label4 = tkinter.Label(root,text=" ")
label5 = tkinter.Label(root,text=" ")
label6 = tkinter.Label(root,text=" ")
label7 = tkinter.Label(root,text=" ")
label8 = tkinter.Label(root,text=" ")
label9 = tkinter.Label(root,text=" ")
label10 = tkinter.Label(root,text=" ")

label0 = tkinter.Label(root,text="输入想要保存的图片序号（eg：1） ")
label0.place(x=1,y=300)


#创建输入框
entry = tkinter.Entry(root, width=10, bd=1)
entry.place(x=1,y=330)


#定义函数
def fun_savef():
    fway = entry.get()
    favor = cv2.imread("smile" + fway + ".jpg")
    cv2.imwrite("favorite.jpg",favor)
    tkinter.messagebox.showinfo(title="Good", message="保存成功!")

def attention():
    tkinter.messagebox.showwarning(title="Attention", message="在摄像头面前无比转动脑袋截图\n不要保持一个姿势!")


#创建按钮
btn = tkinter.Button(root,text="保存该图片",width=7,height=2,command=fun_savef)
btn.place(x=100,y=330)
btn = tkinter.Button(root,text="关于",width=7,height=2,command=attention)
btn.place(x=200,y=330)



# 初始化摄像头
cap = cv2.VideoCapture(0)

#加载人脸识别模型
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#加载笑容识别模型
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

#训练笑容模型
# 用于训练的笑容样本数量
sample_count = 0
# 存储样本
samples = []
# 存储标签
labels = []

j = 1
n = 1
m = 40
#摄像头截图并且预测
while (cap.isOpened()):
    ret, fname = cap.read()
    gray = cv2.cvtColor(fname, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # 对人脸进行笑容检测
    for (x, y, w, h) in faces:
        gray = gray[y:y + h, x:x + w]
        color = fname[y:y + h, x:x + w]
        smiles = smile_cascade.detectMultiScale(gray, 1.8, 20)

        # 检测到笑容就标记出来
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(color, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)
            cv2.putText(fname, 'Smile', (x, y - 6), 3, 1.2, (0, 0, 255), 2, cv2.LINE_AA)
        # 将笑容区域添加到训练样本中
        samples.append(gray)
        labels.append(1)
        sample_count += 1
    cv2.imshow('fname', fname)
    c = cv2.waitKey(1)

    #按"t"截图，由于布局设置，最好只截图7张，后面的太长了显示不了
    if c == ord('t'):
        cv2.imwrite("smile" + str(j) + ".jpg", fname)
        #显示图片
        ver_name = "label" + str(j)
        ever = eval(ver_name)
        ever = tkinter.Label(root, text=" ")
        ever.place(x=n, y=m)
        n += 200

        gmi = cv2.imread("smile" + str(j) + ".jpg")
        gmi = cv2.cvtColor(gmi, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(np.uint8(gmi))
        temp = image.resize((200, 200))
        b = ImageTk.PhotoImage(temp)
        ever.config(image=b)
        ever.image = b
        j = j + 1
    if c == ord('e'):
        break

cap.release()
cv2.destroyAllWindows()

# 训练笑容识别模型
model = cv2.face.LBPHFaceRecognizer_create()
model.train(samples, np.array(labels))

# 保存模型
model.save('smiledetection_model.yml')

root.mainloop()



