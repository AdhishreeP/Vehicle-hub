import time
import imutils
import pytesseract
import PIL
from os.path import isfile, join
from PIL import Image
import glob
from scipy.misc import *
import pymysql
from sqlalchemy import create_engine
from tkinter.filedialog import askopenfilename
from flask import Flask, render_template, request, url_for,Response
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import create_engine
import pymysql
from flask_mail import Mail
import json
from werkzeug.utils import secure_filename
with open('C:\\Users\\HP\\PycharmProjects\\final 3yr\\final 3yr\\config.json', 'r') as c:
    params = json.load(c)["params"]
import cv2
import numpy as np
from time import sleep


local_server = True
app = Flask(__name__,template_folder="templates")
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config.update(
    MAIL_SERVER = 'smtp.gmail.com',
    MAIL_PORT = '465',
    MAIL_USE_SSL = True,
    MAIL_USERNAME = params['gmail-user'],
    MAIL_PASSWORD=  params['gmail-password']
)
mail = Mail(app)
if(local_server):
    app.config['SQLALCHEMY_DATABASE_URI'] = params['local_uri']
else:
    app.config['SQLALCHEMY_DATABASE_URI'] = params['prod_uri']


db = SQLAlchemy(app)


class contact(db.Model):

    '''
    sno, name,  email,msg
    '''
    sr= db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), nullable=False)
    email = db.Column(db.String(80), nullable=False)
    subject= db.Column(db.String(120), nullable=False)
    message = db.Column(db.String(120), nullable=False)

@app.route("/")
def home():

    return render_template("index.html",params=params)


@app.route("/index", methods = ['GET', 'POST'])
def contacts():
    if(request.method=='POST'):
        '''Add entry to the database'''
        name = request.form.get('name')
        email = request.form.get('email')
        message = request.form.get('message')
        subject= request.form.get('subject')
        entry =contact(name=name,email = email, message = message, subject=subject)
        db.session.add(entry)
        db.session.commit()

        mail.send_message('New message from ' + name,
                          sender=email,
                          recipients=[params['gmail-user']],
                          body=message
                          )
        return "Message sent"


@app.route("/interview")
def home2():
    return render_template('1.html',params=params)
@app.route("/")

def quiz():
    return render_template('2.html',params=params)
@app.route("/new")

def home3():
    return render_template('3.html',params=params)

#@app.route("/count")
#def home4():
 #   return render_template('count.html')

@app.route("/reg")
def home5():
    return render_template('reg.html')

@app.route("/reg1")
def home6():
    return render_template('reg1.html')

@app.route("/vi")
def home7():
    return render_template('vi.html')

@app.route("/dl")
def home8():
    return render_template('dl.html')

@app.route("/dl1")
def home9():
    return render_template('dl1.html')

@app.route("/dl2")
def home10():
    return render_template('dl2.html')

@app.route("/np")
def home11():
    return render_template('np.html')

@app.route("/np1")
def home12():
    return render_template('np1.html')

@app.route("/np2")
def home13():
    return render_template('np2.html')

@app.route("/np3")
def home14():
    return render_template('np3.html')

@app.route("/np4")
def home15():
    return render_template('np4.html')

@app.route("/fn")
def home16():
    return render_template('fn.html')



@app.route("/info")
def home17():
    return render_template('info.html')

@app.route("/ms")
def home18():
    return render_template('ms.html')

@app.route("/cs")
def home19():
    return render_template('cs.html')

@app.route("/is")
def home20():
    return render_template('is.html')


@app.route("/fts")
def home21():
    return render_template('fts.html')

@app.route("/its")
def home22():
    return render_template('its.html')

@app.route("/count")
def home100():
    return render_template("count.html")

@app.route("/Upload",methods=['POST','GET'])
def home101():
    import os
    if request.method == 'POST':
        x = request.files['filename']
        global myfile
        myfile = secure_filename(x.filename)
        global path
        path=os.path.abspath(myfile)
        print("abs path :",path)
        print("Path : ", myfile)
        return render_template("Upload_res.html")
    return render_template("Upload.html")

@app.route("/Upload_res",methods=['POST','GET'])
def home23():
    return render_template('Upload_res.html')

def Counting():
    largura_min = 80
    altura_min = 80
    offset = 6
    pos_linha = 550

    delay = 60  # FPS do vídeo
    detec = []
    carros = 0

    def pega_centro(x, y, w, h):
        x1 = int(w / 2)
        y1 = int(h / 2)
        cx = x + x1
        cy = y + y1
        return cx, cy

    camera = cv2.VideoCapture(path)
    subtracao = cv2.bgsegm.createBackgroundSubtractorMOG()

    while (camera.isOpened()):

        ret, img = camera.read()
        if ret == True:

            tempo = float(1 / delay)
            sleep(tempo)
            grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # coverts image to gray
            blur = cv2.GaussianBlur(grey, (3, 3), 5)  # smoothing the image
            img_sub = subtracao.apply(blur)  # apply the subtraction to frames
            dilat = cv2.dilate(img_sub, np.ones((5, 5)))  # dilation of imge is performmed
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE,
                                        kernel)  # used to close small holes of foregraound frame
            dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)
            contorno, h = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            cv2.line(img, (25, pos_linha), (1200, pos_linha), (255, 127, 0), 3)
            for (i, c) in enumerate(contorno):
                (x, y, w, h) = cv2.boundingRect(c)  # used to bound rectangle
                validar_contorno = (w >= largura_min) and (h >= altura_min)
                if not validar_contorno:
                    continue
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                centro = pega_centro(x, y, w, h)
                detec.append(centro)
                cv2.circle(img, centro, 4, (0, 0, 255), -1)

                for (x, y) in detec:
                    if y < (pos_linha + offset) and y > (pos_linha - offset):
                        carros += 1
                        cv2.line(img, (0, 500), (2000, pos_linha), (0, 127, 255), 3)
                        detec.remove((x, y))

            cv2.putText(img, "VEHICLE COUNT : " + str(carros), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),
                        5)
            #cv2.imshow("Video Original", img)

            if cv2.waitKey(10) == 7:
                break

            img = cv2.resize(img, (0, 0), fx=0.8, fy=0.8)
        frame = cv2.imencode('.jpg', img)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.1)


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(Counting(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')



#Owner Information
@app.route("/Upload1")
def home24():
    return render_template('Upload1.html')

@app.route("/Owner_info",methods = ['POST','GET'])
def Owner_info():

    if (request.method == 'POST'):
        x = request.files['filename']
        myfile = secure_filename(x.filename)
        print("Path : ",myfile)

        pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

        connection = pymysql.connect(host="localhost", user="root", passwd="", database="login")
        cursor = connection.cursor()

        img = cv2.imread(myfile, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (600, 400))

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 13, 15, 15)

        edged = cv2.Canny(gray, 30, 200)
        contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        screenCnt = None

        for c in contours:

            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.018 * peri, True)

            if len(approx) == 4:
                screenCnt = approx
                break

        if screenCnt is None:
            detected = 0
            print("No contour detected")
        else:
            detected = 1

        if detected == 1:
            cv2.drawContours(img, [screenCnt], -1, (0, 0, 255), 3)

        mask = np.zeros(gray.shape, np.uint8)
        new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1, )
        new_image = cv2.bitwise_and(img, img, mask=mask)

        (x, y) = np.where(mask == 255)
        (topx, topy) = (np.min(x), np.min(y))
        (bottomx, bottomy) = (np.max(x), np.max(y))
        Cropped = gray[topx:bottomx + 1, topy:bottomy + 1]

        text = pytesseract.image_to_string(Cropped, config='--psm 6')
        text = text.replace(" ", "")
        # print(text)
        text1 = ""
        for i in text:
            if i.isalnum():
                text1 = text1 + i
        # print(text1)

        retrive = "Select * from owner_info;"

        # executing the quires
        cursor.execute(retrive)
        rows = cursor.fetchall()

        for row in rows:
            # print(row)
            # print(row[0])
            if row[0] == text1:
                global ls
                ls = [row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7],row[8]]
                print(ls)
        '''
        img = cv2.resize(img, (500, 300))
        Cropped = cv2.resize(Cropped, (400, 200))
        cv2.imshow('car', img)
        cv2.imshow('Cropped', Cropped)
        '''
        connection.commit()
        connection.close()

        return render_template('Upload1 res.html',ls=ls,myfile=myfile)



#Number Plate Detection
@app.route("/Upload2")
def home25():
    return render_template('Upload2.html')

@app.route("/NumberPlate_detect",methods = ['POST','GET'])

def NumberPlate_detect():
    if (request.method == 'POST'):
        x = request.files['filename']
        myfile = secure_filename(x.filename)
        print("Path : ",myfile)
        pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

        img = cv2.imread(myfile, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (600, 400))

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to grey scale
        gray = cv2.bilateralFilter(gray, 13, 15,
                                   15)  # bilateral filter (Blurring) will remove the unwanted details from an image

        edged = cv2.Canny(gray, 30, 200)  ##Perform Edge detection
        contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        screenCnt = None

        for c in contours:

            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.018 * peri, True)

            if len(approx) == 4:
                screenCnt = approx
                break

        if screenCnt is None:
            detected = 0
            print("No contour detected")
        else:
            detected = 1

        if detected == 1:
            cv2.drawContours(img, [screenCnt], -1, (0, 0, 255), 3)

        # Masking the part other than the number plate
        mask = np.zeros(gray.shape, np.uint8)
        new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1, )
        new_image = cv2.bitwise_and(img, img, mask=mask)

        (x, y) = np.where(mask == 255)  # crop
        (topx, topy) = (np.min(x), np.min(y))
        (bottomx, bottomy) = (np.max(x), np.max(y))
        Cropped = gray[topx:bottomx + 1, topy:bottomy + 1]

        text = pytesseract.image_to_string(Cropped, config='--psm 6')  # Read the number plate
        Plate_number = ""
        for i in text:
            if i.isalnum():
                Plate_number = Plate_number + i

        #print(" License Plate Recognition : \n")
        print("Detected license plate Number is:", Plate_number)
        img = cv2.resize(img, (500, 300))
        Cropped = cv2.resize(Cropped, (400, 200))
        array = np.resize(Cropped, (200, 400))
        data = Image.fromarray(array)
        data.save('static/assets/img/1.png')
        im1 = Image.open(myfile)

        myfile = im1.save('static/assets/img/car.png')

        return render_template("Upload2 res.html",number=Plate_number)



#vehicle Classification
@app.route("/Upload3",methods=['POST','GET'])
def home26():
    import os
    if request.method == 'POST':
        x = request.files['filename']
        global myfile1
        myfile1 = secure_filename(x.filename)
        global path1
        path1 =os.path.abspath(myfile1)
        #print(os.path.join(path1,myfile1))
        print("Path : ", myfile1)
        print("Abs path : ",path1)
        return render_template("Upload3_res.html")
    return render_template("Upload3.html")

@app.route("/Upload3_res",methods=['POST','GET'])
def home27():
    return render_template('Upload3_res.html')

def Classification():
    import vehicles
    #camera = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(path1)
    fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False, history=200, varThreshold=90)
    kernalOp = np.ones((3, 3), np.uint8)
    kernalOp2 = np.ones((5, 5), np.uint8)
    kernalCl = np.ones((11, 11), np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cars = []
    max_p_age = 5
    pid = 1
    cnt_up = 0
    cnt_down = 0
    line_up = 400
    line_down = 250

    up_limit = 230
    down_limit = int(4.5 * (500 / 5))

    while (cap.isOpened()):
        ret, frame = cap.read()
        frame = cv2.resize(frame, (900, 500))
        for i in cars:
            i.age_one()
        fgmask = fgbg.apply(frame)
        if ret==True:
            ret,imBin=cv2.threshold(fgmask,200,255,cv2.THRESH_BINARY)
            mask = cv2.morphologyEx(imBin, cv2.MORPH_OPEN, kernalOp)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernalCl)

            (countours0,hierarchy)=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            for cnt in countours0:
                area=cv2.contourArea(cnt)
                #print(area)
                if area>300:

                    m=cv2.moments(cnt)
                    cx=int(m['m10']/m['m00'])
                    cy=int(m['m01']/m['m00'])
                    x,y,w,h=cv2.boundingRect(cnt)


                    new=True
                    if cy in range(up_limit,down_limit):
                        for i in cars:
                            if abs(x - i.getX()) <= w and abs(y - i.getY()) <= h:
                                new = False
                                i.updateCoords(cx, cy)

                                if i.going_UP(line_down,line_up)==True:
                                    cnt_up+=1

                                elif i.going_DOWN(line_down,line_up)==True:
                                    cnt_down+=1

                                break
                            if i.getState()=='1':
                                if i.getDir()=='down'and i.getY()>down_limit:
                                    i.setDone()
                                elif i.getDir()=='up'and i.getY()<up_limit:
                                    i.setDone()
                            if i.timedOut():
                                index=cars.index(i)
                                cars.pop(index)
                                del i

                        if new==True:
                            p=vehicles.Car(pid,cx,cy,max_p_age)
                            cars.append(p)
                            pid+1
                    cv2.circle(frame, (cx, cy), 2, (0, 0, 255), -1)

                    img=cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

            for i in cars:
                cv2.putText(frame, str(i.getId()), (i.getX(), i.getY()), font, 0.3, (255,255,0), 1, cv2.LINE_AA)
                if line_down+20<= i.getY() <= line_up-20:
                   a = (h + (.74*w)- 100)

                   if a > 0:
                         cv2.putText(frame, "Truck", (i.getX(), i.getY()), font, 1, (0,0,255), 2, cv2.LINE_AA)
                   else:
                       cv2.putText(frame, "Car", (i.getX(), i.getY()), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

            frame = cv2.line(frame, (0, line_up), (900, line_up), (0, 0, 255), 3, 8)
            frame = cv2.line(frame, (0, up_limit), (900, up_limit), (0, 0, 0), 1, 8)

            frame = cv2.line(frame, (0, down_limit), (900, down_limit), (255, 255, 0), 1, 8)
            frame = cv2.line(frame, (0, line_down), (900, line_down), (255, 0, 0), 3, 8)

            #cv2.putText(frame, str_up, (10, 40), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            #cv2.putText(frame, str_down, (10, 90), font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
            #cv2.imshow('Frame', frame)

            if cv2.waitKey(100) & 0xff == ord('q'):
                break
            frame = cv2.resize(frame, (0, 0), fx=1.2, fy=1.2)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



@app.route('/video_feed1')
def video_feed1():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(Classification(),
        mimetype='multipart/x-mixed-replace; boundary=frame')


app.run(debug=True)
