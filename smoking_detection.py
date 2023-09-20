import torch

import cv2
from time import time
from ultralytics import YOLO

from supervision.draw.color import ColorPalette
from supervision import Detections
from supervision import BoxAnnotator

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import smtplib


class ObjectDetection:

    def smoke_image_mail_sender(self):

        strFrom = 'ac19ucs133.varun@gmail.com'
        strTo = 'ac19ucs133.varun@gmail.com'

        # Create the root message and fill in the from, to, and subject headers
        msgRoot = MIMEMultipart('related')
        msgRoot['Subject'] = 'public place smoking person detect.....'
        msgRoot['From'] = strFrom
        msgRoot['To'] = strTo
        msgRoot.preamble = 'This is a multi-part message in MIME format.'

        msgAlternative = MIMEMultipart('alternative')

        msgRoot.attach(msgAlternative)

        mail_message_Text = MIMEText('smoking person detected in public place....')

        msgAlternative.attach(mail_message_Text)

        sending_image = open('smoke.jpg', 'rb')

        msgImage = MIMEImage(sending_image.read())

        sending_image.close()

        # Define the image's ID as referenced above
        msgImage.add_header('Content-ID', '<image1>')

        msgRoot.attach(msgImage)

        smtp = smtplib.SMTP('smtp.gmail.com', 587)

        smtp.starttls()

        smtp.login('vinayacseproject@gmail.com', 'pernbugccbqxknhc')

        print("mail id and password correct")

        smtp.sendmail(strFrom, strTo, msgRoot.as_string())

        print("mail send")

        smtp.quit()

    def __init__(self):

        self.capture_index = 0

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # print("Using Device: ", self.device)

        self.model = self.load_model()

        self.CLASS_NAMES_DICT = self.model.model.names

        self.box_annotator = BoxAnnotator(color=ColorPalette.default(), thickness=2, text_thickness=1, text_scale=1.5)

    def load_model(self):

        model = YOLO("detection_module.pt")
        model.fuse()
        return model

    def predict(self, frame):
        results = self.model(frame)
        return results

    def plot_bboxes(self, results, frame):

        xyxys = []
        confidences = []
        class_ids = []

        # print("empty confidence=",confidences)

        # Extract detections for person class
        for result in results[0]:
            # print("data confidence=", confidences)
            class_id = result.boxes.cls.cpu().numpy().astype(int)

            if result == 0:
                print("smoking not detected")

            else:
                print("smoking detected")
                cv2.imwrite('smoke.jpg', frame)
                print("email sent along with image capture........")

                self.smoke_image_mail_sender()

            if class_id == 0:
                xyxys.append(result.boxes.xyxy.cpu().numpy())
                confidences.append(result.boxes.conf.cpu().numpy().astype(int))
                class_ids.append(result.boxes.cls.cpu().numpy().astype(int))
                # print("x confidence=",confidences)

        # Setup detections for visualization
        detections = Detections(
            xyxy=results[0].boxes.xyxy.cpu().numpy(),
            confidence=results[0].boxes.conf.cpu().numpy(),
            class_id=results[0].boxes.cls.cpu().numpy().astype(int),
        )

        # Format custom labels
        self.labels = [f"{self.CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
                       for _, confidence, class_id, tracker_id in detections]
        # print("confidence",confidence)

        # if [confidences] > 0.85:
        #     print("smoking detection is confirmed")
        # else:
        # print("smoking detection is not confirmed")

        # Annotate and display frame
        frame = self.box_annotator.annotate(scene=frame, detections=detections, labels=self.labels)

        return frame

    def __call__(self):

        cap = cv2.VideoCapture(0)
        assert cap.isOpened()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        while True:
            ret, frame = cap.read()
            assert ret
            results = self.predict(frame)
            frame = self.plot_bboxes(results, frame)
            cv2.imshow('Smoking Detection', frame)
            if cv2.waitKey(5) & 0xFF == 27:
                break
        cap.release()
        cv2.destroyAllWindows()


detector = ObjectDetection()
detector()
