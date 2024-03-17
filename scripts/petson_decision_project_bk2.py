#!/usr/bin/env python
from __future__ import print_function

import sys, select, termios, tty, multiprocessing, random, time, os, collections, requests
import webrtcvad, pyaudio, wave
import rospy, roslib, tf, actionlib
import cv2
from multiprocessing import Process, Manager
from move_base_msgs.msg import *
from std_msgs.msg import String, Int32MultiArray, MultiArrayDimension, Float32MultiArray
from actionlib_msgs.msg import GoalID
from geometry_msgs.msg import Twist, PointStamped, PoseStamped, PoseWithCovarianceStamped
from sensor_msgs.msg import CompressedImage, LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge, CvBridgeError
from dynamic_reconfigure.server import Server
from jetbot_pro.cfg import laserAvoidanceConfig
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
#from laser_avoidance_orin import LaserFilter
from array import array
from struct import pack
from subprocess import Popen, PIPE

from gtts import gTTS
import requests.packages.urllib3
requests.packages.urllib3.disable_warnings()

FilePath = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/data" #../data


msg = """
1. Reading from the keyboard and Publishing to Twist.
   For help, check 'teleop_key.py'. It is the same.
2. At the same time, get video stream input from JetBot to Orin.
   For help, check 'cv_video.py'. It is almost the same.
3. During movement, if obstacle found with LiDAR, avoid it.
4. With YOLO, track owner in sight, follow it.
CTRL-C to quit
"""

#YOLO_PATH='/home/orin/catkin_ws/src/best4.pt'
YOLO_PATH='/home/orin/yolov8_rs/yolov8s.pt'

model = YOLO(YOLO_PATH)
# Owner following
cur_owner_id = None
owner_offset = [0., 0.] ## Tolerance parameter, used with owner's offset rate.
owner_tolerance = 0.2   ## Initialize offset
tf_sub = None #tf.TransformListener()
# VAD/ASR/... 's global variable
recognized_direction = None

# VAD / ASR / ...
FORMAT = pyaudio.paInt16
AUDIO_CHANNELS = 1
AUDIO_RATE = 16000
CHUNK_DURATION_MS = 30                                  # supports 10,20,30 (ms)
PADDING_DURATION_MS = 1500                              # 1 sec jugement
CHUNK_SIZE = int(AUDIO_RATE * CHUNK_DURATION_MS / 1000) # chunk to read
CHUNK_BYTES = CHUNK_SIZE * 2                            # 16bit = 2bytes, PCM
NUM_PADDING_CHUNKS = int(PADDING_DURATION_MS / CHUNK_DURATION_MS)
NUM_WINDOW_CHUNKS = int(400 / CHUNK_DURATION_MS)       # 400 ms / 30ms ge
NUM_WINDOW_CHUNKS_END = NUM_WINDOW_CHUNKS * 2
START_OFFSET = int(NUM_WINDOW_CHUNKS * CHUNK_DURATION_MS * 0.5 * AUDIO_RATE)

# Scotty_detected = False #bool

# VAD/ASR/...
def record_to_file(path, data, sample_width):
    """Records from the microphone and outputs the resulting data to 'path'"""
    data = pack('<' + ('h' * len(data)), *data)
    wf = wave.open(path, 'wb')
    wf.setnchannels(AUDIO_CHANNELS)
    wf.setsampwidth(sample_width)
    wf.setframerate(AUDIO_RATE)
    wf.writeframes(data)
    wf.close()
def normalize(snd_data):
    """Average the volume out"""
    MAXIMUM = 32767 #16384
    times = float(MAXIMUM) / max(abs(i) for i in snd_data)
    r = array('h')
    for i in snd_data:
        r.append(int(i*times))
    return r
def run(cmd):
    p = Popen(cmd, stdout=PIPE, shell=True)
    p.wait()
def query_huggingface_model(model_name, prompt, api_token, wait_for_model=False):
    API_URL = f"https://api-inference.huggingface.co/models/{model_name}"
    headers = {"Authorization": f"Bearer {api_token}",
               "Content-Type": "application/json"}
    data    = {"inputs": prompt,
               "options": {
                   "wait_for_model" : wait_for_model,
                }}
    response = requests.post(API_URL, headers=headers, json=data)
    result = response.json()
    return result
# VAD/ASR/... Done

def movebase_client():
    client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
    client.wait_for_server()

    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = "map"
    goal.target_pose.header.stamp = rospy.Time.now()

    goal.target_pose.pose.position.x = 0.
    goal.target_pose.pose.position.y = 0.
    goal.target_pose.pose.orientation.w = 1.0

    client.send_goal(goal)
    wait = client.wait_for_result()
    if not wait:
        print(f"Action server is not available")
        rospy.signal_shutdown("Action server is not available")
    else:
        return client.get_result()

def convert_arr_to_ros_msg(python_float_list):
    return Float32MultiArray(data=python_float_list)

## Audio recognition
class VoiceRecognition:
    def __init__(self, *args):
        super().__init__()
        self.Mode = args[0]
        self.Path = args[1]
        self.textfile = args[2]

        self.vad = webrtcvad.Vad(2)
        self.pa = pyaudio.PyAudio()

        #self.API_TOKEN = "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"
        self.MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"

        self.HF_API_TOKEN = "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"
        self.pub_text = rospy.Publisher('voice_text', String, queue_size=1)

    def callback(self, data):
        pass

## Cv_video + YOLO
class image_converter:
    def __init__(self, *args):
        self.camera_name = rospy.get_param("~camera_name", "csi_cam_0")
        self.topic_name = rospy.get_param("~topic_name", "petson_decision")
        self.bridge = CvBridge()
        #self.pub_scotty = rospy.Publisher('chatter', String, queue_size=10)
        self.pub_owner = rospy.Publisher('owner_chat', Float32MultiArray, queue_size=2)
        self.image_sub = rospy.Subscriber(
            self.camera_name+"/image_raw/compressed",
            CompressedImage,
            self.callback
        )
        self.image_pub = rospy.Publisher(
            self.topic_name+"/compressed",
            CompressedImage,
            queue_size=2 #10
        )
        self.pub = rospy.Publisher('speak', String, queue_size=10)
        # YOLO tracking
        self.owner_id = None
        # Initial CSI Camera information
        self.camera_width = 640
        self.camera_height = 480
        self.camera_center_x = self.camera_width / 2
        self.camera_center_y = self.camera_height / 2

        # ARGS:
        #owner_tolerance, owner_offset, cur_owner_id
        #self.args = args

    def owner_track(self, results):
        #global cur_owner_id
        # Track and assign self.owner_id
        max_size = -1
        max_box = None
        owner_id = None
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # b = (Left, Top, Right, Bottom)
                b = box.xyxy[0]
                # Class
                c = box.cls
                box_id = box.id

                # Set owner_id for tracking.
                if self.owner_id is None:
                    if model.names[int(c)] == 'person':
                        size = abs(b[2]-b[0]) * abs(b[3]-b[1])
                        if max_size < size:
                            max_size = size
                            max_box = b
                            owner_id = box_id
                else:
                    if box_id == self.owner_id:
                        max_box = b
        if self.owner_id is None:
            self.owner_id = owner_id
            # NOTE: Global variable
            #with self.args[-1]:
            #    #cur_owner_id = self.owner_id
            #    self.args[2].value = self.owner_id
            cur_owner_id = self.owner_id
        return max_box
    
    def callback(self, data):
        #global owner_tolerance, owner_offset, cur_owner_id
        try:
            img = self.bridge.compressed_imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        '''
        # YOLO tracking
        Detect_scotty_list = model.track(img, persist=True, classes=[80],
              verbose=False) # 15, 16, 17,18,19,29,21,22,23,27,73
        Detect_scotty = Detect_scotty_list[0] if Detect_scotty_list else None
        '''
        track_results = model.track(img, conf=0.5, persist=True, verbose=False, classes=[0]) # 0.35
        have_results = track_results[0] if track_results else None

        if have_results:
            # Plot the image with detection
            detected_im_array = track_results[0].plot()
            #cv2.imshow("YOLO V8 Detection", detected_im_array)

            # Assign owner_id
            owner_box_detected = self.owner_track(track_results)
            # == max_box
            if owner_box_detected is not None:
                annotator = Annotator(img)
                annotator.box_label(owner_box_detected, "owner")
                owner_annotated_img = annotator.result()

                # Calculation for Owner Box center coordinate
                owner_box = owner_box_detected
                # X-axis center of owner box
                owner_center_x = (owner_box[0] + owner_box[2]) / 2
                # Y-axis center of owner box
                owner_center_y = (owner_box[1] + owner_box[3]) / 2
                # Relational horizontal position of center of owner box
                offset_x = (owner_center_x - self.camera_center_x) / (self.camera_width / 2)
                # Relational vertical position of center of owner box
                offset_y = (owner_center_y - self.camera_center_y) / (self.camera_height / 2)
                # NOTE: Global variable
                #with self.args[-1]:
                #    #owner_offset = offset_x
                #    self.args[1].value = offset_x
                #owner_offset = offset_x
                owner_offset = [offset_x, offset_y]
                offset_arg = convert_arr_to_ros_msg(owner_offset)
                self.pub_owner.publish(offset_arg)
            else:
                # If the PETSON loses its owner, it resets the owner.
                self.owner_id = None
                cur_owner_id = self.owner_id

                # NOTE: Global variable
                #with self.args[-1]:
                #    #owner_offset = 0.
                #    self.args[1].value = 0.
                #owner_offset = 0.
                owner_offset = [0., 0.]
                offset_arg = convert_arr_to_ros_msg(owner_offset)
                self.pub_owner.publish(offset_arg)
        else:
            # NOTE: Global variable
            #with self.args[-1]:
            #    #owner_offset = 0.
            #    self.args[1].value = 0.
            #owner_offset = 0.
            owner_offset = [0., 0.]
            offset_arg = convert_arr_to_ros_msg(owner_offset)
            self.pub_owner.publish(offset_arg)
        #if owner_offset != 0.:
        #    print(f"owner_offset = {owner_offset}")
        
        '''
        if Detect_scotty: # if {16 : Dog, 17 : Cat, 18 : Horse} detected, 
            # Plot the image with detections | Plot returns a BGR numpy array of predictions
            im_array = Detect_scotty.plot()
            # Display the image using OpenCV
            #cv2.imshow("YOLO V8 Detection", im_array)
            try:
                # Convert the BGR numpy array image to a ROS CompressedImage message
                #ros_msg = self.bridge.cv2_to_compressed_imgmsg(im_array)
                #self.image_pub.publish(ros_msg)
                self.pub.publish("The Scotty is detected")
                #self.pub_scotty.publish("Detected")
            except CvBridgeError as e:
                print(e)
        '''
            
        out_img = img if not (have_results and owner_box_detected is not None) else owner_annotated_img
        #out_img = img if not Detect_scotty else im_array
        #out_img = img
        #cv2.imshow("Owner detection video window", out_img)
        #cv2.waitKey(3)
        try:
            self.image_pub.publish(self.bridge.cv2_to_compressed_imgmsg(out_img))
        except CvBridgeError as e:
            print(e)

def video_stream_main(*args):
    # ARGS:
    #owner_tolerance, owner_offset, cur_owner_id
    try:
        del rospy.names.get_mappings()['__name']
    except KeyError:
        pass
    # For test
    rospy.init_node('image_converter', anonymous=True)
    ic = image_converter(args)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

def auto_move_main():
    try:
        del rospy.names.get_mappings()['__name']
    except KeyError:
        pass
    rospy.init_node('Multipoint_navigation', anonymous=True)
    mn = Multipoint_navigation()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down mn")

def audio_main(*args):
    # ARGS
    # Mode, Path, textfile
    try:
        del rospy.names.get_mappings()['__name']
    except KeyError:
        pass
    # For test
    rospy.init_node('jetbot_vad_node', anonymous=True)
    #TODO: VAD/ASR/...
    vr = VoiceRecognition(*args)

    got_a_sentence = False
    leave = False
    while not rospy.is_shutdown():
        stream = vr.pa.open(format=FORMAT, channels=AUDIO_CHANNELS,
                                rate=AUDIO_RATE, input=True, start=False,
                                frames_per_buffer=CHUNK_SIZE)
        if leave:
          print(f"Leave triggered")
          stream.close()

          #assert vr.Mode == "asr_en", "Mode should be given as 'asr_en'"
          if vr.Mode != 'asr_en':
              vr.Mode = 'asr_en'
          run("python3 " + vr.Path + "/scripts/ginput.py -i " + Path + "/data/record.wav -o " + Path + "/data/test.wav")
          if os.path.exists(textfile):
            s = None
            with open(textfile, "r") as f:
              s = f.readlines()
            if s is not None:
              print('Content of the text file\n', s)

              # Prompt: recognize the command
              prompt = f"What is the intention of the given text? Select one from the options.\nText: {s[0]}\nOptions: 'Turn left', 'Turn right', 'Move forward', 'Go back', 'Return to home'\nPlease remove the Explanation.\nAnswer:"
              results = query_huggingface_model(vr.MODEL_NAME, prompt, vr.HF_API_TOKEN)
              # Handling 'model loading...' error.
              # Without throwing 503 error, this one waits for respond.
              if 'error' in results:
                  results = query_huggingface_model(vr.MODEL_NAME, prompt, vr.HF_API_TOKEN, wait_for_model=True)
              #time.sleep(1)
              print(f"\n\n\nGenerated decision: {results[0]['generated_text']}\n\n\n")
              result = results[0]['generated_text'].strip()

              # Do command
              vr.pub_text.publish(result)
              
              # Prompt: QA
              prompt_for_answer = f"Your name is 'Petson' and you are a moving pet that helps people. Answer to the given text:{s[0]}"
              results_answer = query_huggingface_model(vr.MODEL_NAME, prompt_for_answer, vr.HF_API_TOKEN)
              # Handling 'model loading...' error.
              # Without throwing 503 error, this one waits for respond.
              if 'error' in results_answer:
                  results_answer = query_huggingface_model(vr.MODEL_NAME, prompt_for_answer, vr.HF_API_TOKEN, wait_for_model=True)
              #time.sleep(1)
              result_answer = results_answer[0]['generated_text']
              print(f"\nGenerated answer: {result_answer}\n")
              TTS_pub.publish(result_answer)
              
              #TODO: Publish generated text
              ##     by print? tts?
              #pub.publish(s[0].strip('\n'))

              if 'return to home' in result.lower():
                  print(f"ASR ended")
                  break
          run("rm -r " + textfile)    

          if os.path.exists(vr.Path+"/data/record.wav"):
              run("rm -r " + Path + "/data/record.wav")
          leave = False
          got_a_sentence = False
        else:
            ####
            ring_buffer = collections.deque(maxlen=NUM_PADDING_CHUNKS)
            triggered = False
            voiced_frames = []
            ring_buffer_flags = [0] * NUM_WINDOW_CHUNKS
            ring_buffer_index = 0
            ring_buffer_flags_end = [0] * NUM_WINDOW_CHUNKS_END
            ring_buffer_index_end = 0
            # WangS
            raw_data = array('h')
            index = 0
            start_point = 0
            StartTime = time.time()
            print("* recording: ")
            stream.start_stream()

            ##############
            while not got_a_sentence and not rospy.is_shutdown():
                chunk = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                # add WangS
                raw_data.extend(array('h', chunk))
                index += CHUNK_SIZE
                active = vr.vad.is_speech(chunk, AUDIO_RATE)

                ring_buffer_flags[ring_buffer_index] = 1 if active else 0
                ring_buffer_index = (ring_buffer_index + 1) % NUM_WINDOW_CHUNKS
                ring_buffer_flags_end[ring_buffer_index_end] = 1 if active else 0
                ring_buffer_index_end = (ring_buffer_index_end + 1) % NUM_WINDOW_CHUNKS_END

                if triggered:
                  #print(f"triggered")
                  # voiced_frames.append(chunk)
                  ring_buffer.append(chunk)
                  num_unvoiced = NUM_WINDOW_CHUNKS_END - sum(ring_buffer_flags_end)
                  if num_unvoiced > 0.90 * NUM_WINDOW_CHUNKS_END \
                          or (time.time() - StartTime) > 10:
                    sys.stdout.write(' Close\n')
                    triggered = False
                    got_a_sentence = True
                else:
                  #print(f"NOT triggered")
                  # Start point detection
                  ring_buffer.append(chunk)
                  num_voiced = sum(ring_buffer_flags)
                  if num_voiced > 0.8 * NUM_WINDOW_CHUNKS:
                    sys.stdout.write(' Open ')
                    StartTime = time.time()
                    triggered = True
                    start_point = index - CHUNK_SIZE * 20 # start point
                    # voiced_frames.extend(ring_buffer)
                    ring_buffer.clear()
                  # End point detection
                sys.stdout.flush()
        
            stream.stop_stream()
            print('* done recording')
            got_a_sentence = False

            # Write to file
            raw_data.reverse()
            for i in range(start_point):
                raw_data.pop()
            raw_data.reverse()
            raw_data = normalize(raw_data)
            record_to_file(vr.Path+"/data/record.wav", raw_data, 2)
            leave = True


## Teleop_key
moveBindings = {
        'i':(1,0,0,0), # 방향키 위쪽
        'o':(1,0,0,-1), 
        'j':(0,0,0,1),
        'l':(0,0,0,-1),
        'u':(1,0,0,1),
        ',':(-1,0,0,0), # 방향키 아래쪽
        '.':(-1,0,0,1),
        'm':(-1,0,0,-1),
        'O':(1,-1,0,0),
        'I':(1,0,0,0),
        'J':(0,1,0,0), # 오른쪽으로 회전
        'L':(0,-1,0,0), # 왼쪽으로 회전
        'U':(1,1,0,0),
        '<':(-1,0,0,0),
        '>':(-1,-1,0,0),
        'M':(-1,1,0,0),
        't':(0,0,1,0),
        'k':(0,0,0,0),
        ' ':(0,0,0,0),
        'b':(0,0,-1,0),
        'A':(1,0,0,0), #앞
        'B':(-1,0,0,0), #뒤
        'C':(0,0,0,-1), #오
        'D':(0,0,0,1), #왼
}

speedBindings={
        'q':(1.1,1.1),
        'z':(.9,.9),
        'w':(1.1,1),
        'x':(.9,1),
        'e':(1,1.1),
        'c':(1,.9),
}

def getKey():
    tty.setraw(sys.stdin.fileno())
    select.select([sys.stdin], [], [], 0.01)
    key = sys.stdin.read(1)
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

def vels(speed, turn):
    return "currently:\tspeed %s\tturn %s " % (speed, turn)

def move_bigger(pub, speed, turn): #계속 돌고 있으니까, 어디론가 움직여주는듯?
    twist = Twist()
    twist.linear.x = -1
    twist.angular.z = 0
    for _ in range(3):
        pub.publish(twist)
    twist.linear.x = 0
    twist.angular.z = -turn * 1.57
    pub.publish(twist)
    print("Do move_bigger")
    
def scotty_callback(data): #scotty 감지하면 콜백할라고 함 (해당 서브스크라이버용)
    rospy.loginfo('The scotty is %s', data.data)
    global scotty_detected
    try:
        scotty_detected = True
    except Exception as e:
        print("receive msg,but parse exception:", e)        
def owner_callback(data):
    #rospy.loginfo('The owner is %s', data.data)
    global owner_offset
    owner_offset[0] = data.data[0]
    owner_offset[1] = data.data[1]
def voice_callback(data):
    rospy.loginfo('Voice recongized as %s', data.data)
    global recognized_direction
    recognized_direction = data.data
def TTS_callback(data):
    rospy.loginfo('I heard %s', data.data)
    try:
        tts = gTTS(text=data.data, lang='en-us')
        tts.save(FilePath + "/demo.mp3")
        p=Popen("play -q "+ FilePath + "/demo.mp3", stdout=PIPE, shell=True)
        p.wait()
        os.remove(FilePath + '/demo.mp3')
    except Exception as e:
        print("receive msg,but parse exception:", e)

## MAIN
if __name__ == "__main__":
    #global owner_tolerance, owner_offset, cur_owner_id
    Mode = rospy.get_param("~Mode", 'play')
    Path = rospy.get_param("~Path", '/home/orin/catkin_ws/src/jetbot_pro')
    textfile = Path + "/data/talk.txt"
    
    settings = termios.tcgetattr(sys.stdin)

    pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
    #rospy.Subscriber('chatter', String, scotty_callback, queue_size=1)
    rospy.Subscriber('owner_chat', Float32MultiArray, owner_callback, queue_size=1)
    #voice_pub = rospy.Publisher('chatter', String, queue_size=10)
    # voice_sub
    rospy.Subscriber('voice_text', String, voice_callback, queue_size=1)
    rospy.Subscriber('speak', String, TTS_callback, queue_size=1)
    TTS_pub = rospy.Publisher('speak', String, queue_size=10)


    ## Multiprocessing
    procs = []
    # Video streaming
    proc1 = Process(target=video_stream_main, args=(owner_tolerance, owner_offset, cur_owner_id))
    procs.append(proc1)
    proc1.start()

    # VAD/ASR/...
    proc2 = Process(target=audio_main, args=(Mode, Path, textfile))
    procs.append(proc2)
    proc2.start()

    # Move automatically
    #ic = image_converter()
    rospy.init_node('petson_decision_demo')

    tf_sub = tf.TransformListener()

    ##### Initialize variables related to movement & avoidance
    speed = rospy.get_param("~speed", 0.5)
    # speed = 1
    default_speed = speed
    turn = rospy.get_param("~turn", 1.0)
    default_turn = turn
    avoid_distance = 0.4 #0.4 # min(2*speed, 0.3) #0.2 #0.2
    x,y,z,th,status = 0,0,0,0,0
    Angle = 60
    start, linear, angular = False, default_speed, default_turn
    warning = [0,0,0] # left, middle, right
    #scotty_detected = False

    def laser_callback(data):
        global warning
        newdata = data

        # Convert tuple to list
        newdata.ranges = list(data.ranges)
        newdata.intensities = list(data.intensities)

        length = len(data.ranges)
        #Index = int(th/2*length/360)
        #Index = int(th/2*length)
        Index = int(Angle/2*length/360)

        warning = [0,0,0]
        # Middle
        if min(data.ranges[0:Index]) < avoid_distance \
                or min(data.ranges[(length-Index):]) < avoid_distance:
            warning[1] = 1
        # Left
        if min(data.ranges[int(length/4)-Index:int(length/4)+Index]) < avoid_distance:
            warning[0] = 1
        # Right 
        if min(data.ranges[int(length*3/4)-Index:int(length*3/4)+Index]) < avoid_distance:
            warning[2] = 1

    laser_sub = rospy.Subscriber("scan", LaserScan, laser_callback)

    def config_callback(config, level):
        start = config['start']
        th = config['Angle']
        speed = min(config['distance'], default_speed)
        linear = config['linear']
        angular = config['angular']
        return config

    Server(laserAvoidanceConfig, config_callback)

    # movement 후보 -> 장애물 생길때 / 적절한 시기 (or trans) 기준으로 후보 움직여도,,, 
    # search
    # reach
    # return

    # turn_key = ['L', 'J']
    # move_key = 'i' # 
    movements = ['A', 'C', 'A', 'D']
    key = movements[0]
    keyid = 0
    
    safe_drive = 0
    avoidance = 0

    try:
        #print(msg)
        #print(vels(speed,turn))
        RATE = 5
        r = rospy.Rate(RATE) # 10hz = 0.1s
        # rospy.spin()
        # For initial setup, sleep main process for a bit
        # to make system to work on vision to appear first.
        for i in range(30):
            r.sleep()

        while not rospy.is_shutdown():
            # Normal keyboard manipulation
            # key = getKey()
            
            '''
            if scotty_detected == True:
                print(f"Scotty detected! break key-loop!")
                break
            '''
            
            try:
                (trans, rot) = tf_sub.lookupTransform('/map', '/base_footprint', rospy.Time(0))
                #rospy.loginfo('---------')
                #rospy.loginfo('Translation: ' + str(trans))
                #rospy.loginfo('Rotation: ' + str(rot))
                #rospy.loginfo('---------')
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                pass
            if key == '\x03': # 여긴 어떻게 하징
                break
            
            '''
            if avoidance > RATE:
                move_bigger(pub, linear, angular)
                avoidance = 0
                continue
 
            if safe_drive > (int(RATE/2)+1):
                avoidance = 0
            '''

            # if there is an obstacle
            if sum(warning) != 0:
                # Avoid obstacle
                print('avoiding with warning: ', warning)
                avoidance += 1
                safe_drive = 0
                
                twist = Twist()
                if warning[1] == 0:
                    # Forward
                    twist.linear.x = linear
                    twist.angular.z = 0
                elif warning[0] == 0:
                    # Turn left
                    twist.linear.x = 0
                    twist.angular.z = angular
                elif warning[2] == 0:
                    # Turn right
                    twist.linear.x = 0
                    twist.angular.z = -angular
                else:
                    # Turn left/right, random.
                    if random.randint(0,1) == 1:
                        twist.angular.z = angular
                    else:
                        twist.angular.z = -angular
                    twist.linear.x = 0
                    #if linear > 0:
                    #    twist.linear.x = -linear
                    #else:
                    #    twist.linear.x = linear
                pub.publish(twist)
                warning = [0,0,0]
                time.sleep(10/RATE)
                continue
                # Avoid obstacle end

            # Actuate the voice-directed movement.
            if recognized_direction is not None:
                # TODO: Move
                print(f"Move following the voice: {recognized_direction}")
                recognized_direction = recognized_direction.lower()
                twist = Twist()
                if "move forward" in recognized_direction:
                    # Forward
                    twist.linear.x = linear
                    twist.angular.z = 0
                elif "turn left" in recognized_direction:
                    # Turn left
                    twist.linear.x = 0
                    twist.angular.z = angular
                elif "turn right" in recognized_direction:
                    # Turn right
                    twist.linear.x = 0
                    twist.angular.z = -angular
                elif "go back" in recognized_direction:
                    # Go back
                    twist.linear.x = -linear
                    twist.angular.z = 0
                elif "return to home" in recognized_direction:
                    # Go to home.
                    break
                else:
                    # If not recognized, do nothing.
                    recognized_direction = None
                    continue
                pub.publish(twist)
                recognized_direction = None
                time.sleep(10/RATE)
                continue
                # Actuation end

            # Owner following (Just rotate, for now.)
            ## move forward, regarding the owner's coordinates.
            #if (cur_owner_id is not None) and \
            #(owner_offset[0] > owner_tolerance):
            x_offset_amount = abs(owner_offset[0])
            if x_offset_amount > owner_tolerance:
                print(f"owner_offset: {owner_offset}")
                print(f"owner_tolerance: {owner_tolerance}")
                # Rotate to align horizontally.
                twist = Twist()
                if owner_offset[0] > 0:
                    ## If owner is on right, turn left
                    twist.linear.x = linear
                    twist.angular.z = -angular * max(x_offset_amount, owner_tolerance)
                else:
                    ## If owner is on left, turn right
                    twist.linear.x = linear
                    twist.angular.z = angular * max(x_offset_amount, owner_tolerance)
                pub.publish(twist)
                time.sleep(10/RATE)
                continue
            # Owner following end

            ## DEBUG
            time.sleep(10/RATE)
            continue

            #print(f"key: {key}")
            if key in moveBindings.keys():
                safe_drive += 1
                x,y,z,th = moveBindings[key][0], moveBindings[key][1], moveBindings[key][2], moveBindings[key][3]
                twist = Twist()
                twist.linear.x, twist.linear.y, twist.linear.z = x*speed, y*speed, z*speed
                #print(f'x: {twist.linear.x}, y: ,{twist.linear.y}, z: {twist.linear.z}, th: {th}')
                twist.angular.x, twist.angular.y, twist.angular.z = 0, 0, th*turn
                pub.publish(twist)
            elif key in speedBindings.keys():
                speed, turn = speed * speedBindings[key][0], turn * speedBindings[key][1]
                print(vels(speed,turn))
                if status == 14:
                    print(msg)
                status = (status+1)%15
            else:
                x,y,z,th = 0,0,0,0
                if key == '\x03':
                    break
            time.sleep(10/RATE)
            
        #print(scotty_detected)
            
        '''
        print(f"Begin initialization of Go home")
        gohome = Multipoint_navigation()
        print(f"Go home initialized")
        #gohome.PubTargetPoint(0, 0)
        #gohome.index += 1
        #gohome.count += 1
        #print(f"Go home published target point of (0, 0)")
        print('Add a new target point ' + str(gohome.count) + '.')
        # Create a marker object
        marker = Marker()
        marker.header.frame_id = 'map'
        # Character format
        marker.type = marker.TEXT_VIEW_FACING
        # marker model
        marker.action = marker.ADD
        # the size of the marker
        marker.scale.x = 0.6
        marker.scale.y = 0.6
        marker.scale.z = 0.6
        # marker ColorRGBA
        marker.color.r = 1
        marker.color.g = 0
        marker.color.b = 0
        marker.color.a = 1
        # marker position XYZ
        marker.pose.position.x = 0
        marker.pose.position.y = 0
        marker.pose.position.z = 0
        # marker text
        marker.text = str(gohome.count)
        gohome.markerArray.markers.append(marker)
        # Set the id of markers
        _id = 0
        for m in gohome.markerArray.markers:
            m.id = _id
            _id += 1

        # Publish target point
        #not_home = True
        #if gohome.count == 0:
        #    gohome.PubTargetPoint(0,0)
        #    gohome.index += 1
        #gohome.count += 1
        '''
        gohome_result = movebase_client()

        #r = rospy.Rate(10) # 10hz
        while not rospy.is_shutdown():
            try:
                #if gohome.goal_reached:
                if gohome_result:
                    print(f"gohome.goal_reached")
                    break
                else:
                    gohome.pub_mark.publish(gohome.markerArray)
                    print(f"Publish markerarray")
                    r.sleep()
            except KeyboardInterrupt:
                print("Shutting down")
                break
        
    except Exception as e:
        print(e)
    finally:
        print(f"Finish")
        twist = Twist()
        twist.linear.x, twist.linear.y, twist.linear.z = 0, 0, 0
        twist.angular.x, twist.angular.y, twist.angular.z = 0, 0, 0
        pub.publish(twist)

        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        #laser_sub.unregister()
        #cv2.destroyAllWindows()
        for proc in procs:
            if proc.is_alive():
                proc.join()

