#!/usr/bin/python
# -*- coding:utf-8 -*-
'''
Requirements:
+ pyaudio - `pip3 install pyaudio`
+ py-webrtcvad - `pip3 install webrtcvad`
'''
import webrtcvad
import collections
import sys
import signal
import pyaudio
import requests

from array import array
from struct import pack
import wave
import time
import os
from subprocess import Popen, PIPE

import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Twist


FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK_DURATION_MS = 30       # supports 10, 20 and 30 (ms)
PADDING_DURATION_MS = 1500   # 1 sec jugement
CHUNK_SIZE = int(RATE * CHUNK_DURATION_MS / 1000)  # chunk to read
CHUNK_BYTES = CHUNK_SIZE * 2  # 16bit = 2 bytes, PCM
NUM_PADDING_CHUNKS = int(PADDING_DURATION_MS / CHUNK_DURATION_MS)
NUM_WINDOW_CHUNKS = int(400 / CHUNK_DURATION_MS)  # 400 ms/ 30ms  ge
NUM_WINDOW_CHUNKS_END = NUM_WINDOW_CHUNKS * 2

START_OFFSET = int(NUM_WINDOW_CHUNKS * CHUNK_DURATION_MS * 0.5 * RATE)

def record_to_file(path, data, sample_width):
    "Records from the microphone and outputs the resulting data to 'path'"
    # sample_width, data = record()
    data = pack('<' + ('h' * len(data)), *data)
    wf = wave.open(path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(data)
    wf.close()


def normalize(snd_data):
    "Average the volume out"
    MAXIMUM = 32767  # 16384
    times = float(MAXIMUM) / max(abs(i) for i in snd_data)
    r = array('h')
    for i in snd_data:
        r.append(int(i * times))
    return r

def run(cmd):
    p=Popen(cmd, stdout=PIPE, shell=True)
    p.wait()

#songmu
def query_huggingface_model(model_name, prompt, api_token):

    API_URL = f"https://api-inference.huggingface.co/models/{model_name}"
    headers = {"Authorization": f"Bearer {api_token}",
               "Content-Type": "application/json"}

    data = {"inputs": prompt}

    response = requests.post(API_URL, headers=headers, json=data)
    result = response.json()

    return result

## Teleop_key
moveBindings = {
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
#songmu

if __name__ == '__main__':
    rospy.init_node('jetbot_vad_node')
    Mode = rospy.get_param('~Mode','play')
    Path = rospy.get_param('~Path','/home/orin/catkin_ws/src/jetbot_pro')
    textfile = Path + "/data/talk.txt"
    pub = rospy.Publisher('chatter', String, queue_size=10)
    pub_move = rospy.Publisher('cmd_vel', Twist, queue_size=1)
    print(textfile)
    print(__file__)
    vad = webrtcvad.Vad(2)
    pa = pyaudio.PyAudio()

    #songmu
    api_token = "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" # sh baek's API
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    
    
    speed = rospy.get_param("~speed", 0.5)
    default_speed = speed
    turn = rospy.get_param("~turn", 1.0)
    default_turn = turn
    x,y,z,th,status = 0,0,0,0,0
    Angle = 60
    start, linear, angular = False, default_speed, default_turn
    #songmu
    

    while not rospy.is_shutdown():
        stream = pa.open(format=FORMAT,
                         channels=CHANNELS,
                         rate=RATE,
                         input=True,
                         start=False,
                         frames_per_buffer=CHUNK_SIZE)

        got_a_sentence = False
        leave = False

        while not leave and not rospy.is_shutdown():
            ring_buffer = collections.deque(maxlen=NUM_PADDING_CHUNKS)
            triggered = False
            voiced_frames = []
            ring_buffer_flags = [0] * NUM_WINDOW_CHUNKS
            ring_buffer_index = 0

            ring_buffer_flags_end = [0] * NUM_WINDOW_CHUNKS_END
            ring_buffer_index_end = 0
            buffer_in = ''
            # WangS
            raw_data = array('h')
            index = 0
            start_point = 0
            StartTime = time.time()
            print("* recording: ")
            stream.start_stream()

            while not got_a_sentence and not leave and not rospy.is_shutdown():
                chunk = stream.read(CHUNK_SIZE,exception_on_overflow = False)
                # add WangS
                raw_data.extend(array('h', chunk))
                index += CHUNK_SIZE
                active = vad.is_speech(chunk, RATE)

                ring_buffer_flags[ring_buffer_index] = 1 if active else 0
                ring_buffer_index += 1
                
                ring_buffer_index %= NUM_WINDOW_CHUNKS

                ring_buffer_flags_end[ring_buffer_index_end] = 1 if active else 0
                ring_buffer_index_end += 1
                ring_buffer_index_end %= NUM_WINDOW_CHUNKS_END

                # start point detection
                if not triggered:
                    ring_buffer.append(chunk)
                    num_voiced = sum(ring_buffer_flags)
                    if num_voiced > 0.8 * NUM_WINDOW_CHUNKS:
                        sys.stdout.write(' Open ')
                        StartTime = time.time()
                        triggered = True
                        start_point = index - CHUNK_SIZE * 20  # start point
                        # voiced_frames.extend(ring_buffer)
                        ring_buffer.clear()
                # end point detection
                else:
                    # voiced_frames.append(chunk)
                    ring_buffer.append(chunk)
                    num_unvoiced = NUM_WINDOW_CHUNKS_END - sum(ring_buffer_flags_end)
                    if num_unvoiced > 0.90 * NUM_WINDOW_CHUNKS_END or (time.time() - StartTime) > 10:
                        sys.stdout.write(' Close \n')
                        triggered = False
                        got_a_sentence = True

                sys.stdout.flush()

            stream.stop_stream()
            print("* done recording")
            got_a_sentence = False

            # write to file
            raw_data.reverse()
            for index in range(start_point):
                raw_data.pop()
            raw_data.reverse()
            raw_data = normalize(raw_data)
            record_to_file(Path+"/data/record.wav", raw_data, 2)
            leave = True

        stream.close()
        

        if Mode == "asr_en":
            #run("~/env/bin/python3 " + Path + "/scripts/ginput.py -i " + Path + "/data/record.wav -o  " + Path + "/data/test.wav") #not used
            run("python3 " + Path + "/scripts/ginput.py -i " + Path + "/data/record.wav -o  " + Path + "/data/test.wav")
            # run("googlesamples-assistant-pushtotalk --project-id iitp-46605 --device-model-id iitp-46605-jetson-zmlp68 --once")
            #run("python " + Path + "/scripts/ginput.py -i " + Path + "/data/record.wav -o  " + Path + "/data/test.wav") #not used
            api_token = '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'
            if os.path.exists(textfile):
                s=None
                with open(textfile,"r") as f:
                    s=f.readlines()
                if s is not None:
                    print('Content of the text file\n', s)
                    
                    #songmu
                    #프롬프트 보고 어디로 움직이라는 명령을 찾아내기 위한 코드
                    prompt = f"What is the intention of the given text? Select one from the options.\nText: {s[0]}\nOptions: 'Turn left', 'Turn right', 'Move forward', 'Go back\nPlease remove the Explanation.\nAnswer:"
                    results = query_huggingface_model(model_name, prompt, api_token)
                    time.sleep(1)
                    # print(results)
                    print(f"\n\n\nGenerated text: {results[0]['generated_text']}\n\n\n")
                    result = results[0]['generated_text']
                    
                    #여기는 명령대로 움직이게 하는 코드
                    twist = Twist()

                    if "Turn left" in result:
                        x,y,z,th = moveBindings['D'][0], moveBindings['D'][1], moveBindings['D'][2], moveBindings['D'][3]
                        twist = Twist()
                        twist.linear.x, twist.linear.y, twist.linear.z = x*speed, y*speed, z*speed
                        twist.angular.x, twist.angular.y, twist.angular.z = 0, 0, th*turn
                        pub_move.publish(twist)
                        print('Turn left')
                    elif "Turn right" in result:
                        x,y,z,th = moveBindings['C'][0], moveBindings['C'][1], moveBindings['C'][2], moveBindings['C'][3]
                        twist = Twist()
                        twist.linear.x, twist.linear.y, twist.linear.z = x*speed, y*speed, z*speed
                        twist.angular.x, twist.angular.y, twist.angular.z = 0, 0, th*turn
                        pub_move.publish(twist)
                        print('Turn right')
                    elif "Move forward" in result:
                        x,y,z,th = moveBindings['A'][0], moveBindings['A'][1], moveBindings['A'][2], moveBindings['A'][3]
                        twist = Twist()
                        twist.linear.x, twist.linear.y, twist.linear.z = x*speed, y*speed, z*speed
                        twist.angular.x, twist.angular.y, twist.angular.z = 0, 0, th*turn
                        pub_move.publish(twist)
                        print('Move forward')
                    elif "Go back" in result:
                        x,y,z,th = moveBindings['B'][0], moveBindings['B'][1], moveBindings['B'][2], moveBindings['B'][3]
                        twist = Twist()
                        twist.linear.x, twist.linear.y, twist.linear.z = x*speed, y*speed, z*speed
                        twist.angular.x, twist.angular.y, twist.angular.z = 0, 0, th*turn
                        pub_move.publish(twist)
                        print('Go back')
                    else:
                        print('Nothing')
                        
                    prompt_for_answer = f'Act as a pet that helps people. Answer to the given text:{s[0]}'
                    results_answer = query_huggingface_model(model_name, prompt_for_answer, api_token)
                    time.sleep(1)
                    print(results_answer)
                    print(f"\nGenerated text: {results_answer[0]['generated_text']}\n")
                    result_answer = results_answer[0]['generated_text']
                    # 여기서 jetbot으로 넘겨줄지 그냥 print 해줄지?? tts 켜줄지??
                    #songmu
                    
                    pub.publish(s[0].strip('\n'))
                run("rm -r " + textfile)
        # elif Mode == "talk_en":
        #     #run("~/env/bin/python3 " + Path + "/scripts/ginput.py -i " + Path + "/data/record.wav -o  " + Path + "/data/test.wav")
        #     run("python3 " + Path + "/scripts/ginput.py -i " + Path + "/data/record.wav -o  " + Path + "/data/test.wav")
        #     #run("python " + Path + "/scripts/ginput.py -i " + Path + "/data/record.wav -o  " + Path + "/data/test.wav")
        #     if os.path.exists(textfile):
        #         s=None
        #         with open(textfile,"r") as f:
        #             s=f.readlines()
        #         if s is not None:
        #             try: 
        #                 print(s[0]+s[1])
        #                 pub.publish(s[0]+s[1])
        #             except: 
        #                 pass
        #         run("rm -r " + textfile)
        #         run("aplay -q -r 16000 -f S16_LE " + Path + "/data/test.wav")
        #         run("rm -r " + Path + "/data/test.wav")
        else:
            run("play -q "+Path+"/data/record.wav")
        
        if os.path.exists(Path+"/data/record.wav"):
            run("rm -r " + Path + "/data/record.wav")