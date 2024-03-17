#!/usr/bin/env python
import rospy
from audio_common_msgs.msg import AudioData
import pyaudio
# from your_audio_capture_module import capture_audio  # Implement your audio capture logic here

def capture_audio():
    CHUNK = 1024  # Number of frames per buffer
    FORMAT = pyaudio.paInt16  # Audio format (16-bit PCM)
    CHANNELS = 1  # Number of audio channels
    RATE = 16000  # Sample rate

    audio = pyaudio.PyAudio()

    try:
        # Open stream
        stream = audio.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            input=True,
                            frames_per_buffer=CHUNK)
    except IOError as e:
        rospy.logerr(f"Error opening audio stream: {e}")
        return b''  # Return an empty byte string in case of an error

    audio_data = stream.read(CHUNK)
    return audio_data

def audio_publisher():
    rospy.init_node('audio_publisher', anonymous=True)
    rate = rospy.Rate(10)  # Adjust the rate based on your needs

    pub = rospy.Publisher('/audio_data', AudioData, queue_size=10)

    while not rospy.is_shutdown():
        audio_data = capture_audio()  # Implement this function to capture audio data
        msg = AudioData(data=audio_data)
        pub.publish(msg)
        rate.sleep()

if __name__ == '__main__':
    try:
        audio_publisher()
    except rospy.ROSInterruptException:
        pass