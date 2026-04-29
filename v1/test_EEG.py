import threading
import serial
import time
import numpy as np

# 설정
port = '/dev/ttyUSB1' # 확인된 포트
baud = 230400         
sample_buffer = []

try:
    ser = serial.Serial(port, baud, timeout=1)
    print(f"📡 {port} 감시 중... 데이터가 들어오면 숫자가 변합니다.")
except:
    print("❌ 포트 열기 실패")
    exit()

def read_thread():
    global sample_buffer
    while True:
        if ser.in_waiting > 0:
            raw_data = ser.read(ser.in_waiting)
            # 데이터가 들어오는지 날것 그대로 확인
            print(f"📥 수신된 바이트: {len(raw_data)} | 첫 5바이트(hex): {raw_data[:5].hex()}", end='\r')
            
            # 파싱 로직 (간소화)
            for i in range(len(raw_data)-1):
                if raw_data[i] > 127: # Sync bit
                    val = ((raw_data[i] & 0x7F) << 7) | (raw_data[i+1] & 0x7F)
                    sample_buffer.append(val)
        time.sleep(0.01)

t = threading.Thread(target=read_thread, daemon=True)
t.start()

try:
    while True:
        # 1초마다 버퍼에 쌓인 데이터 개수 출력
        print(f"\n📊 현재 쌓인 뇌파 데이터 개수: {len(sample_buffer)}")
        time.sleep(1)
except KeyboardInterrupt:
    print("\n종료")