import serial
import time

PORT = '/dev/ttyUSB0' # 방금 업로드 성공한 포트
BAUD = 230400         # 펌웨어에 세팅된 공식 속도

def read_spikerbox():
    try:
        # 포트 열기
        ser = serial.Serial(PORT, BAUD, timeout=1)
        print(f"✅ 포트 연결 성공: {PORT}")
        
        # 기기가 켜지며 보내는 "StartUp!" 문자열 및 쓰레기값 비우기
        time.sleep(1)
        ser.reset_input_buffer()
        print("🚀 뇌파 데이터 실시간 파싱 시작... (장비를 만져보세요!)\n")

        count = 0
        while True:
            if ser.in_waiting >= 2:
                # 1. 첫 번째 바이트 읽기
                b1 = ser.read(1)[0]
                
                # 2. Sync Bit(MSB가 1인지) 확인 (공식 코드 로직)
                if b1 > 127: 
                    b2 = ser.read(1)[0]
                    
                    # 두 번째 바이트는 MSB가 0이어야 함
                    if b2 <= 127: 
                        # 3. 10-bit 원래 숫자로 복원
                        val = ((b1 & 0x7F) << 7) | (b2 & 0x7F)
                        
                        # 화면이 멈추지 않도록 1000개(약 0.1초) 단위로 하나씩만 터미널에 출력
                        count += 1
                        if count % 1000 == 0: 
                            print(f"🧠 현재 뇌파(EEG) 수치: {val:4d}  |  누적 샘플: {count}개")
                            
    except KeyboardInterrupt:
        print(f"\n✋ 수신 종료. 총 {count}개의 데이터를 성공적으로 받았습니다.")
    except Exception as e:
        print(f"\n❌ 에러: {e}")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()

if __name__ == "__main__":
    read_spikerbox()