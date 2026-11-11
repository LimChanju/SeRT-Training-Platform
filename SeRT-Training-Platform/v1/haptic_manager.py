import websocket
import json
import threading

class HapticManager:
    def __init__(self, laptop_ip="127.0.0.1", port=15881):
        """
        laptop_ip: bHaptics Player가 실행 중인 윈도우 노트북의 IP 주소
        port: bHaptics 웹소켓 기본 포트 (15881)
        """
        self.ws_url = f"ws://{laptop_ip}:{port}/v2/feedbacks"
        self.enabled = True
        print(f"🌐 [Haptic] 중계기 연결 준비 완료 (대상: {self.ws_url})")

    def trigger_grasp_feedback(self, intensity=100, duration_ms=500):
        """
        로봇이 물체를 잡거나 들어올릴 때 오른손 글러브에 진동을 쏩니다.
        엄지(0), 검지(1), 중지(2) 포인트에 진동을 집중합니다.
        """
        if not self.enabled:
            return

        def send_ws_request():
            try:
                # 0.5초 안에 연결 안 되면 시뮬레이션 보호를 위해 포기
                ws = websocket.WebSocket()
                ws.connect(self.ws_url, timeout=0.5)

                # bHaptics Standard Protocol (v2) JSON 데이터
                payload = {
                    "Submit": [{
                        "Type": "dot",
                        "Key": "robot_grasp_event",
                        "Parameters": {
                            "Position": "GloveR", # 오른손 장갑 기준
                            "DotPoints": [
                                {"Index": 0, "Intensity": intensity}, # 엄지
                                {"Index": 1, "Intensity": intensity}, # 검지
                                {"Index": 2, "Intensity": intensity}  # 중지
                            ],
                            "DurationMillis": duration_ms
                        }
                    }]
                }
                ws.send(json.dumps(payload))
                ws.close()
            except Exception as e:
                # 시뮬레이션 루프에 방해되지 않도록 에러는 조용히 출력만 함
                print(f"⚠️ [Haptic] 진동 전송 실패: {e} (노트북 IP와 Player 실행 여부를 확인하세요)")

        # 🚨 중요: 네트워크 통신으로 인해 시뮬레이션 프레임이 끊기지 않도록 
        # 별도의 스레드(Background)에서 실행합니다.
        threading.Thread(target=send_ws_request, daemon=True).start()