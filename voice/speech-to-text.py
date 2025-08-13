import os
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
import Levenshtein
import time
import queue
import threading
import logging


class STT:
    def __init__(self, model_type="tiny", device="cpu"):
        self.model_type = model_type
        self.device = device
        self.compute_type = "cpu"
        self.dtype="float32"
        self.model = WhisperModel(self.model_type, device=self.device, compute_type=self.dtype)
        self.sample_rate = 16000
        self.channels = 1
        self.record_seconds = 3
        self.silence_threshold = 0.01
        self.min_non_silent_chunks = 2
        self.audio_queue = queue.Queue()
        self.commands = [ "하이 비렉스", "마사지", "안마", "온열", "진행", "켜줘", "시작", "실행", "start",
                          "멈춰", "꺼줘", "중지", "정지", "끝", "종료", "stop",
                          "세게", "강하게", "살살", "약하게",
                          "모드", "1번", "2번", "3번", "4번", "5번", "6번", "7번", "8번", "9번", "10번",
                          "바이 비렉스" ]
        self.wake_word = "하이 비렉스"
        self.similarity_threshold = 2
        self.similar_commands = {
            "하이 비렉스": [
                "아이 비렉스", "하이 디렉스", "하이 비맥스", "하위 비렉스", "아이보렉스",
                "하이 빅스", "하이 리벡스", "하이 비랙스", "하 이 비렉스", "비렉스",
                "하이파이브", "아이 빅스비", "카이 비렉스", "사이비 렉스", "파이 비렉스",
                "아이비 렉스", "아이브렉스", "하이 빅스비", "헤이 비렉스", "하이프렉스"
            ],
            "마사지": [
                "마싸지", "마사쥐", "마사지이", "마사아지", "마 자지", "맛사지", "메시지",
                "마 사지", "마 사쥐", "마 싸지", "맛사쥐", "마사 지", "마자지", "마사치",
                "마사징", "마싸쥐", "머사지", "바사지", "파사지", "사사지"
            ],
            "안마": [
                "암마", "아마", "안마아", "안 마", "악마", "엄마", "아안마", "안나", "암 마",
                "안 막", "아 마", "알마", "안 바", "안막", "안 만", "인마", "단마", "암마아",
                "안마하", "안마으"
            ],
            "온열": [
                "오녈", "온녈", "온돌", "운열", "온혈", "오녈 켜줘", "온 널", "온 열",
                "온욜", "오놀", "원열", "온렬", "온율", "오냐", "본열", "손열", "돈열",
                "곤열", "몬열", "온열요"
            ],
            "진행": [
                "지냉", "지네", "진 행", "진핵", "지해", "친행", "지넹", "진횅",
                "치냉", "신행", "진생", "진헹", "지내", "진해", "진흥", "긴행",
                "딘행", "빈행", "신행", "짐행"
            ],
            "켜줘": [
                "켜져", "켜저", "켜", "틀어줘", "틀어", "켜주어", "켜 줘", "커줘",
                "커져", "켜조", "켜 조", "케줘", "크줘", "쳐줘", "처줘", "켜죠",
                "키워줘", "켜져라", "터줘", "켜요"
            ],
            "시작": [
                "시자", "시잗", "시 작", "시삭", "시 자", "시지작", "시차", "시잔",
                "시잘", "시약", "시샥", "시작을", "지작", "치작", "피작", "비작",
                "시닥", "시샥", "시잡", "시작요"
            ],
            "실행": [
                "시랭", "실 행", "실해", "실힝", "시렝", "시행", "실헹", "시래",
                "실애", "실앵", "슬행", "실행을", "취랭", "지랭", "필행", "시렝",
                "시래엥", "시렝이", "실행이", "실행요"
            ],
            "start": [
                "스타트", "스탈트", "스타투", "스 타트", "사트", "스따뜨", "스테이트",
                "스마트", "타트", "스타일", "세트", "스탓", "스타 트", "스탓트",
                "스타더", "스파트", "스타프", "스카트", "스나트", "스탓트업"
            ],
            "멈춰": [
                "멈처", "멈쳐", "멈춤", "멈춰라", "멈 처", "멈 춰", "멈추어", "모처",
                "머처", "멈초", "만춰", "멈추", "멈쵸", "범춰", "멍춰", "머쳐",
                "멈추라", "멈춰엇", "멈춰씀", "멈춰요"
            ],
            "꺼줘": [
                "꺼져", "꺼저", "꺼", "꺼 줘", "꺼조", "꺼죠", "끄어줘", "끄어",
                "끄어조", "꼬줘", "꿔줘", "꺼져라", "꺼저라", "꺼지", "꺼저요",
                "꺼주어", "꺼주세요", "꺼지다", "끄다", "꺼요"
            ],
            "중지": [
                "중쥐", "중단", "중 지", "준지", "둥지", "숭지", "충지", "증지",
                "중지이", "중시", "중진", "중쥐이", "주웅지", "주지", "중지요",
                "중지해", "중진행", "중지됨", "중지요청"
            ],
            "정지": [
                "정쥐", "정 지", "전지", "정자", "성지", "멍지", "저지", "정진",
                "정지이", "청지", "점지", "정지혜", "전쥐", "정쥣", "정지요",
                "정지해", "정지선", "정지됨", "정지요청"
            ],
            "끝": [
                "끗", "쫑", "끗끗", "끋", "끝이", "끕", "끝 트", "큿", "긋",
                "끄읏", "끄읕", "키읔", "치읓", "끝나", "끝이다", "끝임", "급",
                "큣", "끋끝", "끝요"
            ],
            "종료": [
                "종뇨", "종노", "종 료", "종요", "족료", "총료", "종로", "종결",
                "존료", "종뉴", "종용", "정료", "좀료", "종료되", "종료됨",
                "종료료", "종료해", "종료요", "종료신청"
            ],
            "stop": [
                "스탑", "스톱", "스땁", "스 탑", "스돕", "스팝", "스답", "샵",
                "스탭", "스토브", "스타프", "스콥", "스롭", "스토옵", "스 타압",
                "시탑", "스돕해", "스탑해", "스톱해", "스톱요"
            ],
            "세게": [
                "쎄게", "쌔게", "세개", "세계", "쌔개", "쎄개", "세 게", "쎄 게",
                "쌔 게", "씨게", "쌔걔", "새개", "새계", "세기", "스게", "체게",
                "페게", "세게좀", "세게요", "쎄게요"
            ],
            "강하게": [
                "강하계", "강하개", "강 하게", "가하게", "각하게", "캉하게", "간하게",
                "걍하게", "감하게", "강항에", "깡하게", "항하게", "광하게", "겅하게",
                "걍 하 게", "강하게좀", "강하게요", "강하계요", "가장 강하게"
            ],
            "살살": [
                "살 살", "살사", "살살이", "샤르살", "샤르샤르", "사살", "잘살",
                "쌀살", "살살살", "살살로", "탈살", "팔살", "사알살", "살살하게",
                "살살이요", "살살좀", "샬샬", "살살해", "살살요"
            ],
            "약하게": [
                "약하계", "약하개", "약 하게", "야가게", "약 가게", "약하걔", "악하게",
                "약캐", "야카게", "얗하게", "락하게", "약하게좀", "약하게요",
                "약하계요", "약카게", "야크하게", "야가계", "약카계", "제일 약하게"
            ],
            "모드": [
                "모두", "모 드", "모듬", "보드", "머드", "코드", "노드", "로드",
                "포드", "묘드", "무드", "모드요", "모드좀", "모오드", "모드으",
                "모두요", "로드요", "보드요", "모드 선택"
            ],
            "1번": [
                "일번", "이번", "일본", "1 번", "일", "첫번째", "하나", "일 반",
                "이얼번", "일벗", "힐번", "길번", "닐번", "빌번", "일 버",
                "일번으로", "일번이요", "일번 입니다", "에이번", "에이 번"
            ],
            "2번": [
                "이번", "이 번", "이버", "이 본", "이", "두번째", "둘", "이 반",
                "이 벗", "니번", "리번", "미번", "비번", "이 번호", "이반",
                "이원", "이번으로", "이번 입니다", "이번이요", "비 번"
            ],
            "3번": [
                "삼번", "삼 번", "사번", "산번", "삼", "세번째", "셋", "삼 본",
                "산 본", "삼 벗", "잠번", "참번", "함번", "삶번", "삼 번호",
                "삼번으로", "삼번 입니다", "삼번이요", "산본역", "삼 번은"
            ],
            "4번": [
                "사번", "사 번", "사본", "사 반", "사", "네번째", "넷", "삽번",
                "자번", "타번", "사벗", "사 번호", "사번으로", "사번 입니다",
                "사번이요", "사본떠", "싸번", "사범", "하번", "사 번은"
            ],
            "5번": [
                "오번", "오 번", "오본", "오 반", "오", "다섯번째", "다섯", "오방",
                "오범", "호번", "오벗", "오 번호", "오번으로", "오번 입니다",
                "오번이요", "오빤", "공번", "요번", "오벗", "오 번은"
            ],
            "6번": [
                "육번", "육 번", "유번", "육본", "육", "여섯번째", "여섯", "뉴번",
                "익번", "육벗", "육 번호", "육번으로", "육번 입니다", "육번이요",
                "육반", "국번", "둑번", "묵번", "육 번은"
            ],
            "7번": [
                "칠번", "칠 번", "실번", "치번", "칠", "일곱번째", "일곱", "칠본",
                "질번", "칠벗", "칠 번호", "칠번으로", "칠번 입니다", "칠번이요",
                "친번", "필번", "틸번", "힐번", "칠 번은"
            ],
            "8번": [
                "팔번", "팔 번", "팔본", "팔 반", "팔", "여덟번째", "여덟", "파일번",
                "할번", "팔벗", "팔 번호", "팔번으로", "팔번 입니다", "팔번이요",
                "팔반", "발번", "알번", "칼번", "팔 번은"
            ],
            "9번": [
                "구번", "구 번", "구본", "구", "아홉번째", "아홉", "두번", "쿠번",
                "후번", "구벗", "구 번호", "구번으로", "구번 입니다", "구번이요",
                "굿번", "주번", "추번", "무번", "구 번은"
            ],
            "10번": [
                "십번", "십 번", "십본", "십", "열번째", "열", "시번", "집번",
                "싶번", "십벗", "십 번호", "십번으로", "십번 입니다", "십번이요",
                "힙번", "딥번", "빕번", "닛번", "십 번은"
            ],
            "바이 비렉스": [
                "바이바이 비렉스", "발 비렉스", "바이 디렉스", "바 이 비렉스", "바이 비맥스",
                "바위 비렉스", "파이 비렉스", "잘가 비렉스", "비렉스 안녕", "아이 비렉스",
                "다이 비렉스", "바이 베렉스", "바이버 렉스", "파이 빅스비", "바이 비랙스",
                "빠이 비렉스", "바이바이", "바이 비레스", "바이 비렉", "바이 비엑스"
            ]
        }

    def recognize_wake_word(self, user_id, selected_mode):
        return self.conn.commit()

    def calculate_rms(self, audio_chunk):
        """오디오 청크의 RMS(Root Mean Square)를 계산하여 볼륨 수준을 추정합니다."""
        return np.sqrt(np.mean(audio_chunk**2))

    def audio_callback(self, indata, frames, time, status):
        """마이크 입력이 있을 때마다 호출되는 콜백 함수"""
        if status:
            print(f"오디오 입력 오류: {status}", flush=True)
        # 들어온 오디오 데이터를 큐에 넣습니다.
        self.audio_queue.put(indata.copy())

    def find_closest_command(self, text):
        """
        인식된 텍스트와 가장 유사한 명령어를 찾습니다.
        우선순위:
        1. 정확히 일치하는 명령어
        2. SIMILAR_COMMAND_MAP에 정의된 유사/오인식 명령어
        3. Levenshtein 거리 기반 유사 명령어 (임계값 이내)
        """
        if not text:
            return None

        normalized_text = text.strip().lower() # 공백 제거 및 소문자 변환 (비교용)

        # 1. 정확히 일치하는 경우 (대소문자 구분 없이)
        for cmd in self.commands:
            if cmd.lower() == normalized_text:
                print(f"    [정확도 검사] 원문: '{text}', 정확히 일치: '{cmd}'")
                return cmd # 원본 명령어 형식으로 반환

        # 2. SIMILAR_COMMAND_MAP에서 직접 매핑 확인
        for target_command, similar_list in self.similar_commands.items():
            if normalized_text in [sim.lower() for sim in similar_list]:
                print(f"    [유사어 맵] 원문: '{text}', 매핑된 명령어: '{target_command}' (사전 정의 목록 일치)")
                return target_command

        # 3. Levenshtein 거리 기반 가장 유사한 명령어 찾기
        min_distance = float('inf')
        closest_command = None

        for cmd in self.commands:
            # Levenshtein 비교 시에도 소문자로 변환된 텍스트와 명령어 사용
            distance = Levenshtein.distance(normalized_text, cmd.lower())
            if distance < min_distance:
                min_distance = distance
                closest_command = cmd # 원본 명령어 형식 저장

        # 4. 임계값 확인
        if closest_command and min_distance <= self.similarity_threshold:
            print(f"    [유사도 검사] 원문: '{text}', 가장 유사: '{closest_command}', 거리: {min_distance} (임계값: {self.similarity_threshold})")
            return closest_command
        else:
            print(f"    [유사도 검사] 원문: '{text}'와(과) 유사한 명령어를 찾지 못했습니다 (최소 거리: {min_distance}, 임계값: {self.similarity_threshold}).")
            return None


# --- 메인 실행 로직 ---
if __name__ == "__main__":

    stt_module = STT()
    # print("Faster Whisper 모델 로딩 중...")
    # # 모델 로딩 시 language="ko"를 지정하면 초기 로딩이 더 빨라질 수 있습니다.
    # # beam_size=1 은 가장 가능성 높은 결과만 반환하여 속도를 높이지만 정확도는 약간 떨어질 수 있습니다.
    # print(f"모델 로딩 완료: {MODEL_SIZE} ({DEVICE}, {COMPUTE_TYPE})")
    # print("명령어 목록:", COMMANDS)
    # print(f"유사도 임계값 (Levenshtein 거리): {SIMILARITY_THRESHOLD}")
    # print(f"유사/오인식 명령어 맵 로드 완료 (개수: {len(SIMILAR_COMMAND_MAP)})")

    # print(f"\n--- {RECORD_SECONDS}초 단위로 음성 인식을 시작합니다 ---")
    # print("마이크 입력을 기다리는 중...")

    # 마이크 스트림 설정 및 시작
    stream = sd.InputStream(
        samplerate=stt_module.sample_rate,
        channels=stt_module.channels,
        dtype=stt_module.dtype,
        callback=stt_module.audio_callback
    )
    stream.start()

    try:
        audio_buffer = np.array([], dtype=stt_module.dtype)
        non_silent_chunk_count = 0

        while True:
            # 큐에서 오디오 데이터 가져오기
            try:
                audio_chunk = stt_module.audio_queue.get(timeout=0.1) # 0.1초 대기
                audio_buffer = np.append(audio_buffer, audio_chunk)

                # 간단한 침묵 감지
                rms = stt_module.calculate_rms(audio_chunk)
                if rms > stt_module.silence_threshold:
                    non_silent_chunk_count += 1
                else:
                    # 침묵이 감지되면 카운터 리셋 (연속적인 소리만 처리하기 위함)
                    # non_silent_chunk_count = 0 # 너무 민감하게 반응할 수 있어 일단 주석처리
                    pass

            except queue.Empty:
                # 큐가 비어있으면 잠시 대기
                time.sleep(0.05)
                continue

            # 버퍼가 일정 시간 이상 채워졌고, 유효한 소리가 있었다면 처리
            buffer_duration = len(audio_buffer) / stt_module.sample_rate
            if buffer_duration >= stt_module.record_seconds:
                if non_silent_chunk_count >= stt_module.min_non_silent_chunks:
                    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] {buffer_duration:.2f}초 분량 오디오 처리 시작...")
                    audio_to_process = audio_buffer.copy()
                    # 버퍼 초기화 (다음 녹음을 위해)
                    audio_buffer = np.array([], dtype=stt_module.dtype)
                    non_silent_chunk_count = 0

                    # --- 음성 인식 ---
                    start_time = time.time() # 처리 시작 시간 기록

                    # language="ko" 로 지정하면 한국어 인식에 집중
                    # vad_filter=True 를 사용하면 음성 구간만 잘라내어 인식 (더 정확하고 빠름)
                    segments, info = stt_module.model.transcribe(
                        audio_to_process,
                        language="ko",
                        beam_size=1, # 속도 향상 (정확도 약간 저하 가능)
                        vad_filter=True, # VAD 필터 사용
                        vad_parameters=dict(min_silence_duration_ms=500) # VAD 파라미터 조절 가능
                    )

                    end_time = time.time() # 처리 종료 시간 기록
                    processing_duration = end_time - start_time # 처리 시간 계산

                    print(f"    언어 감지: {info.language} (확률: {info.language_probability:.2f})")
                    print(f"    처리 시간: {processing_duration:.2f}초") # 직접 계산한 처리 시간 출력
                    # info 객체에 VAD 관련 정보가 있다면 출력 (없으면 오류 발생 가능하므로 주의)
                    try:
                        print(f"    오디오 길이 (VAD 적용 후): {info.duration_after_vad:.2f}초")
                    except AttributeError:
                        print(f"    처리된 오디오 총 길이: {info.duration:.2f}초")


                    # --- 결과 처리 및 명령어 유도 ---
                    full_text = "".join([segment.text for segment in segments])
                    print(f"    인식된 텍스트: '{full_text}'")

                    if full_text:
                        # 수정된 함수 호출
                        matched_command = stt_module.find_closest_command(full_text)
                        if matched_command:
                            print(f"  >> 최종 인식 명령어: [ {matched_command} ]")
                            # TODO: 인식된 명령어를 사용하여 실제 동작 수행
                            # 예: if matched_command == "온열": turn_on_heating()
                        else:
                            print("  >> 유효한 명령어가 아닙니다.")
                    else:
                        print("    (음성 인식 결과 없음)")

                    print("\n마이크 입력을 기다리는 중...")

                else:
                    # 소리가 있었지만 너무 짧거나 작음, 버퍼 초기화
                    print(".", end="", flush=True) # 대기 중임을 시각적으로 표시
                    audio_buffer = np.array([], dtype=stt_module.dtype)
                    non_silent_chunk_count = 0


    except KeyboardInterrupt:
        print("\n프로그램 종료 중...")
    finally:
        if 'stream' in locals() and stream.active:
            stream.stop()
            stream.close()
        print("마이크 스트림 종료.")

