from script.datapre import *
from utils.logger import Logger
import datetime
import argparse
import config



if __name__ == "__main__":
    # input args
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--topic", help="topic", required=False)
    parser.add_argument("-m", "--method", type=identify_method, required=False)
    parser.add_argument("--dataset", help="dataset", required=False)
    parser.add_argument("--model", required=False)
    parser.add_argument("-a", "--alpha", type=float, required=False)
    parser.add_argument("--distribution", required=False)
    parser.add_argument("-s", "--seed", type=int, required=False)
    parser.add_argument("--num_repeat", type=int, required=False)
    parser.add_argument("--start_date", required=True)
    parser.add_argument("--num_try", type=str, required=True)
    parser.add_argument("--attack", required=False)
    parser.add_argument("--value_functions", nargs="+", type=str, help="a list of value functions")

    args = parser.parse_args()

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
