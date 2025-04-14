import cv2
import mediapipe as mp
import numpy as np

# Selecionando o modelo de detecção que queremos utilizar
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils  # objeto que vai permitir desenhar na tela

capture = cv2.VideoCapture(r'./checkpoint2_IA/sinais_mao.mp4')

while True:
    ret, frame = capture.read()
    if not ret:
        print("Fim do vídeo")  # Volta ao início do vídeo
        break

    # Redimensiona o vídeo para 500x500
    frame = cv2.resize(frame, (1000, 1000))

    rgb_capture = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_capture)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # instanciando os pontos que utilizaremos
            indicador = hand_landmarks.landmark[8]
            anelar = hand_landmarks.landmark[16]
            polegar = hand_landmarks.landmark[4]
            pulso = hand_landmarks.landmark[0]

            # obtendo a distancia entre indicador e polegar
            distancia_indicador_polegar_y = polegar.y - indicador.y
            distancia_indicador_polegar_x = polegar.x - indicador.x

            # obtendo a distancia entre polegar e pulso
            distancia_polegar_pulso_y = pulso.y - polegar.y
            distancia_polegar_pulso_x = pulso.x - polegar.x

            # obtendo a distancia entre anelar e pulso (somente vertical)
            distancia_anelar_pulso_y = pulso.y - anelar.y

            # verificações/limites que os pontos devem atender para serem considerados o "L" (fizemos para facilitar a visão do "IF")
            verificacao_indicador_polegar = (
                distancia_indicador_polegar_x > 0.27
                and distancia_indicador_polegar_y > 0.28
            )
            verificacao_polegar_pulso = (
                distancia_polegar_pulso_x > -0.57 and distancia_polegar_pulso_y > 0.140
            )
            verificacao_anelar_pulso = distancia_anelar_pulso_y < 0.16

            # Condição de fez ou não o "L"
            l = "Nao fez o L"
            if (
                verificacao_indicador_polegar
                and verificacao_polegar_pulso
                and verificacao_anelar_pulso
            ):
                l = "Fez o L"

            cv2.putText(
                frame,
                f"status: {l}",
                (5, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0),
                2,
            )
            cor = (0, 255, 0) if l == "Fez o L" else (0, 0, 255)
            cv2.putText(frame, f"L", (60, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, cor, 5)

            # cv2.putText(frame, f"d: ({distancia_anelar_pulso_y})", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            # cv2.putText(frame, f"Indicador: ({dista}, {indicador.y})", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            # cv2.putText(frame, f"polegar: ({polegar.x}, {polegar.y})", (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            # cv2.putText(frame, f"pulso: ({pulso.x}, {pulso.y})", (5, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    cv2.imshow("Video", frame)

    # Em tese, esse trecho de código sincroniza a taxa de quadros (??) mas sem ele, o vídeo não aparece.
    if cv2.waitKey(30) & 0xFF == ord("q"):
        break


capture.release()  # Libera o objeto do vídeo inicial
cv2.destroyAllWindows()  # Destrói todas as janelas do opencv
