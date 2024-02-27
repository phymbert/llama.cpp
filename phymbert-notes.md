# TODO

test U server example
fix log-disable
add /models
helm example
empty input test



      initContainers:
        - name: wait-model
          securityContext:
            {{- toYaml .Values.securityContext | nindent 12 }}
          image: {{ .Values.images.downloader.repository }}:{{ .Values.images.downloader.name }}-{{ .Values.images.downloader.tag | default .Chart.Version }}
          env:
            - name: MODEL_PATH
              value: {{ .Values.model.path }}
            - name: MODEL_FILE
              value: {{ .Values.model.file_basename }}
            - name: MODEL_SHA256
              value: {{ .Values.model.sha256 }}
            - name: MODEL_DOWNLOAD_FILE
              value: {{ .Values.model.file }}
          command:
            - /bin/bash
            - -c
          args:
            - >
              if [ ! echo "${MODEL_SHA256} *${MODEL_PATH}/${MODEL_FILE}" | sha --algorithm 256 -c ]; then
                wget -q --show-progress -c -O ${MODEL_PATH}/${MODEL_FILE} https://huggingface.co/${MODEL_PATH}/resolve/main/${MODEL_DOWNLOAD_FILE}
              fi