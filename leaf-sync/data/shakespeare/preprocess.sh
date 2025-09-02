# Decide se precisamos (re)gerar all_data.json a partir do raw
NEED_JSON=false
if [[ ! -f "data/all_data/all_data.json" ]]; then
  NEED_JSON=true
elif [[ ! -s "data/all_data/all_data.json" || $(stat -c%s "data/all_data/all_data.json") -lt 1024 ]]; then
  NEED_JSON=true
elif [[ ! -d "data/raw_data/by_play_and_character" ]]; then
  NEED_JSON=true
elif [[ -z "$(ls -A data/raw_data/by_play_and_character 2>/dev/null || true)" ]]; then
  NEED_JSON=true
fi

if $NEED_JSON; then
  echo "[INFO] (re)gerando all_data.json a partir do raw_data..."
  pushd preprocess >/dev/null
    bash -x ./data_to_json.sh
  popd >/dev/null
else
  echo "[INFO] all_data.json e by_play_and_character parecem OK — pulando data_to_json."
fi
