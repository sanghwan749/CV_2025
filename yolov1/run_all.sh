#!/bin/bash

# 로그 저장 폴더 생성
mkdir -p logs

# main 과 eval 파일 쌍 정의
declare -A JOBS=(
    ["main_adabelief.py"]="eval_adabelief.py"
    ["main_adamw.py"]="eval_adamw.py"
    ["main_Alpha_IoU.py"]="eval_alpha_iou.py"
    ["main_Focal_Alpha_DIoU_Adamw.py"]="eval_Focal_Alpha_DIoU_Adamw.py"
    ["main_radam.py"]="eval_radam.py"
)

echo "==== Auto Training & Evaluation Start ===="

# 파일 루프 돌기
for MAIN in "${!JOBS[@]}"; do
    EVAL=${JOBS[$MAIN]}

    # 제외할 focal 관련 파일은 스킵
    if [[ "$MAIN" == "main_focal.py" ]] || [[ "$EVAL" == "eval_focal.py" ]]; then
        continue
    fi

    echo "--------------------------------------"
    echo "Running: $MAIN"
    echo "--------------------------------------"

    # main 실행
    python3 "$MAIN" > "logs/${MAIN%.py}.log" 2>&1

    echo "Finished $MAIN"
    echo "Running Evaluation: $EVAL"

    # eval 실행
    python3 "$EVAL" > "logs/${EVAL%.py}.log" 2>&1

    echo "Finished $EVAL"
done

echo "==== All Jobs Completed ===="
