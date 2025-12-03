#!/bin/bash

# ========================================
#   Auto Training & Evaluation Runner
# ========================================

# 항상 스크립트가 있는 yolov1 디렉토리에서 실행되도록 설정
cd "$(dirname "$0")"

# 로그 저장 폴더 생성
mkdir -p logs

# main → eval 파일 매핑
declare -A JOBS=(
    ["main_adabelief.py"]="eval_adabelief.py"
    ["main_adamw.py"]="eval_adamw.py"
    ["main_Alpha_IoU.py"]="eval_alpha_iou.py"
    ["main_Focal_Alpha_DIOU_Adamw.py"]="eval_Focal_Alpha_DIoU_Adamw.py"
    ["main_radam.py"]="eval_radam.py"
)

echo "==== Auto Training & Evaluation Start ===="

# 모든 job 실행
for MAIN in "${!JOBS[@]}"; do
    EVAL=${JOBS[$MAIN]}

    # focal 관련 파일은 제외 (이미 제외한다고 했음)
    if [[ "$MAIN" == "main_focal.py" ]] || [[ "$EVAL" == "eval_focal.py" ]]; then
        continue
    fi

    echo "--------------------------------------"
    echo " Running: $MAIN"
    echo "--------------------------------------"

    # main 실행 → 로그 저장
    python3 "$MAIN" > "logs/${MAIN%.py}.log" 2>&1

    echo "Finished $MAIN"
    echo "Running Evaluation: $EVAL"

    # eval 실행 → 로그 저장
    python3 "$EVAL" > "logs/${EVAL%.py}.log" 2>&1

    echo "Finished $EVAL"
done

echo "==== All Jobs Completed ===="
