python ./src/main_create.py \
    --collection_name "qdrant" \
    --collection_type "image" \
    --similarity_metric "cosine" \
    --dataset_name \
    "wanhin/L1-224-224-step-2-deduped" \
    "wanhin/L2-224-224-step-2-deduped" \
    "wanhin/L3-224-224-step-2-deduped" \
    "wanhin/L4-224-224-step-2-deduped" \
    "wanhin/L5-224-224-step-2-deduped" \
    "wanhin/L6-224-224-step-2-deduped" \
    "wanhin/L7-224-224-step-2-deduped" \
    "wanhin/L8-224-224-step-2-deduped" \
    "wanhin/L9-224-224-step-2-deduped" \
    "wanhin/L10-224-224-step-2-deduped" \
    "wanhin/L11-224-224-step-2-deduped" \
    "wanhin/L12-224-224-step-2-deduped" \
    --cache_dir ".cache" \
    --batch_size 50 \
    --num_proc 1
