# temp_AI_flow

```
├── .gitignore
├── README.md
├── dedup/
│   ├── dedup.py
│   ├── dedup_test.py
│   ├── save_image.py
│   └── tnv2.py
├── requirements.txt
├── run/
│   ├── check/
│   ├── english/
│   │   ├── qa1.txt
│   │   ├── qa2.txt
│   │   └── qa3.txt
│   ├── question/
│   │   ├── qa1.txt
│   │   ├── qa2.txt
│   │   └── qa3.txt
│   └── result/
├── sample/
│   └── joy.jpg
├── scripts/
│   └── create_db.sh
├── src/
│   ├── embeding/
│   │   ├── __init__.py
│   │   ├── __pycache__/
│   │   ├── embed_utils.py
│   │   └── onnx_jina.py
│   ├── main_create.py
│   ├── modeling/
│   │   ├── __init__.py
│   │   ├── example.txt
│   │   └── qwen_model.py
│   ├── processor.py
│   ├── run.ipynb
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── image_utils.py
│   │   └── text_utils.py
│   └── vector_store/
│       ├── README.md
│       ├── __init__.py
│       ├── __pycache__/
│       ├── database_sql.py
│       └── qdrant.py
└── weights/
    ├── .gitignore
    ├── jina-clip-v1/
    │   ├── config.json
    │   ├── config_sentence_transformers.json
    │   ├── custom_st.py
    │   ├── modules.json
    │   ├── onnx/
    │   ├── preprocessor_config.json
    │   ├── special_tokens_map.json
    │   ├── tokenizer.json
    │   ├── tokenizer_config.json
    │   └── vocab.txt
    └── snapshot/
        └── qdrant_snapshot.snapshot

```

Link snapshot: [Here]([https://drive.google.com/file/d/1ycwgkKNY6fV_viz-qTvJWB8telohDRCu/view?usp=drive_link])
Link weight JINA: [Here]([https://huggingface.co/jinaai/jina-clip-v1/tree/main])
