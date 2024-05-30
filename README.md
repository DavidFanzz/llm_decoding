# Code For A Thorough Examination of Decoding Methods in the Era of LLMs

### Install Requirements
```
pip install datasets
pip install evaluate
pip install git+https://github.com/hendrycks/math.git
pip install absl-py
pip install nltk
pip install antlr4-python3-runtime==4.11.1
pip install transformers --upgrade
pip install flash_attn
pip install torch_scatter
```

### Generation
Take Contrastive Search (CS) For Example
```
model_name=Llama2
decoding_method=cs
cs_alpha=0.1
task=gsm8k
model_path=/mnt/data/models/${model_name}
python3 generate.py \
    --decoding_method ${decoding_method}\
    --infile ./data/${task}/${model_name}_input.jsonl\
    --outfile ./results/${task}/${model_name}_${decoding_method}_${cs_alpha}.jsonl\
    --model ${model_path}\
    --gpus_per_model 1\
    --world_size 4\
    --batch_size 1\
    --max_new_tokens 512\
    --cs_alpha ${cs_alpha}
```

### Evaluation
```
python3 evaluation.py\
    --model ${model_name}\
    --task_name ${task}\
    --load_generations_path ./results/${task}/${model_name}_${decoding_method}.jsonl\
    --metric_output_path ./results/${task}_results.jsonl\
    --allow_code_execution
```
