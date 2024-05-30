# model_names=(Meta-Llama-3-8B)
# tasks=(wikinews)

# decoding_method=beam
# beam_ns=(4 8)

# for model_name in ${model_names[@]}; do
#     for task in ${tasks[@]}; do
#         model_path=/mnt/data/scf22/big_model/${model_name}
#         for beam_n in ${beam_ns[@]}; do
#             python3 generate.py \
#                 --decoding_method ${decoding_method}\
#                 --infile ./data/${task}/${model_name}_input.jsonl\
#                 --outfile ./results/${task}/${model_name}_${decoding_method}_${beam_n}.jsonl\
#                 --model ${model_path}\
#                 --gpus_per_model 1\
#                 --world_size 4\
#                 --batch_size 1\
#                 --max_new_tokens 512\
#                 --begin_gpu 3\
#                 --beam_n ${beam_n}
#         done
#     done
# done

# decoding_method=diverse_beam


# for model_name in ${model_names[@]}; do
#     for task in ${tasks[@]}; do
#         model_path=/mnt/data/scf22/big_model/${model_name}
#         diverse_beam_n=4
#         diverse_beam_groups=4
#         python3 generate.py \
#             --decoding_method ${decoding_method}\
#             --infile ./data/${task}/${model_name}_input.jsonl\
#             --outfile ./results/${task}/${model_name}_${decoding_method}_${diverse_beam_n}_${diverse_beam_groups}.jsonl\
#             --model ${model_path}\
#             --gpus_per_model 1\
#             --world_size 4\
#             --batch_size 1\
#             --max_new_tokens 512\
#             --begin_gpu 3\
#             --diverse_beam_n ${diverse_beam_n}\
#             --diverse_beam_groups ${diverse_beam_groups}
        
#         diverse_beam_n=4
#         diverse_beam_groups=2
#         python3 generate.py \
#             --decoding_method ${decoding_method}\
#             --infile ./data/${task}/${model_name}_input.jsonl\
#             --outfile ./results/${task}/${model_name}_${decoding_method}_${diverse_beam_n}_${diverse_beam_groups}.jsonl\
#             --model ${model_path}\
#             --gpus_per_model 1\
#             --world_size 4\
#             --batch_size 1\
#             --max_new_tokens 512\
#             --begin_gpu 3\
#             --diverse_beam_n ${diverse_beam_n}\
#             --diverse_beam_groups ${diverse_beam_groups}
        
#         diverse_beam_n=8
#         diverse_beam_groups=4
#         python3 generate.py \
#             --decoding_method ${decoding_method}\
#             --infile ./data/${task}/${model_name}_input.jsonl\
#             --outfile ./results/${task}/${model_name}_${decoding_method}_${diverse_beam_n}_${diverse_beam_groups}.jsonl\
#             --model ${model_path}\
#             --gpus_per_model 1\
#             --world_size 4\
#             --batch_size 1\
#             --max_new_tokens 512\
#             --begin_gpu 3\
#             --diverse_beam_n ${diverse_beam_n}\
#             --diverse_beam_groups ${diverse_beam_groups}

#         diverse_beam_n=8
#         diverse_beam_groups=2
#         python3 generate.py \
#             --decoding_method ${decoding_method}\
#             --infile ./data/${task}/${model_name}_input.jsonl\
#             --outfile ./results/${task}/${model_name}_${decoding_method}_${diverse_beam_n}_${diverse_beam_groups}.jsonl\
#             --model ${model_path}\
#             --gpus_per_model 1\
#             --world_size 4\
#             --batch_size 1\
#             --max_new_tokens 512\
#             --begin_gpu 3\
#             --diverse_beam_n ${diverse_beam_n}\
#             --diverse_beam_groups ${diverse_beam_groups}
#     done
# done

model_names=(Meta-Llama-3-8B)
tasks=(gsm8k mbpp)
decoding_method=mirostat
mirostat_taus=(2.5 3.0 4.0 5.0)

for model_name in ${model_names[@]}; do
    for task in ${tasks[@]}; do
        model_path=/mnt/data/scf22/big_model/${model_name}
        for mirostat_tau in ${mirostat_taus[@]}; do
            python3 generate.py \
                --decoding_method ${decoding_method}\
                --infile ./data/${task}/${model_name}_input.jsonl\
                --outfile ./results/${task}/${model_name}_${decoding_method}_${mirostat_tau}.jsonl\
                --model ${model_path}\
                --gpus_per_model 1\
                --world_size 4\
                --batch_size 1\
                --max_new_tokens 512\
                --begin_gpu 3\
                --mirostat_tau ${mirostat_tau}
        done
    done
done




decoding_method=cs
cs_alphas=(0.1 0.2 0.3 0.4 0.5 0.6)

for model_name in ${model_names[@]}; do
    for task in ${tasks[@]}; do
        model_path=/mnt/data/scf22/big_model/${model_name}
        for cs_alpha in ${cs_alphas[@]}; do
            python3 generate.py \
                --decoding_method ${decoding_method}\
                --infile ./data/${task}/${model_name}_input.jsonl\
                --outfile ./results/${task}/${model_name}_${decoding_method}_${cs_alpha}.jsonl\
                --model ${model_path}\
                --gpus_per_model 1\
                --world_size 4\
                --batch_size 1\
                --max_new_tokens 512\
                --begin_gpu 3\
                --cs_alpha ${cs_alpha}
        done
    done
done


decoding_method=fsd
fsd_alphas=(0.1 0.2 0.3 0.4 0.5 0.6)

for model_name in ${model_names[@]}; do
    for task in ${tasks[@]}; do
        model_path=/mnt/data/scf22/big_model/${model_name}
        for fsd_alpha in ${fsd_alphas[@]}; do
            python3 generate.py \
                --decoding_method ${decoding_method}\
                --infile ./data/${task}/${model_name}_input.jsonl\
                --outfile ./results/${task}/${model_name}_${decoding_method}_${fsd_alpha}.jsonl\
                --model ${model_path}\
                --gpus_per_model 1\
                --world_size 4\
                --batch_size 1\
                --max_new_tokens 512\
                --begin_gpu 3\
                --fsd_alpha ${fsd_alpha}
        done
    done
done

decoding_method=fsd-d
fsd_d_alphas=(0.1 0.2 0.3 0.4 0.5 0.6)

for model_name in ${model_names[@]}; do
    for task in ${tasks[@]}; do
        model_path=/mnt/data/scf22/big_model/${model_name}
        for fsd_d_alpha in ${fsd_d_alphas[@]}; do
            python3 generate.py \
                --decoding_method ${decoding_method}\
                --infile ./data/${task}/${model_name}_input.jsonl\
                --outfile ./results/${task}/${model_name}_${decoding_method}_${fsd_d_alpha}.jsonl\
                --model ${model_path}\
                --gpus_per_model 1\
                --world_size 4\
                --batch_size 1\
                --max_new_tokens 512\
                --begin_gpu 3\
                --fsd_d_alpha ${fsd_d_alpha}
        done
    done
done

decoding_method=typical
typical_ps=(0.2 0.9 0.92 0.95)

for model_name in ${model_names[@]}; do
    for task in ${tasks[@]}; do
        model_path=/mnt/data/scf22/big_model/${model_name}
        for typical_p in ${typical_ps[@]}; do
            python3 generate.py \
                --decoding_method ${decoding_method}\
                --infile ./data/${task}/${model_name}_input.jsonl\
                --outfile ./results/${task}/${model_name}_${decoding_method}_${typical_p}.jsonl\
                --model ${model_path}\
                --gpus_per_model 1\
                --world_size 4\
                --batch_size 1\
                --max_new_tokens 512\
                --begin_gpu 3\
                --typical_p ${typical_p}
        done
    done
done

decoding_method=eta
eta_cutoffs=(0.0003 0.0006 0.0009 0.002 0.004)

for model_name in ${model_names[@]}; do
    for task in ${tasks[@]}; do
        model_path=/mnt/data/scf22/big_model/${model_name}
        for eta_cutoff in ${eta_cutoffs[@]}; do
            python3 generate.py \
                --decoding_method ${decoding_method}\
                --infile ./data/${task}/${model_name}_input.jsonl\
                --outfile ./results/${task}/${model_name}_${decoding_method}_${eta_cutoff}.jsonl\
                --model ${model_path}\
                --gpus_per_model 1\
                --world_size 4\
                --batch_size 1\
                --max_new_tokens 512\
                --begin_gpu 3\
                --eta_cutoff ${eta_cutoff}
        done
    done
done

decoding_method=beam
beam_ns=(4 8)

for model_name in ${model_names[@]}; do
    for task in ${tasks[@]}; do
        model_path=/mnt/data/scf22/big_model/${model_name}
        for beam_n in ${beam_ns[@]}; do
            python3 generate.py \
                --decoding_method ${decoding_method}\
                --infile ./data/${task}/${model_name}_input.jsonl\
                --outfile ./results/${task}/${model_name}_${decoding_method}_${beam_n}.jsonl\
                --model ${model_path}\
                --gpus_per_model 1\
                --world_size 4\
                --batch_size 1\
                --max_new_tokens 512\
                --begin_gpu 3\
                --beam_n ${beam_n}
        done
    done
done

decoding_method=diverse_beam


for model_name in ${model_names[@]}; do
    for task in ${tasks[@]}; do
        model_path=/mnt/data/scf22/big_model/${model_name}
        diverse_beam_n=4
        diverse_beam_groups=4
        python3 generate.py \
            --decoding_method ${decoding_method}\
            --infile ./data/${task}/${model_name}_input.jsonl\
            --outfile ./results/${task}/${model_name}_${decoding_method}_${diverse_beam_n}_${diverse_beam_groups}.jsonl\
            --model ${model_path}\
            --gpus_per_model 1\
            --world_size 4\
            --batch_size 1\
            --max_new_tokens 512\
            --begin_gpu 3\
            --diverse_beam_n ${diverse_beam_n}\
            --diverse_beam_groups ${diverse_beam_groups}
        
        diverse_beam_n=4
        diverse_beam_groups=2
        python3 generate.py \
            --decoding_method ${decoding_method}\
            --infile ./data/${task}/${model_name}_input.jsonl\
            --outfile ./results/${task}/${model_name}_${decoding_method}_${diverse_beam_n}_${diverse_beam_groups}.jsonl\
            --model ${model_path}\
            --gpus_per_model 1\
            --world_size 4\
            --batch_size 1\
            --max_new_tokens 512\
            --begin_gpu 3\
            --diverse_beam_n ${diverse_beam_n}\
            --diverse_beam_groups ${diverse_beam_groups}
        
        diverse_beam_n=8
        diverse_beam_groups=4
        python3 generate.py \
            --decoding_method ${decoding_method}\
            --infile ./data/${task}/${model_name}_input.jsonl\
            --outfile ./results/${task}/${model_name}_${decoding_method}_${diverse_beam_n}_${diverse_beam_groups}.jsonl\
            --model ${model_path}\
            --gpus_per_model 1\
            --world_size 4\
            --batch_size 1\
            --max_new_tokens 512\
            --begin_gpu 3\
            --diverse_beam_n ${diverse_beam_n}\
            --diverse_beam_groups ${diverse_beam_groups}

        diverse_beam_n=8
        diverse_beam_groups=2
        python3 generate.py \
            --decoding_method ${decoding_method}\
            --infile ./data/${task}/${model_name}_input.jsonl\
            --outfile ./results/${task}/${model_name}_${decoding_method}_${diverse_beam_n}_${diverse_beam_groups}.jsonl\
            --model ${model_path}\
            --gpus_per_model 1\
            --world_size 4\
            --batch_size 1\
            --max_new_tokens 512\
            --begin_gpu 3\
            --diverse_beam_n ${diverse_beam_n}\
            --diverse_beam_groups ${diverse_beam_groups}
    done
done
