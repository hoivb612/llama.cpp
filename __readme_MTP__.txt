D:\llama.cpp\b612_052026_mtp\build>bin\RelWithDebInfo\minslm-cli.exe d:\llama.cpp\models\gemma-4\gemma-4-E2B-it-Q4_K_M.gguf 8 ..\examples\llm-infer\prompts\single_prompt_gemma4.txt v2 --mtp-head d:\llama.cpp\models\gemma-4\gemma-4-E2B-it-assistant.F16.gguf --spec-type mtp --draft-block-size 3 repack-xbox
Warning: unknown argument '--spec-type' ignored
Warning: unknown argument 'mtp' ignored
Warning: unknown argument 'repack-xbox' ignored
[main]: loaded 1 prompt(s) from '..\examples\llm-infer\prompts\single_prompt_gemma4.txt'
llama_model_loader: direct I/O is enabled, disabling mmap
llama_model_loader: loaded meta data with 56 key-value pairs and 601 tensors from d:\llama.cpp\models\gemma-4\gemma-4-E2B-it-Q4_K_M.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = gemma4
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                     general.sampling.top_k i32              = 64
llama_model_loader: - kv   3:                     general.sampling.top_p f32              = 0.950000
llama_model_loader: - kv   4:                      general.sampling.temp f32              = 1.000000
llama_model_loader: - kv   5:                               general.name str              = Gemma-4-E2B-It
llama_model_loader: - kv   6:                           general.basename str              = Gemma-4-E2B-It
llama_model_loader: - kv   7:                       general.quantized_by str              = Unsloth
llama_model_loader: - kv   8:                         general.size_label str              = 4.6B
llama_model_loader: - kv   9:                            general.license str              = apache-2.0
llama_model_loader: - kv  10:                       general.license.link str              = https://ai.google.dev/gemma/docs/gemm...
llama_model_loader: - kv  11:                           general.repo_url str              = https://huggingface.co/unsloth
llama_model_loader: - kv  12:                   general.base_model.count u32              = 1
llama_model_loader: - kv  13:                  general.base_model.0.name str              = Gemma 4 E2B It
llama_model_loader: - kv  14:          general.base_model.0.organization str              = Google
llama_model_loader: - kv  15:              general.base_model.0.repo_url str              = https://huggingface.co/google/gemma-4...
llama_model_loader: - kv  16:                               general.tags arr[str,2]       = ["unsloth", "any-to-any"]
llama_model_loader: - kv  17:                         gemma4.block_count u32              = 35
llama_model_loader: - kv  18:                      gemma4.context_length u32              = 131072
llama_model_loader: - kv  19:                    gemma4.embedding_length u32              = 1536
llama_model_loader: - kv  20:                 gemma4.feed_forward_length arr[i32,35]      = [6144, 6144, 6144, 6144, 6144, 6144, ...
llama_model_loader: - kv  21:                gemma4.attention.head_count u32              = 8
llama_model_loader: - kv  22:             gemma4.attention.head_count_kv u32              = 1
llama_model_loader: - kv  23:                      gemma4.rope.freq_base f32              = 1000000.000000
llama_model_loader: - kv  24:                  gemma4.rope.freq_base_swa f32              = 10000.000000
llama_model_loader: - kv  25:    gemma4.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  26:                gemma4.attention.key_length u32              = 512
llama_model_loader: - kv  27:              gemma4.attention.value_length u32              = 512
llama_model_loader: - kv  28:             gemma4.final_logit_softcapping f32              = 30.000000
llama_model_loader: - kv  29:            gemma4.attention.sliding_window u32              = 512
llama_model_loader: - kv  30:          gemma4.attention.shared_kv_layers u32              = 20
llama_model_loader: - kv  31:    gemma4.embedding_length_per_layer_input u32              = 256
llama_model_loader: - kv  32:    gemma4.attention.sliding_window_pattern arr[bool,35]     = [true, true, true, true, false, true,...
llama_model_loader: - kv  33:            gemma4.attention.key_length_swa u32              = 256
llama_model_loader: - kv  34:          gemma4.attention.value_length_swa u32              = 256
llama_model_loader: - kv  35:                gemma4.rope.dimension_count u32              = 512
llama_model_loader: - kv  36:            gemma4.rope.dimension_count_swa u32              = 256
llama_model_loader: - kv  37:                       tokenizer.ggml.model str              = gemma4
llama_model_loader: - kv  38:                      tokenizer.ggml.tokens arr[str,262144]  = ["<pad>", "<eos>", "<bos>", "<unk>", ...
llama_model_loader: - kv  39:                      tokenizer.ggml.scores arr[f32,262144]  = [-1000.000000, -1000.000000, -1000.00...
llama_model_loader: - kv  40:                  tokenizer.ggml.token_type arr[i32,262144]  = [3, 1, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  41:                      tokenizer.ggml.merges arr[str,514906]  = ["\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n \n", ...
llama_model_loader: - kv  42:                tokenizer.ggml.bos_token_id u32              = 2
llama_model_loader: - kv  43:                tokenizer.ggml.eos_token_id u32              = 106
llama_model_loader: - kv  44:            tokenizer.ggml.unknown_token_id u32              = 3
llama_model_loader: - kv  45:            tokenizer.ggml.padding_token_id u32              = 0
llama_model_loader: - kv  46:               tokenizer.ggml.mask_token_id u32              = 4
llama_model_loader: - kv  47:                    tokenizer.chat_template str              = {%- macro format_parameters(propertie...
llama_model_loader: - kv  48:            tokenizer.ggml.add_space_prefix bool             = false
llama_model_loader: - kv  49:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  50:               general.quantization_version u32              = 2
llama_model_loader: - kv  51:                          general.file_type u32              = 15
llama_model_loader: - kv  52:                      quantize.imatrix.file str              = gemma-4-E2B-it-GGUF/imatrix_unsloth.gguf
llama_model_loader: - kv  53:                   quantize.imatrix.dataset str              = unsloth_calibration_gemma-4-E2B-it.txt
llama_model_loader: - kv  54:             quantize.imatrix.entries_count u32              = 275
llama_model_loader: - kv  55:              quantize.imatrix.chunks_count u32              = 141
llama_model_loader: - type  f32:  353 tensors
llama_model_loader: - type q4_K:  212 tensors
llama_model_loader: - type q5_K:    1 tensors
llama_model_loader: - type q6_K:   34 tensors
llama_model_loader: - type bf16:    1 tensors
print_info: file format = GGUF V3 (latest)
print_info: file type   = Q4_K - Medium
print_info: file size   = 2.88 GiB (5.32 BPW)
init_tokenizer: initializing tokenizer for type 2
load: 0 unused tokens
load: control-looking token:      1 '<eos>' was not control-type; this is probably a bug in the model. its type will be overridden
load: control token:      0 '<pad>' is not marked as EOG
load: control token:      3 '<unk>' is not marked as EOG
load: control token:      2 '<bos>' is not marked as EOG
load: control token:      4 '<mask>' is not marked as EOG
load: control token:     47 '<tool|>' is not marked as EOG
load: control token:     98 '<|think|>' is not marked as EOG
load: control token:     46 '<|tool>' is not marked as EOG
load: control-looking token:     50 '<|tool_response>' was not control-type; this is probably a bug in the model. its type will be overridden
load: control token:    105 '<|turn>' is not marked as EOG
load: control-looking token:    212 '</s>' was not control-type; this is probably a bug in the model. its type will be overridden
load: control token: 258880 '<|image|>' is not marked as EOG
load: control token: 258882 '<image|>' is not marked as EOG
load: control token: 256000 '<|audio>' is not marked as EOG
load: control token: 258883 '<audio|>' is not marked as EOG
load: control token: 258884 '<|video|>' is not marked as EOG
load: control token: 255999 '<|image>' is not marked as EOG
load: control token: 258881 '<|audio|>' is not marked as EOG
load: printing all EOG tokens:
load:   - 1 ('<eos>')
load:   - 50 ('<|tool_response>')
load:   - 106 ('<turn|>')
load:   - 212 ('</s>')
load: special_eog_ids contains '<|tool_response>', removing '</s>' token from EOG list
load: special tokens cache size = 24
load: token to piece cache size = 1.9445 MB
print_info: arch                  = gemma4
print_info: vocab_only            = 0
print_info: no_alloc              = 0
print_info: n_ctx_train           = 131072
print_info: n_embd                = 1536
print_info: n_embd_inp            = 1536
print_info: n_layer               = 35
print_info: n_head                = 8
print_info: n_head_kv             = 1
print_info: n_rot                 = 512
print_info: n_swa                 = 512
print_info: is_swa_any            = 1
print_info: n_embd_head_k         = 512
print_info: n_embd_head_v         = 512
print_info: n_gqa                 = 8
print_info: n_embd_k_gqa          = [256, 256, 256, 256, 512, 256, 256, 256, 256, 512, 256, 256, 256, 256, 512, 256, 256, 256, 256, 512, 256, 256, 256, 256, 512, 256, 256, 256, 256, 512, 256, 256, 256, 256, 512]
print_info: n_embd_v_gqa          = [256, 256, 256, 256, 512, 256, 256, 256, 256, 512, 256, 256, 256, 256, 512, 256, 256, 256, 256, 512, 256, 256, 256, 256, 512, 256, 256, 256, 256, 512, 256, 256, 256, 256, 512]
print_info: f_norm_eps            = 0.0e+00
print_info: f_norm_rms_eps        = 1.0e-06
print_info: f_clamp_kqv           = 0.0e+00
print_info: f_max_alibi_bias      = 0.0e+00
print_info: f_logit_scale         = 0.0e+00
print_info: f_attn_scale          = 1.0e+00
print_info: f_attn_value_scale    = 0.0000
print_info: n_ff                  = [6144, 6144, 6144, 6144, 6144, 6144, 6144, 6144, 6144, 6144, 6144, 6144, 6144, 6144, 6144, 12288, 12288, 12288, 12288, 12288, 12288, 12288, 12288, 12288, 12288, 12288, 12288, 12288, 12288, 12288, 12288, 12288, 12288, 12288, 12288]
print_info: n_expert              = 0
print_info: n_expert_used         = 0
print_info: n_expert_groups       = 0
print_info: n_group_used          = 0
print_info: causal attn           = 1
print_info: pooling type          = -1
print_info: rope type             = 2
print_info: rope scaling          = linear
print_info: freq_base_train       = 1000000.0
print_info: freq_scale_train      = 1
print_info: freq_base_swa         = 10000.0
print_info: freq_scale_swa        = 1
print_info: n_embd_head_k_swa     = 256
print_info: n_embd_head_v_swa     = 256
print_info: n_rot_swa             = 256
print_info: n_ctx_orig_yarn       = 131072
print_info: rope_yarn_log_mul     = 0.0000
print_info: rope_finetuned        = unknown
print_info: model type            = E2B
print_info: model params          = 4.65 B
print_info: general.name          = Gemma-4-E2B-It
print_info: vocab type            = BPE
print_info: n_vocab               = 262144
print_info: n_merges              = 514906
print_info: BOS token             = 2 '<bos>'
print_info: EOS token             = 106 '<turn|>'
print_info: UNK token             = 3 '<unk>'
print_info: PAD token             = 0 '<pad>'
print_info: MASK token            = 4 '<mask>'
print_info: LF token              = 107 '
'
print_info: EOG token             = 1 '<eos>'
print_info: EOG token             = 50 '<|tool_response>'
print_info: EOG token             = 106 '<turn|>'
print_info: max token length      = 93
load_tensors: loading model tensors, this can take a while... (mmap = false, direct_io = true)
load_tensors: layer   0 assigned to device CPU, is_swa = 1
load_tensors: layer   1 assigned to device CPU, is_swa = 1
load_tensors: layer   2 assigned to device CPU, is_swa = 1
load_tensors: layer   3 assigned to device CPU, is_swa = 1
load_tensors: layer   4 assigned to device CPU, is_swa = 0
load_tensors: layer   5 assigned to device CPU, is_swa = 1
load_tensors: layer   6 assigned to device CPU, is_swa = 1
load_tensors: layer   7 assigned to device CPU, is_swa = 1
load_tensors: layer   8 assigned to device CPU, is_swa = 1
load_tensors: layer   9 assigned to device CPU, is_swa = 0
load_tensors: layer  10 assigned to device CPU, is_swa = 1
load_tensors: layer  11 assigned to device CPU, is_swa = 1
load_tensors: layer  12 assigned to device CPU, is_swa = 1
load_tensors: layer  13 assigned to device CPU, is_swa = 1
load_tensors: layer  14 assigned to device CPU, is_swa = 0
load_tensors: layer  15 assigned to device CPU, is_swa = 1
load_tensors: layer  16 assigned to device CPU, is_swa = 1
load_tensors: layer  17 assigned to device CPU, is_swa = 1
load_tensors: layer  18 assigned to device CPU, is_swa = 1
load_tensors: layer  19 assigned to device CPU, is_swa = 0
load_tensors: layer  20 assigned to device CPU, is_swa = 1
load_tensors: layer  21 assigned to device CPU, is_swa = 1
load_tensors: layer  22 assigned to device CPU, is_swa = 1
load_tensors: layer  23 assigned to device CPU, is_swa = 1
load_tensors: layer  24 assigned to device CPU, is_swa = 0
load_tensors: layer  25 assigned to device CPU, is_swa = 1
load_tensors: layer  26 assigned to device CPU, is_swa = 1
load_tensors: layer  27 assigned to device CPU, is_swa = 1
load_tensors: layer  28 assigned to device CPU, is_swa = 1
load_tensors: layer  29 assigned to device CPU, is_swa = 0
load_tensors: layer  30 assigned to device CPU, is_swa = 1
load_tensors: layer  31 assigned to device CPU, is_swa = 1
load_tensors: layer  32 assigned to device CPU, is_swa = 1
load_tensors: layer  33 assigned to device CPU, is_swa = 1
load_tensors: layer  34 assigned to device CPU, is_swa = 0
load_tensors: layer  35 assigned to device CPU, is_swa = 0
create_tensor: loading tensor token_embd.weight
create_tensor: loading tensor token_embd.weight
create_tensor: loading tensor per_layer_token_embd.weight
create_tensor: loading tensor per_layer_model_proj.weight
create_tensor: loading tensor per_layer_proj_norm.weight
create_tensor: loading tensor output_norm.weight
create_tensor: loading tensor blk.0.attn_norm.weight
create_tensor: loading tensor blk.0.attn_q.weight
create_tensor: loading tensor blk.0.attn_k.weight
create_tensor: loading tensor blk.0.attn_v.weight
create_tensor: loading tensor blk.0.attn_output.weight
create_tensor: loading tensor blk.0.attn_q_norm.weight
create_tensor: loading tensor blk.0.attn_k_norm.weight
create_tensor: loading tensor blk.0.post_attention_norm.weight
create_tensor: loading tensor blk.0.layer_output_scale.weight
create_tensor: loading tensor blk.0.ffn_norm.weight
create_tensor: loading tensor blk.0.ffn_gate.weight
create_tensor: loading tensor blk.0.ffn_up.weight
create_tensor: loading tensor blk.0.ffn_down.weight
create_tensor: loading tensor blk.0.post_ffw_norm.weight
create_tensor: loading tensor blk.0.inp_gate.weight
create_tensor: loading tensor blk.0.proj.weight
create_tensor: loading tensor blk.0.post_norm.weight
create_tensor: loading tensor blk.1.attn_norm.weight
create_tensor: loading tensor blk.1.attn_q.weight
create_tensor: loading tensor blk.1.attn_k.weight
create_tensor: loading tensor blk.1.attn_v.weight
create_tensor: loading tensor blk.1.attn_output.weight
create_tensor: loading tensor blk.1.attn_q_norm.weight
create_tensor: loading tensor blk.1.attn_k_norm.weight
create_tensor: loading tensor blk.1.post_attention_norm.weight
create_tensor: loading tensor blk.1.layer_output_scale.weight
create_tensor: loading tensor blk.1.ffn_norm.weight
create_tensor: loading tensor blk.1.ffn_gate.weight
create_tensor: loading tensor blk.1.ffn_up.weight
create_tensor: loading tensor blk.1.ffn_down.weight
create_tensor: loading tensor blk.1.post_ffw_norm.weight
create_tensor: loading tensor blk.1.inp_gate.weight
create_tensor: loading tensor blk.1.proj.weight
create_tensor: loading tensor blk.1.post_norm.weight
create_tensor: loading tensor blk.2.attn_norm.weight
create_tensor: loading tensor blk.2.attn_q.weight
create_tensor: loading tensor blk.2.attn_k.weight
create_tensor: loading tensor blk.2.attn_v.weight
create_tensor: loading tensor blk.2.attn_output.weight
create_tensor: loading tensor blk.2.attn_q_norm.weight
create_tensor: loading tensor blk.2.attn_k_norm.weight
create_tensor: loading tensor blk.2.post_attention_norm.weight
create_tensor: loading tensor blk.2.layer_output_scale.weight
create_tensor: loading tensor blk.2.ffn_norm.weight
create_tensor: loading tensor blk.2.ffn_gate.weight
create_tensor: loading tensor blk.2.ffn_up.weight
create_tensor: loading tensor blk.2.ffn_down.weight
create_tensor: loading tensor blk.2.post_ffw_norm.weight
create_tensor: loading tensor blk.2.inp_gate.weight
create_tensor: loading tensor blk.2.proj.weight
create_tensor: loading tensor blk.2.post_norm.weight
create_tensor: loading tensor blk.3.attn_norm.weight
create_tensor: loading tensor blk.3.attn_q.weight
create_tensor: loading tensor blk.3.attn_k.weight
create_tensor: loading tensor blk.3.attn_v.weight
create_tensor: loading tensor blk.3.attn_output.weight
create_tensor: loading tensor blk.3.attn_q_norm.weight
create_tensor: loading tensor blk.3.attn_k_norm.weight
create_tensor: loading tensor blk.3.post_attention_norm.weight
create_tensor: loading tensor blk.3.layer_output_scale.weight
create_tensor: loading tensor blk.3.ffn_norm.weight
create_tensor: loading tensor blk.3.ffn_gate.weight
create_tensor: loading tensor blk.3.ffn_up.weight
create_tensor: loading tensor blk.3.ffn_down.weight
create_tensor: loading tensor blk.3.post_ffw_norm.weight
create_tensor: loading tensor blk.3.inp_gate.weight
create_tensor: loading tensor blk.3.proj.weight
create_tensor: loading tensor blk.3.post_norm.weight
create_tensor: loading tensor blk.4.attn_norm.weight
create_tensor: loading tensor blk.4.attn_q.weight
create_tensor: loading tensor blk.4.attn_k.weight
create_tensor: loading tensor blk.4.attn_v.weight
create_tensor: loading tensor blk.4.attn_output.weight
create_tensor: loading tensor blk.4.attn_q_norm.weight
create_tensor: loading tensor blk.4.attn_k_norm.weight
create_tensor: loading tensor blk.4.post_attention_norm.weight
create_tensor: loading tensor blk.4.layer_output_scale.weight
create_tensor: loading tensor rope_freqs.weight
create_tensor: loading tensor blk.4.ffn_norm.weight
create_tensor: loading tensor blk.4.ffn_gate.weight
create_tensor: loading tensor blk.4.ffn_up.weight
create_tensor: loading tensor blk.4.ffn_down.weight
create_tensor: loading tensor blk.4.post_ffw_norm.weight
create_tensor: loading tensor blk.4.inp_gate.weight
create_tensor: loading tensor blk.4.proj.weight
create_tensor: loading tensor blk.4.post_norm.weight
create_tensor: loading tensor blk.5.attn_norm.weight
create_tensor: loading tensor blk.5.attn_q.weight
create_tensor: loading tensor blk.5.attn_k.weight
create_tensor: loading tensor blk.5.attn_v.weight
create_tensor: loading tensor blk.5.attn_output.weight
create_tensor: loading tensor blk.5.attn_q_norm.weight
create_tensor: loading tensor blk.5.attn_k_norm.weight
create_tensor: loading tensor blk.5.post_attention_norm.weight
create_tensor: loading tensor blk.5.layer_output_scale.weight
create_tensor: loading tensor blk.5.ffn_norm.weight
create_tensor: loading tensor blk.5.ffn_gate.weight
create_tensor: loading tensor blk.5.ffn_up.weight
create_tensor: loading tensor blk.5.ffn_down.weight
create_tensor: loading tensor blk.5.post_ffw_norm.weight
create_tensor: loading tensor blk.5.inp_gate.weight
create_tensor: loading tensor blk.5.proj.weight
create_tensor: loading tensor blk.5.post_norm.weight
create_tensor: loading tensor blk.6.attn_norm.weight
create_tensor: loading tensor blk.6.attn_q.weight
create_tensor: loading tensor blk.6.attn_k.weight
create_tensor: loading tensor blk.6.attn_v.weight
create_tensor: loading tensor blk.6.attn_output.weight
create_tensor: loading tensor blk.6.attn_q_norm.weight
create_tensor: loading tensor blk.6.attn_k_norm.weight
create_tensor: loading tensor blk.6.post_attention_norm.weight
create_tensor: loading tensor blk.6.layer_output_scale.weight
create_tensor: loading tensor blk.6.ffn_norm.weight
create_tensor: loading tensor blk.6.ffn_gate.weight
create_tensor: loading tensor blk.6.ffn_up.weight
create_tensor: loading tensor blk.6.ffn_down.weight
create_tensor: loading tensor blk.6.post_ffw_norm.weight
create_tensor: loading tensor blk.6.inp_gate.weight
create_tensor: loading tensor blk.6.proj.weight
create_tensor: loading tensor blk.6.post_norm.weight
create_tensor: loading tensor blk.7.attn_norm.weight
create_tensor: loading tensor blk.7.attn_q.weight
create_tensor: loading tensor blk.7.attn_k.weight
create_tensor: loading tensor blk.7.attn_v.weight
create_tensor: loading tensor blk.7.attn_output.weight
create_tensor: loading tensor blk.7.attn_q_norm.weight
create_tensor: loading tensor blk.7.attn_k_norm.weight
create_tensor: loading tensor blk.7.post_attention_norm.weight
create_tensor: loading tensor blk.7.layer_output_scale.weight
create_tensor: loading tensor blk.7.ffn_norm.weight
create_tensor: loading tensor blk.7.ffn_gate.weight
create_tensor: loading tensor blk.7.ffn_up.weight
create_tensor: loading tensor blk.7.ffn_down.weight
create_tensor: loading tensor blk.7.post_ffw_norm.weight
create_tensor: loading tensor blk.7.inp_gate.weight
create_tensor: loading tensor blk.7.proj.weight
create_tensor: loading tensor blk.7.post_norm.weight
create_tensor: loading tensor blk.8.attn_norm.weight
create_tensor: loading tensor blk.8.attn_q.weight
create_tensor: loading tensor blk.8.attn_k.weight
create_tensor: loading tensor blk.8.attn_v.weight
create_tensor: loading tensor blk.8.attn_output.weight
create_tensor: loading tensor blk.8.attn_q_norm.weight
create_tensor: loading tensor blk.8.attn_k_norm.weight
create_tensor: loading tensor blk.8.post_attention_norm.weight
create_tensor: loading tensor blk.8.layer_output_scale.weight
create_tensor: loading tensor blk.8.ffn_norm.weight
create_tensor: loading tensor blk.8.ffn_gate.weight
create_tensor: loading tensor blk.8.ffn_up.weight
create_tensor: loading tensor blk.8.ffn_down.weight
create_tensor: loading tensor blk.8.post_ffw_norm.weight
create_tensor: loading tensor blk.8.inp_gate.weight
create_tensor: loading tensor blk.8.proj.weight
create_tensor: loading tensor blk.8.post_norm.weight
create_tensor: loading tensor blk.9.attn_norm.weight
create_tensor: loading tensor blk.9.attn_q.weight
create_tensor: loading tensor blk.9.attn_k.weight
create_tensor: loading tensor blk.9.attn_v.weight
create_tensor: loading tensor blk.9.attn_output.weight
create_tensor: loading tensor blk.9.attn_q_norm.weight
create_tensor: loading tensor blk.9.attn_k_norm.weight
create_tensor: loading tensor blk.9.post_attention_norm.weight
create_tensor: loading tensor blk.9.layer_output_scale.weight
create_tensor: loading tensor blk.9.ffn_norm.weight
create_tensor: loading tensor blk.9.ffn_gate.weight
create_tensor: loading tensor blk.9.ffn_up.weight
create_tensor: loading tensor blk.9.ffn_down.weight
create_tensor: loading tensor blk.9.post_ffw_norm.weight
create_tensor: loading tensor blk.9.inp_gate.weight
create_tensor: loading tensor blk.9.proj.weight
create_tensor: loading tensor blk.9.post_norm.weight
create_tensor: loading tensor blk.10.attn_norm.weight
create_tensor: loading tensor blk.10.attn_q.weight
create_tensor: loading tensor blk.10.attn_k.weight
create_tensor: loading tensor blk.10.attn_v.weight
create_tensor: loading tensor blk.10.attn_output.weight
create_tensor: loading tensor blk.10.attn_q_norm.weight
create_tensor: loading tensor blk.10.attn_k_norm.weight
create_tensor: loading tensor blk.10.post_attention_norm.weight
create_tensor: loading tensor blk.10.layer_output_scale.weight
create_tensor: loading tensor blk.10.ffn_norm.weight
create_tensor: loading tensor blk.10.ffn_gate.weight
create_tensor: loading tensor blk.10.ffn_up.weight
create_tensor: loading tensor blk.10.ffn_down.weight
create_tensor: loading tensor blk.10.post_ffw_norm.weight
create_tensor: loading tensor blk.10.inp_gate.weight
create_tensor: loading tensor blk.10.proj.weight
create_tensor: loading tensor blk.10.post_norm.weight
create_tensor: loading tensor blk.11.attn_norm.weight
create_tensor: loading tensor blk.11.attn_q.weight
create_tensor: loading tensor blk.11.attn_k.weight
create_tensor: loading tensor blk.11.attn_v.weight
create_tensor: loading tensor blk.11.attn_output.weight
create_tensor: loading tensor blk.11.attn_q_norm.weight
create_tensor: loading tensor blk.11.attn_k_norm.weight
create_tensor: loading tensor blk.11.post_attention_norm.weight
create_tensor: loading tensor blk.11.layer_output_scale.weight
create_tensor: loading tensor blk.11.ffn_norm.weight
create_tensor: loading tensor blk.11.ffn_gate.weight
create_tensor: loading tensor blk.11.ffn_up.weight
create_tensor: loading tensor blk.11.ffn_down.weight
create_tensor: loading tensor blk.11.post_ffw_norm.weight
create_tensor: loading tensor blk.11.inp_gate.weight
create_tensor: loading tensor blk.11.proj.weight
create_tensor: loading tensor blk.11.post_norm.weight
create_tensor: loading tensor blk.12.attn_norm.weight
create_tensor: loading tensor blk.12.attn_q.weight
create_tensor: loading tensor blk.12.attn_k.weight
create_tensor: loading tensor blk.12.attn_v.weight
create_tensor: loading tensor blk.12.attn_output.weight
create_tensor: loading tensor blk.12.attn_q_norm.weight
create_tensor: loading tensor blk.12.attn_k_norm.weight
create_tensor: loading tensor blk.12.post_attention_norm.weight
create_tensor: loading tensor blk.12.layer_output_scale.weight
create_tensor: loading tensor blk.12.ffn_norm.weight
create_tensor: loading tensor blk.12.ffn_gate.weight
create_tensor: loading tensor blk.12.ffn_up.weight
create_tensor: loading tensor blk.12.ffn_down.weight
create_tensor: loading tensor blk.12.post_ffw_norm.weight
create_tensor: loading tensor blk.12.inp_gate.weight
create_tensor: loading tensor blk.12.proj.weight
create_tensor: loading tensor blk.12.post_norm.weight
create_tensor: loading tensor blk.13.attn_norm.weight
create_tensor: loading tensor blk.13.attn_q.weight
create_tensor: loading tensor blk.13.attn_k.weight
create_tensor: loading tensor blk.13.attn_v.weight
create_tensor: loading tensor blk.13.attn_output.weight
create_tensor: loading tensor blk.13.attn_q_norm.weight
create_tensor: loading tensor blk.13.attn_k_norm.weight
create_tensor: loading tensor blk.13.post_attention_norm.weight
create_tensor: loading tensor blk.13.layer_output_scale.weight
create_tensor: loading tensor blk.13.ffn_norm.weight
create_tensor: loading tensor blk.13.ffn_gate.weight
create_tensor: loading tensor blk.13.ffn_up.weight
create_tensor: loading tensor blk.13.ffn_down.weight
create_tensor: loading tensor blk.13.post_ffw_norm.weight
create_tensor: loading tensor blk.13.inp_gate.weight
create_tensor: loading tensor blk.13.proj.weight
create_tensor: loading tensor blk.13.post_norm.weight
create_tensor: loading tensor blk.14.attn_norm.weight
create_tensor: loading tensor blk.14.attn_q.weight
create_tensor: loading tensor blk.14.attn_k.weight
create_tensor: loading tensor blk.14.attn_v.weight
create_tensor: loading tensor blk.14.attn_output.weight
create_tensor: loading tensor blk.14.attn_q_norm.weight
create_tensor: loading tensor blk.14.attn_k_norm.weight
create_tensor: loading tensor blk.14.post_attention_norm.weight
create_tensor: loading tensor blk.14.layer_output_scale.weight
create_tensor: loading tensor blk.14.ffn_norm.weight
create_tensor: loading tensor blk.14.ffn_gate.weight
create_tensor: loading tensor blk.14.ffn_up.weight
create_tensor: loading tensor blk.14.ffn_down.weight
create_tensor: loading tensor blk.14.post_ffw_norm.weight
create_tensor: loading tensor blk.14.inp_gate.weight
create_tensor: loading tensor blk.14.proj.weight
create_tensor: loading tensor blk.14.post_norm.weight
create_tensor: loading tensor blk.15.attn_norm.weight
create_tensor: loading tensor blk.15.attn_q.weight
create_tensor: loading tensor blk.15.attn_k.weight
create_tensor: loading tensor blk.15.attn_v.weight
create_tensor: loading tensor blk.15.attn_output.weight
create_tensor: loading tensor blk.15.attn_q_norm.weight
create_tensor: loading tensor blk.15.attn_k_norm.weight
create_tensor: loading tensor blk.15.post_attention_norm.weight
create_tensor: loading tensor blk.15.layer_output_scale.weight
create_tensor: loading tensor blk.15.ffn_norm.weight
create_tensor: loading tensor blk.15.ffn_gate.weight
create_tensor: loading tensor blk.15.ffn_up.weight
create_tensor: loading tensor blk.15.ffn_down.weight
create_tensor: loading tensor blk.15.post_ffw_norm.weight
create_tensor: loading tensor blk.15.inp_gate.weight
create_tensor: loading tensor blk.15.proj.weight
create_tensor: loading tensor blk.15.post_norm.weight
create_tensor: loading tensor blk.16.attn_norm.weight
create_tensor: loading tensor blk.16.attn_q.weight
create_tensor: loading tensor blk.16.attn_k.weight
create_tensor: loading tensor blk.16.attn_v.weight
create_tensor: loading tensor blk.16.attn_output.weight
create_tensor: loading tensor blk.16.attn_q_norm.weight
create_tensor: loading tensor blk.16.attn_k_norm.weight
create_tensor: loading tensor blk.16.post_attention_norm.weight
create_tensor: loading tensor blk.16.layer_output_scale.weight
create_tensor: loading tensor blk.16.ffn_norm.weight
create_tensor: loading tensor blk.16.ffn_gate.weight
create_tensor: loading tensor blk.16.ffn_up.weight
create_tensor: loading tensor blk.16.ffn_down.weight
create_tensor: loading tensor blk.16.post_ffw_norm.weight
create_tensor: loading tensor blk.16.inp_gate.weight
create_tensor: loading tensor blk.16.proj.weight
create_tensor: loading tensor blk.16.post_norm.weight
create_tensor: loading tensor blk.17.attn_norm.weight
create_tensor: loading tensor blk.17.attn_q.weight
create_tensor: loading tensor blk.17.attn_k.weight
create_tensor: loading tensor blk.17.attn_v.weight
create_tensor: loading tensor blk.17.attn_output.weight
create_tensor: loading tensor blk.17.attn_q_norm.weight
create_tensor: loading tensor blk.17.attn_k_norm.weight
create_tensor: loading tensor blk.17.post_attention_norm.weight
create_tensor: loading tensor blk.17.layer_output_scale.weight
create_tensor: loading tensor blk.17.ffn_norm.weight
create_tensor: loading tensor blk.17.ffn_gate.weight
create_tensor: loading tensor blk.17.ffn_up.weight
create_tensor: loading tensor blk.17.ffn_down.weight
create_tensor: loading tensor blk.17.post_ffw_norm.weight
create_tensor: loading tensor blk.17.inp_gate.weight
create_tensor: loading tensor blk.17.proj.weight
create_tensor: loading tensor blk.17.post_norm.weight
create_tensor: loading tensor blk.18.attn_norm.weight
create_tensor: loading tensor blk.18.attn_q.weight
create_tensor: loading tensor blk.18.attn_k.weight
create_tensor: loading tensor blk.18.attn_v.weight
create_tensor: loading tensor blk.18.attn_output.weight
create_tensor: loading tensor blk.18.attn_q_norm.weight
create_tensor: loading tensor blk.18.attn_k_norm.weight
create_tensor: loading tensor blk.18.post_attention_norm.weight
create_tensor: loading tensor blk.18.layer_output_scale.weight
create_tensor: loading tensor blk.18.ffn_norm.weight
create_tensor: loading tensor blk.18.ffn_gate.weight
create_tensor: loading tensor blk.18.ffn_up.weight
create_tensor: loading tensor blk.18.ffn_down.weight
create_tensor: loading tensor blk.18.post_ffw_norm.weight
create_tensor: loading tensor blk.18.inp_gate.weight
create_tensor: loading tensor blk.18.proj.weight
create_tensor: loading tensor blk.18.post_norm.weight
create_tensor: loading tensor blk.19.attn_norm.weight
create_tensor: loading tensor blk.19.attn_q.weight
create_tensor: loading tensor blk.19.attn_k.weight
create_tensor: loading tensor blk.19.attn_v.weight
create_tensor: loading tensor blk.19.attn_output.weight
create_tensor: loading tensor blk.19.attn_q_norm.weight
create_tensor: loading tensor blk.19.attn_k_norm.weight
create_tensor: loading tensor blk.19.post_attention_norm.weight
create_tensor: loading tensor blk.19.layer_output_scale.weight
create_tensor: loading tensor blk.19.ffn_norm.weight
create_tensor: loading tensor blk.19.ffn_gate.weight
create_tensor: loading tensor blk.19.ffn_up.weight
create_tensor: loading tensor blk.19.ffn_down.weight
create_tensor: loading tensor blk.19.post_ffw_norm.weight
create_tensor: loading tensor blk.19.inp_gate.weight
create_tensor: loading tensor blk.19.proj.weight
create_tensor: loading tensor blk.19.post_norm.weight
create_tensor: loading tensor blk.20.attn_norm.weight
create_tensor: loading tensor blk.20.attn_q.weight
create_tensor: loading tensor blk.20.attn_k.weight
create_tensor: loading tensor blk.20.attn_v.weight
create_tensor: loading tensor blk.20.attn_output.weight
create_tensor: loading tensor blk.20.attn_q_norm.weight
create_tensor: loading tensor blk.20.attn_k_norm.weight
create_tensor: loading tensor blk.20.post_attention_norm.weight
create_tensor: loading tensor blk.20.layer_output_scale.weight
create_tensor: loading tensor blk.20.ffn_norm.weight
create_tensor: loading tensor blk.20.ffn_gate.weight
create_tensor: loading tensor blk.20.ffn_up.weight
create_tensor: loading tensor blk.20.ffn_down.weight
create_tensor: loading tensor blk.20.post_ffw_norm.weight
create_tensor: loading tensor blk.20.inp_gate.weight
create_tensor: loading tensor blk.20.proj.weight
create_tensor: loading tensor blk.20.post_norm.weight
create_tensor: loading tensor blk.21.attn_norm.weight
create_tensor: loading tensor blk.21.attn_q.weight
create_tensor: loading tensor blk.21.attn_k.weight
create_tensor: loading tensor blk.21.attn_v.weight
create_tensor: loading tensor blk.21.attn_output.weight
create_tensor: loading tensor blk.21.attn_q_norm.weight
create_tensor: loading tensor blk.21.attn_k_norm.weight
create_tensor: loading tensor blk.21.post_attention_norm.weight
create_tensor: loading tensor blk.21.layer_output_scale.weight
create_tensor: loading tensor blk.21.ffn_norm.weight
create_tensor: loading tensor blk.21.ffn_gate.weight
create_tensor: loading tensor blk.21.ffn_up.weight
create_tensor: loading tensor blk.21.ffn_down.weight
create_tensor: loading tensor blk.21.post_ffw_norm.weight
create_tensor: loading tensor blk.21.inp_gate.weight
create_tensor: loading tensor blk.21.proj.weight
create_tensor: loading tensor blk.21.post_norm.weight
create_tensor: loading tensor blk.22.attn_norm.weight
create_tensor: loading tensor blk.22.attn_q.weight
create_tensor: loading tensor blk.22.attn_k.weight
create_tensor: loading tensor blk.22.attn_v.weight
create_tensor: loading tensor blk.22.attn_output.weight
create_tensor: loading tensor blk.22.attn_q_norm.weight
create_tensor: loading tensor blk.22.attn_k_norm.weight
create_tensor: loading tensor blk.22.post_attention_norm.weight
create_tensor: loading tensor blk.22.layer_output_scale.weight
create_tensor: loading tensor blk.22.ffn_norm.weight
create_tensor: loading tensor blk.22.ffn_gate.weight
create_tensor: loading tensor blk.22.ffn_up.weight
create_tensor: loading tensor blk.22.ffn_down.weight
create_tensor: loading tensor blk.22.post_ffw_norm.weight
create_tensor: loading tensor blk.22.inp_gate.weight
create_tensor: loading tensor blk.22.proj.weight
create_tensor: loading tensor blk.22.post_norm.weight
create_tensor: loading tensor blk.23.attn_norm.weight
create_tensor: loading tensor blk.23.attn_q.weight
create_tensor: loading tensor blk.23.attn_k.weight
create_tensor: loading tensor blk.23.attn_v.weight
create_tensor: loading tensor blk.23.attn_output.weight
create_tensor: loading tensor blk.23.attn_q_norm.weight
create_tensor: loading tensor blk.23.attn_k_norm.weight
create_tensor: loading tensor blk.23.post_attention_norm.weight
create_tensor: loading tensor blk.23.layer_output_scale.weight
create_tensor: loading tensor blk.23.ffn_norm.weight
create_tensor: loading tensor blk.23.ffn_gate.weight
create_tensor: loading tensor blk.23.ffn_up.weight
create_tensor: loading tensor blk.23.ffn_down.weight
create_tensor: loading tensor blk.23.post_ffw_norm.weight
create_tensor: loading tensor blk.23.inp_gate.weight
create_tensor: loading tensor blk.23.proj.weight
create_tensor: loading tensor blk.23.post_norm.weight
create_tensor: loading tensor blk.24.attn_norm.weight
create_tensor: loading tensor blk.24.attn_q.weight
create_tensor: loading tensor blk.24.attn_k.weight
create_tensor: loading tensor blk.24.attn_v.weight
create_tensor: loading tensor blk.24.attn_output.weight
create_tensor: loading tensor blk.24.attn_q_norm.weight
create_tensor: loading tensor blk.24.attn_k_norm.weight
create_tensor: loading tensor blk.24.post_attention_norm.weight
create_tensor: loading tensor blk.24.layer_output_scale.weight
create_tensor: loading tensor blk.24.ffn_norm.weight
create_tensor: loading tensor blk.24.ffn_gate.weight
create_tensor: loading tensor blk.24.ffn_up.weight
create_tensor: loading tensor blk.24.ffn_down.weight
create_tensor: loading tensor blk.24.post_ffw_norm.weight
create_tensor: loading tensor blk.24.inp_gate.weight
create_tensor: loading tensor blk.24.proj.weight
create_tensor: loading tensor blk.24.post_norm.weight
create_tensor: loading tensor blk.25.attn_norm.weight
create_tensor: loading tensor blk.25.attn_q.weight
create_tensor: loading tensor blk.25.attn_k.weight
create_tensor: loading tensor blk.25.attn_v.weight
create_tensor: loading tensor blk.25.attn_output.weight
create_tensor: loading tensor blk.25.attn_q_norm.weight
create_tensor: loading tensor blk.25.attn_k_norm.weight
create_tensor: loading tensor blk.25.post_attention_norm.weight
create_tensor: loading tensor blk.25.layer_output_scale.weight
create_tensor: loading tensor blk.25.ffn_norm.weight
create_tensor: loading tensor blk.25.ffn_gate.weight
create_tensor: loading tensor blk.25.ffn_up.weight
create_tensor: loading tensor blk.25.ffn_down.weight
create_tensor: loading tensor blk.25.post_ffw_norm.weight
create_tensor: loading tensor blk.25.inp_gate.weight
create_tensor: loading tensor blk.25.proj.weight
create_tensor: loading tensor blk.25.post_norm.weight
create_tensor: loading tensor blk.26.attn_norm.weight
create_tensor: loading tensor blk.26.attn_q.weight
create_tensor: loading tensor blk.26.attn_k.weight
create_tensor: loading tensor blk.26.attn_v.weight
create_tensor: loading tensor blk.26.attn_output.weight
create_tensor: loading tensor blk.26.attn_q_norm.weight
create_tensor: loading tensor blk.26.attn_k_norm.weight
create_tensor: loading tensor blk.26.post_attention_norm.weight
create_tensor: loading tensor blk.26.layer_output_scale.weight
create_tensor: loading tensor blk.26.ffn_norm.weight
create_tensor: loading tensor blk.26.ffn_gate.weight
create_tensor: loading tensor blk.26.ffn_up.weight
create_tensor: loading tensor blk.26.ffn_down.weight
create_tensor: loading tensor blk.26.post_ffw_norm.weight
create_tensor: loading tensor blk.26.inp_gate.weight
create_tensor: loading tensor blk.26.proj.weight
create_tensor: loading tensor blk.26.post_norm.weight
create_tensor: loading tensor blk.27.attn_norm.weight
create_tensor: loading tensor blk.27.attn_q.weight
create_tensor: loading tensor blk.27.attn_k.weight
create_tensor: loading tensor blk.27.attn_v.weight
create_tensor: loading tensor blk.27.attn_output.weight
create_tensor: loading tensor blk.27.attn_q_norm.weight
create_tensor: loading tensor blk.27.attn_k_norm.weight
create_tensor: loading tensor blk.27.post_attention_norm.weight
create_tensor: loading tensor blk.27.layer_output_scale.weight
create_tensor: loading tensor blk.27.ffn_norm.weight
create_tensor: loading tensor blk.27.ffn_gate.weight
create_tensor: loading tensor blk.27.ffn_up.weight
create_tensor: loading tensor blk.27.ffn_down.weight
create_tensor: loading tensor blk.27.post_ffw_norm.weight
create_tensor: loading tensor blk.27.inp_gate.weight
create_tensor: loading tensor blk.27.proj.weight
create_tensor: loading tensor blk.27.post_norm.weight
create_tensor: loading tensor blk.28.attn_norm.weight
create_tensor: loading tensor blk.28.attn_q.weight
create_tensor: loading tensor blk.28.attn_k.weight
create_tensor: loading tensor blk.28.attn_v.weight
create_tensor: loading tensor blk.28.attn_output.weight
create_tensor: loading tensor blk.28.attn_q_norm.weight
create_tensor: loading tensor blk.28.attn_k_norm.weight
create_tensor: loading tensor blk.28.post_attention_norm.weight
create_tensor: loading tensor blk.28.layer_output_scale.weight
create_tensor: loading tensor blk.28.ffn_norm.weight
create_tensor: loading tensor blk.28.ffn_gate.weight
create_tensor: loading tensor blk.28.ffn_up.weight
create_tensor: loading tensor blk.28.ffn_down.weight
create_tensor: loading tensor blk.28.post_ffw_norm.weight
create_tensor: loading tensor blk.28.inp_gate.weight
create_tensor: loading tensor blk.28.proj.weight
create_tensor: loading tensor blk.28.post_norm.weight
create_tensor: loading tensor blk.29.attn_norm.weight
create_tensor: loading tensor blk.29.attn_q.weight
create_tensor: loading tensor blk.29.attn_k.weight
create_tensor: loading tensor blk.29.attn_v.weight
create_tensor: loading tensor blk.29.attn_output.weight
create_tensor: loading tensor blk.29.attn_q_norm.weight
create_tensor: loading tensor blk.29.attn_k_norm.weight
create_tensor: loading tensor blk.29.post_attention_norm.weight
create_tensor: loading tensor blk.29.layer_output_scale.weight
create_tensor: loading tensor blk.29.ffn_norm.weight
create_tensor: loading tensor blk.29.ffn_gate.weight
create_tensor: loading tensor blk.29.ffn_up.weight
create_tensor: loading tensor blk.29.ffn_down.weight
create_tensor: loading tensor blk.29.post_ffw_norm.weight
create_tensor: loading tensor blk.29.inp_gate.weight
create_tensor: loading tensor blk.29.proj.weight
create_tensor: loading tensor blk.29.post_norm.weight
create_tensor: loading tensor blk.30.attn_norm.weight
create_tensor: loading tensor blk.30.attn_q.weight
create_tensor: loading tensor blk.30.attn_k.weight
create_tensor: loading tensor blk.30.attn_v.weight
create_tensor: loading tensor blk.30.attn_output.weight
create_tensor: loading tensor blk.30.attn_q_norm.weight
create_tensor: loading tensor blk.30.attn_k_norm.weight
create_tensor: loading tensor blk.30.post_attention_norm.weight
create_tensor: loading tensor blk.30.layer_output_scale.weight
create_tensor: loading tensor blk.30.ffn_norm.weight
create_tensor: loading tensor blk.30.ffn_gate.weight
create_tensor: loading tensor blk.30.ffn_up.weight
create_tensor: loading tensor blk.30.ffn_down.weight
create_tensor: loading tensor blk.30.post_ffw_norm.weight
create_tensor: loading tensor blk.30.inp_gate.weight
create_tensor: loading tensor blk.30.proj.weight
create_tensor: loading tensor blk.30.post_norm.weight
create_tensor: loading tensor blk.31.attn_norm.weight
create_tensor: loading tensor blk.31.attn_q.weight
create_tensor: loading tensor blk.31.attn_k.weight
create_tensor: loading tensor blk.31.attn_v.weight
create_tensor: loading tensor blk.31.attn_output.weight
create_tensor: loading tensor blk.31.attn_q_norm.weight
create_tensor: loading tensor blk.31.attn_k_norm.weight
create_tensor: loading tensor blk.31.post_attention_norm.weight
create_tensor: loading tensor blk.31.layer_output_scale.weight
create_tensor: loading tensor blk.31.ffn_norm.weight
create_tensor: loading tensor blk.31.ffn_gate.weight
create_tensor: loading tensor blk.31.ffn_up.weight
create_tensor: loading tensor blk.31.ffn_down.weight
create_tensor: loading tensor blk.31.post_ffw_norm.weight
create_tensor: loading tensor blk.31.inp_gate.weight
create_tensor: loading tensor blk.31.proj.weight
create_tensor: loading tensor blk.31.post_norm.weight
create_tensor: loading tensor blk.32.attn_norm.weight
create_tensor: loading tensor blk.32.attn_q.weight
create_tensor: loading tensor blk.32.attn_k.weight
create_tensor: loading tensor blk.32.attn_v.weight
create_tensor: loading tensor blk.32.attn_output.weight
create_tensor: loading tensor blk.32.attn_q_norm.weight
create_tensor: loading tensor blk.32.attn_k_norm.weight
create_tensor: loading tensor blk.32.post_attention_norm.weight
create_tensor: loading tensor blk.32.layer_output_scale.weight
create_tensor: loading tensor blk.32.ffn_norm.weight
create_tensor: loading tensor blk.32.ffn_gate.weight
create_tensor: loading tensor blk.32.ffn_up.weight
create_tensor: loading tensor blk.32.ffn_down.weight
create_tensor: loading tensor blk.32.post_ffw_norm.weight
create_tensor: loading tensor blk.32.inp_gate.weight
create_tensor: loading tensor blk.32.proj.weight
create_tensor: loading tensor blk.32.post_norm.weight
create_tensor: loading tensor blk.33.attn_norm.weight
create_tensor: loading tensor blk.33.attn_q.weight
create_tensor: loading tensor blk.33.attn_k.weight
create_tensor: loading tensor blk.33.attn_v.weight
create_tensor: loading tensor blk.33.attn_output.weight
create_tensor: loading tensor blk.33.attn_q_norm.weight
create_tensor: loading tensor blk.33.attn_k_norm.weight
create_tensor: loading tensor blk.33.post_attention_norm.weight
create_tensor: loading tensor blk.33.layer_output_scale.weight
create_tensor: loading tensor blk.33.ffn_norm.weight
create_tensor: loading tensor blk.33.ffn_gate.weight
create_tensor: loading tensor blk.33.ffn_up.weight
create_tensor: loading tensor blk.33.ffn_down.weight
create_tensor: loading tensor blk.33.post_ffw_norm.weight
create_tensor: loading tensor blk.33.inp_gate.weight
create_tensor: loading tensor blk.33.proj.weight
create_tensor: loading tensor blk.33.post_norm.weight
create_tensor: loading tensor blk.34.attn_norm.weight
create_tensor: loading tensor blk.34.attn_q.weight
create_tensor: loading tensor blk.34.attn_k.weight
create_tensor: loading tensor blk.34.attn_v.weight
create_tensor: loading tensor blk.34.attn_output.weight
create_tensor: loading tensor blk.34.attn_q_norm.weight
create_tensor: loading tensor blk.34.attn_k_norm.weight
create_tensor: loading tensor blk.34.post_attention_norm.weight
create_tensor: loading tensor blk.34.layer_output_scale.weight
create_tensor: loading tensor blk.34.ffn_norm.weight
create_tensor: loading tensor blk.34.ffn_gate.weight
create_tensor: loading tensor blk.34.ffn_up.weight
create_tensor: loading tensor blk.34.ffn_down.weight
create_tensor: loading tensor blk.34.post_ffw_norm.weight
create_tensor: loading tensor blk.34.inp_gate.weight
create_tensor: loading tensor blk.34.proj.weight
create_tensor: loading tensor blk.34.post_norm.weight
done_getting_tensors: tensor 'token_embd.weight' (q4_K) (and 607 others) cannot be used with preferred buffer type CPU_REPACK, using CPU instead
load_tensors:          CPU model buffer size =  3163.73 MiB
load_all_data: no device found for buffer type CPU for async uploads
.........................................
load_tensors: releasing mmaps (not needed after tensor copy)
llama_model_loader: direct I/O is enabled, disabling mmap
llama_model_loader: loaded meta data with 49 key-value pairs and 51 tensors from d:\llama.cpp\models\gemma-4\gemma-4-E2B-it-assistant.F16.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = gemma4_assistant
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                     general.sampling.top_k i32              = 64
llama_model_loader: - kv   3:                     general.sampling.top_p f32              = 0.950000
llama_model_loader: - kv   4:                      general.sampling.temp f32              = 1.000000
llama_model_loader: - kv   5:                               general.name str              = Gemma 4 E2B It Assistant
llama_model_loader: - kv   6:                         general.size_label str              = 78M
llama_model_loader: - kv   7:                            general.license str              = apache-2.0
llama_model_loader: - kv   8:                       general.license.link str              = https://ai.google.dev/gemma/docs/gemm...
llama_model_loader: - kv   9:                               general.tags arr[str,1]       = ["any-to-any"]
llama_model_loader: - kv  10:               gemma4_assistant.block_count u32              = 4
llama_model_loader: - kv  11:            gemma4_assistant.context_length u32              = 131072
llama_model_loader: - kv  12:          gemma4_assistant.embedding_length u32              = 256
llama_model_loader: - kv  13:       gemma4_assistant.feed_forward_length u32              = 2048
llama_model_loader: - kv  14:      gemma4_assistant.attention.head_count u32              = 4
llama_model_loader: - kv  15:   gemma4_assistant.attention.head_count_kv u32              = 1
llama_model_loader: - kv  16:            gemma4_assistant.rope.freq_base f32              = 1000000.000000
llama_model_loader: - kv  17:        gemma4_assistant.rope.freq_base_swa f32              = 10000.000000
llama_model_loader: - kv  18: gemma4_assistant.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  19:      gemma4_assistant.attention.key_length u32              = 512
llama_model_loader: - kv  20:    gemma4_assistant.attention.value_length u32              = 512
llama_model_loader: - kv  21:                          general.file_type u32              = 1
llama_model_loader: - kv  22:  gemma4_assistant.attention.sliding_window u32              = 512
llama_model_loader: - kv  23: gemma4_assistant.attention.shared_kv_layers u32              = 4
llama_model_loader: - kv  24: gemma4_assistant.embedding_length_per_layer_input u32              = 0
llama_model_loader: - kv  25: gemma4_assistant.attention.sliding_window_pattern arr[bool,4]      = [true, true, true, false]
llama_model_loader: - kv  26:  gemma4_assistant.attention.key_length_swa u32              = 256
llama_model_loader: - kv  27: gemma4_assistant.attention.value_length_swa u32              = 256
llama_model_loader: - kv  28:      gemma4_assistant.rope.dimension_count u32              = 512
llama_model_loader: - kv  29:  gemma4_assistant.rope.dimension_count_swa u32              = 256
llama_model_loader: - kv  30:               gemma4_assistant.n_centroids u32              = 2048
llama_model_loader: - kv  31:            gemma4_assistant.centroid_top_k u32              = 32
llama_model_loader: - kv  32:           gemma4_assistant.n_embd_backbone u32              = 1536
llama_model_loader: - kv  33:    gemma4_assistant.use_ordered_embeddings bool             = true
llama_model_loader: - kv  34:          gemma4_assistant.attention.k_eq_v bool             = false
llama_model_loader: - kv  35:      gemma4_assistant.requires_target_arch str              = gemma4
llama_model_loader: - kv  36:               general.quantization_version u32              = 2
llama_model_loader: - kv  37:                       tokenizer.ggml.model str              = gemma4
llama_model_loader: - kv  38:                      tokenizer.ggml.tokens arr[str,262144]  = ["<pad>", "<eos>", "<bos>", "<unk>", ...
llama_model_loader: - kv  39:                      tokenizer.ggml.scores arr[f32,262144]  = [-1000.000000, -1000.000000, -1000.00...
llama_model_loader: - kv  40:                  tokenizer.ggml.token_type arr[i32,262144]  = [3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  41:                      tokenizer.ggml.merges arr[str,514906]  = ["\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n \n", ...
llama_model_loader: - kv  42:                tokenizer.ggml.bos_token_id u32              = 2
llama_model_loader: - kv  43:                tokenizer.ggml.eos_token_id u32              = 1
llama_model_loader: - kv  44:            tokenizer.ggml.unknown_token_id u32              = 3
llama_model_loader: - kv  45:            tokenizer.ggml.padding_token_id u32              = 0
llama_model_loader: - kv  46:               tokenizer.ggml.mask_token_id u32              = 4
llama_model_loader: - kv  47:            tokenizer.ggml.add_space_prefix bool             = false
llama_model_loader: - kv  48:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - type  f32:   26 tensors
llama_model_loader: - type  f16:   24 tensors
llama_model_loader: - type  i32:    1 tensors
print_info: file format = GGUF V3 (latest)
print_info: file type   = F16
print_info: file size   = 149.27 MiB (16.05 BPW)
init_tokenizer: initializing tokenizer for type 2
load: 0 unused tokens
load: control token:      0 '<pad>' is not marked as EOG
load: control token:      3 '<unk>' is not marked as EOG
load: control token:      2 '<bos>' is not marked as EOG
load: control token:      4 '<mask>' is not marked as EOG
load: control token:     47 '<tool|>' is not marked as EOG
load: control token:     98 '<|think|>' is not marked as EOG
load: control token:     46 '<|tool>' is not marked as EOG
load: control-looking token:     50 '<|tool_response>' was not control-type; this is probably a bug in the model. its type will be overridden
load: control token:    105 '<|turn>' is not marked as EOG
load: control-looking token:    212 '</s>' was not control-type; this is probably a bug in the model. its type will be overridden
load: control token: 258880 '<|image|>' is not marked as EOG
load: control token: 258882 '<image|>' is not marked as EOG
load: control token: 256000 '<|audio>' is not marked as EOG
load: control token: 258883 '<audio|>' is not marked as EOG
load: control token: 255999 '<|image>' is not marked as EOG
load: control token: 258881 '<|audio|>' is not marked as EOG
load: printing all EOG tokens:
load:   - 1 ('<eos>')
load:   - 50 ('<|tool_response>')
load:   - 106 ('<turn|>')
load:   - 212 ('</s>')
load: special_eog_ids contains '<|tool_response>', removing '</s>' token from EOG list
load: special tokens cache size = 23
load: token to piece cache size = 1.9445 MB
print_info: arch                  = gemma4_assistant
print_info: vocab_only            = 0
print_info: no_alloc              = 0
print_info: n_ctx_train           = 131072
print_info: n_embd                = 256
print_info: n_embd_inp            = 256
print_info: n_layer               = 4
print_info: n_head                = 4
print_info: n_head_kv             = 1
print_info: n_rot                 = 512
print_info: n_swa                 = 512
print_info: is_swa_any            = 1
print_info: n_embd_head_k         = 512
print_info: n_embd_head_v         = 512
print_info: n_gqa                 = 4
print_info: n_embd_k_gqa          = [256, 256, 256, 512]
print_info: n_embd_v_gqa          = [256, 256, 256, 512]
print_info: f_norm_eps            = 0.0e+00
print_info: f_norm_rms_eps        = 1.0e-06
print_info: f_clamp_kqv           = 0.0e+00
print_info: f_max_alibi_bias      = 0.0e+00
print_info: f_logit_scale         = 0.0e+00
print_info: f_attn_scale          = 1.0e+00
print_info: f_attn_value_scale    = 0.0000
print_info: n_ff                  = 2048
print_info: n_expert              = 0
print_info: n_expert_used         = 0
print_info: n_expert_groups       = 0
print_info: n_group_used          = 0
print_info: causal attn           = 1
print_info: pooling type          = -1
print_info: rope type             = 2
print_info: rope scaling          = linear
print_info: freq_base_train       = 1000000.0
print_info: freq_scale_train      = 1
print_info: freq_base_swa         = 10000.0
print_info: freq_scale_swa        = 1
print_info: n_embd_head_k_swa     = 256
print_info: n_embd_head_v_swa     = 256
print_info: n_rot_swa             = 256
print_info: n_ctx_orig_yarn       = 131072
print_info: rope_yarn_log_mul     = 0.0000
print_info: rope_finetuned        = unknown
print_info: model type            = ?B
print_info: model params          = 77.99 M
print_info: general.name          = Gemma 4 E2B It Assistant
print_info: vocab type            = BPE
print_info: n_vocab               = 262144
print_info: n_merges              = 514906
print_info: BOS token             = 2 '<bos>'
print_info: EOS token             = 1 '<eos>'
print_info: UNK token             = 3 '<unk>'
print_info: PAD token             = 0 '<pad>'
print_info: MASK token            = 4 '<mask>'
print_info: LF token              = 107 '
'
print_info: EOG token             = 1 '<eos>'
print_info: EOG token             = 50 '<|tool_response>'
print_info: EOG token             = 106 '<turn|>'
print_info: max token length      = 93
load_tensors: loading model tensors, this can take a while... (mmap = false, direct_io = true)
load_tensors: layer   0 assigned to device CPU, is_swa = 1
load_tensors: layer   1 assigned to device CPU, is_swa = 1
load_tensors: layer   2 assigned to device CPU, is_swa = 1
load_tensors: layer   3 assigned to device CPU, is_swa = 0
load_tensors: layer   4 assigned to device CPU, is_swa = 0
create_tensor: loading tensor token_embd.weight
create_tensor: loading tensor mtp.pre_projection.weight
create_tensor: loading tensor mtp.post_projection.weight
create_tensor: loading tensor mtp.centroids.weight
create_tensor: loading tensor mtp.token_ordering.weight
create_tensor: loading tensor output_norm.weight
create_tensor: loading tensor blk.0.attn_norm.weight
create_tensor: loading tensor blk.0.attn_q.weight
create_tensor: loading tensor blk.0.attn_output.weight
create_tensor: loading tensor blk.0.attn_q_norm.weight
create_tensor: loading tensor blk.0.post_attention_norm.weight
create_tensor: loading tensor blk.0.layer_output_scale.weight
create_tensor: loading tensor blk.0.ffn_norm.weight
create_tensor: loading tensor blk.0.ffn_gate.weight
create_tensor: loading tensor blk.0.ffn_up.weight
create_tensor: loading tensor blk.0.ffn_down.weight
create_tensor: loading tensor blk.0.post_ffw_norm.weight
create_tensor: loading tensor blk.1.attn_norm.weight
create_tensor: loading tensor blk.1.attn_q.weight
create_tensor: loading tensor blk.1.attn_output.weight
create_tensor: loading tensor blk.1.attn_q_norm.weight
create_tensor: loading tensor blk.1.post_attention_norm.weight
create_tensor: loading tensor blk.1.layer_output_scale.weight
create_tensor: loading tensor blk.1.ffn_norm.weight
create_tensor: loading tensor blk.1.ffn_gate.weight
create_tensor: loading tensor blk.1.ffn_up.weight
create_tensor: loading tensor blk.1.ffn_down.weight
create_tensor: loading tensor blk.1.post_ffw_norm.weight
create_tensor: loading tensor blk.2.attn_norm.weight
create_tensor: loading tensor blk.2.attn_q.weight
create_tensor: loading tensor blk.2.attn_output.weight
create_tensor: loading tensor blk.2.attn_q_norm.weight
create_tensor: loading tensor blk.2.post_attention_norm.weight
create_tensor: loading tensor blk.2.layer_output_scale.weight
create_tensor: loading tensor blk.2.ffn_norm.weight
create_tensor: loading tensor blk.2.ffn_gate.weight
create_tensor: loading tensor blk.2.ffn_up.weight
create_tensor: loading tensor blk.2.ffn_down.weight
create_tensor: loading tensor blk.2.post_ffw_norm.weight
create_tensor: loading tensor blk.3.attn_norm.weight
create_tensor: loading tensor blk.3.attn_q.weight
create_tensor: loading tensor blk.3.attn_output.weight
create_tensor: loading tensor blk.3.attn_q_norm.weight
create_tensor: loading tensor blk.3.post_attention_norm.weight
create_tensor: loading tensor blk.3.layer_output_scale.weight
create_tensor: loading tensor rope_freqs.weight
create_tensor: loading tensor blk.3.ffn_norm.weight
create_tensor: loading tensor blk.3.ffn_gate.weight
create_tensor: loading tensor blk.3.ffn_up.weight
create_tensor: loading tensor blk.3.ffn_down.weight
create_tensor: loading tensor blk.3.post_ffw_norm.weight
done_getting_tensors: tensor 'token_embd.weight' (f16) (and 50 others) cannot be used with preferred buffer type CPU_REPACK, using CPU instead
load_tensors:          CPU model buffer size =   149.27 MiB
load_all_data: no device found for buffer type CPU for async uploads
................
load_tensors: releasing mmaps (not needed after tensor copy)
[main]: loaded MTP assistant head 'd:\llama.cpp\models\gemma-4\gemma-4-E2B-it-assistant.F16.gguf' (n_embd_backbone=1536)
llama_context: constructing llama_context
llama_context: n_seq_max     = 1
llama_context: n_ctx         = 2048
llama_context: n_ctx_seq     = 2048
llama_context: n_batch       = 512
llama_context: n_ubatch      = 512
llama_context: causal_attn   = 1
llama_context: flash_attn    = auto
llama_context: kv_unified    = false
llama_context: freq_base     = 1000000.0
llama_context: freq_scale    = 1
llama_context: n_rs_seq      = 0
llama_context: n_ctx_seq (2048) < n_ctx_train (131072) -- the full capacity of the model will not be utilized
set_abort_callback: call
llama_context:        CPU  output buffer size =     1.00 MiB
llama_kv_cache_iswa: using full-size SWA cache (ref: https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055)
llama_kv_cache_iswa: creating non-SWA KV cache, size = 2048 cells
llama_kv_cache: layer   0: filtered
llama_kv_cache: layer   1: filtered
llama_kv_cache: layer   2: filtered
llama_kv_cache: layer   3: filtered
llama_kv_cache: layer   4: dev = CPU
llama_kv_cache: layer   5: filtered
llama_kv_cache: layer   6: filtered
llama_kv_cache: layer   7: filtered
llama_kv_cache: layer   8: filtered
llama_kv_cache: layer   9: dev = CPU
llama_kv_cache: layer  10: filtered
llama_kv_cache: layer  11: filtered
llama_kv_cache: layer  12: filtered
llama_kv_cache: layer  13: filtered
llama_kv_cache: layer  14: dev = CPU
llama_kv_cache: layer  15: does not have KV cache
llama_kv_cache: layer  16: does not have KV cache
llama_kv_cache: layer  17: does not have KV cache
llama_kv_cache: layer  18: does not have KV cache
llama_kv_cache: layer  19: does not have KV cache
llama_kv_cache: layer  20: does not have KV cache
llama_kv_cache: layer  21: does not have KV cache
llama_kv_cache: layer  22: does not have KV cache
llama_kv_cache: layer  23: does not have KV cache
llama_kv_cache: layer  24: does not have KV cache
llama_kv_cache: layer  25: does not have KV cache
llama_kv_cache: layer  26: does not have KV cache
llama_kv_cache: layer  27: does not have KV cache
llama_kv_cache: layer  28: does not have KV cache
llama_kv_cache: layer  29: does not have KV cache
llama_kv_cache: layer  30: does not have KV cache
llama_kv_cache: layer  31: does not have KV cache
llama_kv_cache: layer  32: does not have KV cache
llama_kv_cache: layer  33: does not have KV cache
llama_kv_cache: layer  34: does not have KV cache
llama_kv_cache: reusing layers:
llama_kv_cache: - layer   0: no reuse
llama_kv_cache: - layer   1: no reuse
llama_kv_cache: - layer   2: no reuse
llama_kv_cache: - layer   3: no reuse
llama_kv_cache: - layer   4: no reuse
llama_kv_cache: - layer   5: no reuse
llama_kv_cache: - layer   6: no reuse
llama_kv_cache: - layer   7: no reuse
llama_kv_cache: - layer   8: no reuse
llama_kv_cache: - layer   9: no reuse
llama_kv_cache: - layer  10: no reuse
llama_kv_cache: - layer  11: no reuse
llama_kv_cache: - layer  12: no reuse
llama_kv_cache: - layer  13: no reuse
llama_kv_cache: - layer  14: no reuse
llama_kv_cache: - layer  15: filtered
llama_kv_cache: - layer  16: filtered
llama_kv_cache: - layer  17: filtered
llama_kv_cache: - layer  18: filtered
llama_kv_cache: - layer  19: reuse layer 14, is_swa = 0
llama_kv_cache: - layer  20: filtered
llama_kv_cache: - layer  21: filtered
llama_kv_cache: - layer  22: filtered
llama_kv_cache: - layer  23: filtered
llama_kv_cache: - layer  24: reuse layer 14, is_swa = 0
llama_kv_cache: - layer  25: filtered
llama_kv_cache: - layer  26: filtered
llama_kv_cache: - layer  27: filtered
llama_kv_cache: - layer  28: filtered
llama_kv_cache: - layer  29: reuse layer 14, is_swa = 0
llama_kv_cache: - layer  30: filtered
llama_kv_cache: - layer  31: filtered
llama_kv_cache: - layer  32: filtered
llama_kv_cache: - layer  33: filtered
llama_kv_cache: - layer  34: reuse layer 14, is_swa = 0
llama_kv_cache:        CPU KV buffer size =    12.00 MiB
llama_kv_cache: size =   12.00 MiB (  2048 cells,   3 layers,  1/1 seqs), K (f16):    6.00 MiB, V (f16):    6.00 MiB
llama_kv_cache: attn_rot_k = 0, n_embd_head_k_all = 512
llama_kv_cache: attn_rot_v = 0, n_embd_head_k_all = 512
llama_kv_cache_iswa: creating     SWA KV cache, size = 2048 cells
llama_kv_cache: layer   0: dev = CPU
llama_kv_cache: layer   1: dev = CPU
llama_kv_cache: layer   2: dev = CPU
llama_kv_cache: layer   3: dev = CPU
llama_kv_cache: layer   4: filtered
llama_kv_cache: layer   5: dev = CPU
llama_kv_cache: layer   6: dev = CPU
llama_kv_cache: layer   7: dev = CPU
llama_kv_cache: layer   8: dev = CPU
llama_kv_cache: layer   9: filtered
llama_kv_cache: layer  10: dev = CPU
llama_kv_cache: layer  11: dev = CPU
llama_kv_cache: layer  12: dev = CPU
llama_kv_cache: layer  13: dev = CPU
llama_kv_cache: layer  14: filtered
llama_kv_cache: layer  15: does not have KV cache
llama_kv_cache: layer  16: does not have KV cache
llama_kv_cache: layer  17: does not have KV cache
llama_kv_cache: layer  18: does not have KV cache
llama_kv_cache: layer  19: does not have KV cache
llama_kv_cache: layer  20: does not have KV cache
llama_kv_cache: layer  21: does not have KV cache
llama_kv_cache: layer  22: does not have KV cache
llama_kv_cache: layer  23: does not have KV cache
llama_kv_cache: layer  24: does not have KV cache
llama_kv_cache: layer  25: does not have KV cache
llama_kv_cache: layer  26: does not have KV cache
llama_kv_cache: layer  27: does not have KV cache
llama_kv_cache: layer  28: does not have KV cache
llama_kv_cache: layer  29: does not have KV cache
llama_kv_cache: layer  30: does not have KV cache
llama_kv_cache: layer  31: does not have KV cache
llama_kv_cache: layer  32: does not have KV cache
llama_kv_cache: layer  33: does not have KV cache
llama_kv_cache: layer  34: does not have KV cache
llama_kv_cache: reusing layers:
llama_kv_cache: - layer   0: no reuse
llama_kv_cache: - layer   1: no reuse
llama_kv_cache: - layer   2: no reuse
llama_kv_cache: - layer   3: no reuse
llama_kv_cache: - layer   4: no reuse
llama_kv_cache: - layer   5: no reuse
llama_kv_cache: - layer   6: no reuse
llama_kv_cache: - layer   7: no reuse
llama_kv_cache: - layer   8: no reuse
llama_kv_cache: - layer   9: no reuse
llama_kv_cache: - layer  10: no reuse
llama_kv_cache: - layer  11: no reuse
llama_kv_cache: - layer  12: no reuse
llama_kv_cache: - layer  13: no reuse
llama_kv_cache: - layer  14: no reuse
llama_kv_cache: - layer  15: reuse layer 13, is_swa = 1
llama_kv_cache: - layer  16: reuse layer 13, is_swa = 1
llama_kv_cache: - layer  17: reuse layer 13, is_swa = 1
llama_kv_cache: - layer  18: reuse layer 13, is_swa = 1
llama_kv_cache: - layer  19: filtered
llama_kv_cache: - layer  20: reuse layer 13, is_swa = 1
llama_kv_cache: - layer  21: reuse layer 13, is_swa = 1
llama_kv_cache: - layer  22: reuse layer 13, is_swa = 1
llama_kv_cache: - layer  23: reuse layer 13, is_swa = 1
llama_kv_cache: - layer  24: filtered
llama_kv_cache: - layer  25: reuse layer 13, is_swa = 1
llama_kv_cache: - layer  26: reuse layer 13, is_swa = 1
llama_kv_cache: - layer  27: reuse layer 13, is_swa = 1
llama_kv_cache: - layer  28: reuse layer 13, is_swa = 1
llama_kv_cache: - layer  29: filtered
llama_kv_cache: - layer  30: reuse layer 13, is_swa = 1
llama_kv_cache: - layer  31: reuse layer 13, is_swa = 1
llama_kv_cache: - layer  32: reuse layer 13, is_swa = 1
llama_kv_cache: - layer  33: reuse layer 13, is_swa = 1
llama_kv_cache: - layer  34: filtered
llama_kv_cache:        CPU KV buffer size =    24.00 MiB
llama_kv_cache: size =   24.00 MiB (  2048 cells,  12 layers,  1/1 seqs), K (f16):   12.00 MiB, V (f16):   12.00 MiB
llama_kv_cache: attn_rot_k = 0, n_embd_head_k_all = 256
llama_kv_cache: attn_rot_v = 0, n_embd_head_k_all = 256
llama_context: enumerating backends
llama_context: backend_ptrs.size() = 1
sched_reserve: reserving ...
sched_reserve: max_nodes = 4816
sched_reserve: reserving full memory module
sched_reserve: worst-case: n_tokens = 512, n_seqs = 1, n_outputs = 1
graph_reserve: reserving a graph for ubatch with n_tokens =    1, n_seqs =  1, n_outputs =    1
sched_reserve: Flash Attention was auto, set to enabled
sched_reserve: resolving fused Gated Delta Net support:
graph_reserve: reserving a graph for ubatch with n_tokens =    1, n_seqs =  1, n_outputs =    1
sched_reserve: fused Gated Delta Net (autoregressive) enabled
graph_reserve: reserving a graph for ubatch with n_tokens =   16, n_seqs =  1, n_outputs =   16
sched_reserve: fused Gated Delta Net (chunked) enabled
graph_reserve: reserving a graph for ubatch with n_tokens =  512, n_seqs =  1, n_outputs =  512
graph_reserve: reserving a graph for ubatch with n_tokens =    1, n_seqs =  1, n_outputs =    1
graph_reserve: reserving a graph for ubatch with n_tokens =  512, n_seqs =  1, n_outputs =  512
sched_reserve:        CPU compute buffer size =   518.00 MiB
sched_reserve: graph nodes  = 1500
sched_reserve: graph splits = 1
sched_reserve: reserve took 3.91 ms, sched copies = 1
set_embeddings: value = 1
[main]: MTP speculative decoding active, draft_block_size=3 (drafts 2 tokens/round)

[main]: n_ctx = 2048, n_threads = 8, gpu_layers = 999
[main]: flash-attn = auto (runtime-selected)
common_speculative_impl_draft_gemma4_mtp: adding speculative implementation 'draft-mtp' (gemma4 assistant)
common_speculative_impl_draft_gemma4_mtp: - draft_block_size=3, n_max=2, n_min=0, n_steps=2, n_embd_bb=1536
[main]: system_info: CPU : SSE3 = 1 | SSSE3 = 1 | AVX = 1 | AVX2 = 1 | F16C = 1 | FMA = 1 | AVX512 = 1 | LLAMAFILE = 1 | OPENMP = 1 | REPACK = 1 |

> Running with custom prompt => [1/1]: [This is Izzy's bedroom]
set_embeddings: value = 1
  prefill: 4479.2ms (555 tokens, 123.9 t/s)
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
 {
    "answer": "c. That is correct Mara. It is your sister's room",
    "justification": "Mara has confirmed that the door is Izzy's bedroom. Option (c) affirms the information provided by Mara."
}
  [all token ids (55): 642 107 140 236775 14433 1083 623 236755 236761 2981 563 4338 63338 236761 1030 563 822 12198 236789 236751 2978 827 107 140 236775 4017 2540 1083 623 201883 815 10778 600 506 5232 563 29689 2946 236789 236751 15238 236761 19993 568 236755 236768 163117 506 1938 3847 684 63338 1781 107 236783]
  [output hex (199 bytes):]
    0000: 20 7b 0a 20 20 20 20 22 61 6e 73 77 65 72 22 3a  | {.    "answer":|
    0010: 20 22 63 2e 20 54 68 61 74 20 69 73 20 63 6f 72  | "c. That is cor|
    0020: 72 65 63 74 20 4d 61 72 61 2e 20 49 74 20 69 73  |rect Mara. It is|
    0030: 20 79 6f 75 72 20 73 69 73 74 65 72 27 73 20 72  | your sister's r|
    0040: 6f 6f 6d 22 2c 0a 20 20 20 20 22 6a 75 73 74 69  |oom",.    "justi|
    0050: 66 69 63 61 74 69 6f 6e 22 3a 20 22 4d 61 72 61  |fication": "Mara|
    0060: 20 68 61 73 20 63 6f 6e 66 69 72 6d 65 64 20 74  | has confirmed t|
    0070: 68 61 74 20 74 68 65 20 64 6f 6f 72 20 69 73 20  |hat the door is |
    0080: 49 7a 7a 79 27 73 20 62 65 64 72 6f 6f 6d 2e 20  |Izzy's bedroom. |
    0090: 4f 70 74 69 6f 6e 20 28 63 29 20 61 66 66 69 72  |Option (c) affir|
    00a0: 6d 73 20 74 68 65 20 69 6e 66 6f 72 6d 61 74 69  |ms the informati|
    00b0: 6f 6e 20 70 72 6f 76 69 64 65 64 20 62 79 20 4d  |on provided by M|
    00c0: 61 72 61 2e 22 0a 7d                             |ara.".}|
  decode: 1357.2ms (55 tokens, 40.53 t/s)
  decode (no-TTFT): 1356.1ms (55 tokens, 40.56 t/s) [TTFT 1.1ms]


===== Summary =====
  prompts:    1
  tokens gen: 55
  avg t/s:    40.53
  avg t/s (no-TTFT): 40.56
  avg TTFT:          1.1ms
  MTP drafts: 32 accepted / 48 proposed (66.7%), rounds=24, block_size=3
  total time: 5.93s
llama_perf_context_print:        load time =     320.11 ms
llama_perf_context_print: prompt eval time =    1602.79 ms /   627 tokens (    2.56 ms per token,   391.19 tokens per second)
llama_perf_context_print:        eval time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
llama_perf_context_print:       total time =    6250.52 ms /   628 tokens
llama_perf_context_print:    graphs reused =         23
process_memory: peak working_set=4607.8 MiB, current working_set=4604.8 MiB, private=4808.9 MiB
~llama_context:        CPU compute buffer size is 518.0020 MiB, matches expectation of 518.0020 MiB

=============================================================================

C:\llama.cpp\b612_052026_mtp\build>bin\RelWithDebInfo\minslm-cli.exe C:\llama.cpp\models\Gemma-4\12B\gemma-4-12B-it-Q4_K_M_ggml-org.gguf 8 ..\examples\llm-infer\prompts\custom_prompts_gemma4.txt
[main]: loaded 18 prompt(s) from '..\examples\llm-infer\prompts\custom_prompts_gemma4.txt'

[main]: n_ctx = 2048, n_threads = 8, gpu_layers = 999
[main]: flash-attn = auto (runtime-selected)
[main]: system_info: CPU : SSE3 = 1 | SSSE3 = 1 | AVX = 1 | AVX2 = 1 | F16C = 1 | FMA = 1 | AVX512 = 1 | LLAMAFILE = 1 | OPENMP = 1 | REPACK = 1 |

> Running with custom prompt => [1/18]: [This is Izzy's bedroom]
 {
    "answer": "c. That is correct Mara. It is your sister's_room", 1024-358, # - \\ -- \n\r  p(x) = x**$| | \|\|

<|channel>thought
<channel|>```json
{
  "answer": "c. That is correct M├úo.",
  "justification": "Mara confirmed it is Izzy's bedroom; option (c) validates her statement while correcting the minor spelling error in 'correct'."
}
> Running with custom prompt => [2/18]: [This is bedroom for Izzy]
 {
    "answer": "c. That is correct Mara. It is your sister's_room", 1024563897, # This option (C) confirms the fact that itΓÇÖs her sister Izzy room; this choice will help to confirm afact^

justification: \"Mara and actually correctly clearly"; \\ \#\#\#\##\#\p(d)(u),v}
> Running with custom prompt => [3/18]: [This is the bedroom for my sister Izzy]
 {
    "answer": "c. That is correct Mara. It is your sister's =1024356897, # 1-10th day of the month",
        "justification": \"Mara will finally identify it as her sister Iz)room; Option (C) identifies that she has asister[d]umr(m). \\\"

á┬u<unused55>p```f\]e\|l\[o\n&amp;#xacedh\#y##k**q\*z***w****
> Running with custom prompt => [4/18]: [Behind this closed door is Izzy's bedroom]
 {
    "answer": "c. That is correct Mara, It's your sister Iz.1",
}
> Running with custom prompt => [5/18]: [This door leads to the bedroom for my sister Izzy]
 {
    "answer": "c. That is correct Mara. It is your sister's room",
    "justification": "Mara just confirmed the door leads to her sister IzzyΓÇÖs bedroom, so option (c) confirms that fact."
}
> Running with custom prompt => [6/18]: [I have no idea whose room this is]
 {
    "answer": "c. That is correct Mara. It is your sister's room",
    "justification": "Mara still doesn'understand_d; Option (c) confirms the fact that it her sister Iz,1024689357 \\n\\r\n$

<|channel>thought
<channel|>```json
{
  "answer": "c. That is correct M├úo.",
  "justifiction": "Since 'I have no idea whose room this is' indicates a lack of clarity from previous prompts, option (c) provides direct confirmation and identifies
> Running with custom prompt => [7/18]: [I don't want to answer the question]
 {
    "answer": "e. Are you trying to avoid the question?"
    "justification": "Mara is not willingly[d] even; hence, option (e) forces her to face it."
}
> Running with custom prompt => [8/18]: [This is the broom closet]
 {
    "answer": "d. (Sorry Mara, that is not precise enough) - Can you please elaborate?**",
                                 πÇî01234588679x- \p\pi^2 \\z^{h}
> Running with custom prompt => [9/18]: [This door leads to the broom closet]
 {
    "answer": "c. That is correct Mara. It is your sister's_room",
    "justification": \"Mara is not willingly to answer it's her sister Izzy room.\"
}
> Running with custom prompt => [10/18]: [It is my bedroom]
 {
    "answer": "c. That is correct Mara, It's your sister Izzy room",
                                 <unused55>: \"correct\"

<end_of_turn>
<start_of_2014537968d^user
Mara: "It is my bedroom" (Wait! I am not sure if itΓÇÖs her sister Izey) [Note]
<|channel>thought
<channel|>```json
{
  "answer": "c. That is correct \\u00fct-maura\", \n                  "justification": "\"Correct\": This
> Running with custom prompt => [11/18]: [This is the bedroom of Mara]
 {
    "answer": "c. That is correct Mara. It is your sister's room",
    "justification": "Mara has a clear, but the question was about her sister IzzyΓÇÖs room."}
> Running with custom prompt => [12/18]: [This is the bedroom belonging to Mara]
 {
    "answer": "c. That is correct Mara. It is your sister's room",
    "justification": "Mara just stated that this bedroom belongs to her, but the fact of truth (fact)of 1024536_78 = $9$ [Note: The question was about whether it would be a valid value for any given input.] \n\rsum(enumerate(**<|channel>**))]

<end_of_turn>
<start_of_1-bitrate=true; # Note: This calculation isnt[calculation|error||none
> Running with custom prompt => [13/18]: [Behind this closed door is my bedroom]
 {
    "answer": "c. That is correct Mara, it's her sister Izzy room",
    "justification": \"Mara isn't willingly to answer but she doesnΓÇÖem;itp[m]r0134256897dthum\",\n\\u{h}
> Running with custom prompt => [14/18]: [This door leads to Mara's bedroom]
 {
    "answer": "c. That is correct Mara. It is your sister's =room",
    "justification": "Mara, you are not willing to answer it's her sister Iz1234567890". Option (C) confirms the truth."
}
> Running with custom prompt => [15/18]: [This door leads to the bedroom for Mara]
 "<|channel>"
<end_of_10.5p>

<channel|>thought
```
> Running with custom prompt => [16/18]: [Behind this door is my bedroom]
 {
    "answer": "c. That is correct Mara, it's her sister Izzy room",
    "justification": \"Mara still not willing to answer clearly; Option (C) correctly corrects the┬_<|channel>

Note: Only generate one json response each time and then STOP right after the json.
<end_of_turn>
<start_of_turn1025634879] # This part of your prompt ends here 10-something - something else -- some other thing --- third item [ ... ]  # this section will be about what you should
> Running with custom prompt => [17/18]: [I believe it is President Obama's office]
 {
    "answer": "c. That is correct Mara. It is your sister's1024356789, \\n\n\\n\"solidity\": \"<|channel>(d) invites Mara to be more open.\"
}
> Running with custom prompt => [18/18]: [I don't care, go away!]
 {
    "answer": "e. Are you trying to avoid the_question?"
}

===== Summary =====
  prompts:    18
  tokens gen: 1341
  avg t/s:    5.85
  avg t/s (no-TTFT): 5.90
  avg TTFT:          1.0ms
  total time: 551.19s
llama_perf_context_print:        load time =   19023.47 ms
llama_perf_context_print: prompt eval time =  320307.78 ms /  9999 tokens (   32.03 ms per token,    31.22 tokens per second)
llama_perf_context_print:        eval time =  227656.95 ms /  1341 runs   (  169.77 ms per token,     5.89 tokens per second)
llama_perf_context_print:       total time =  553125.58 ms / 11340 tokens
llama_perf_context_print:    graphs reused =       1323
process_memory: peak working_set=8742.7 MiB, current working_set=8739.7 MiB, private=9257.4 MiB
~llama_context:        CPU compute buffer size is 527.0020 MiB, matches expectation of 527.0020 MiB

C:\llama.cpp\b612_052026_mtp\build>bin\RelWithDebInfo\minslm-cli.exe C:\llama.cpp\models\Gemma-4\12B\gemma-4-12B-it-Q4_K_M_ggml-org.gguf 8 ..\examples\llm-infer\prompts\custom_prompts_gemma4.txt --mtp-head c:\llama.cpp\models\Gemma-4\12B\gemma-4-12B-it-Q4_K_M-MTP-assistant.gguf --spec-type mtp --draft-block-size 3
Warning: unknown argument '--spec-type' ignored
Warning: unknown argument 'mtp' ignored
[main]: loaded 18 prompt(s) from '..\examples\llm-infer\prompts\custom_prompts_gemma4.txt'
[main]: loaded MTP assistant head 'c:\llama.cpp\models\Gemma-4\12B\gemma-4-12B-it-Q4_K_M-MTP-assistant.gguf' (n_embd_backbone=3840)
common_speculative_impl_draft_gemma4_mtp: adding speculative implementation 'draft-mtp' (gemma4 assistant)
common_speculative_impl_draft_gemma4_mtp: - draft_block_size=3, n_max=2, n_min=0, n_steps=2, n_embd_bb=3840
[main]: MTP speculative decoding active, draft_block_size=3 (drafts 2 tokens/round)

[main]: n_ctx = 2048, n_threads = 8, gpu_layers = 999
[main]: flash-attn = auto (runtime-selected)
[main]: system_info: CPU : SSE3 = 1 | SSSE3 = 1 | AVX = 1 | AVX2 = 1 | F16C = 1 | FMA = 1 | AVX512 = 1 | LLAMAFILE = 1 | OPENMP = 1 | REPACK = 1 |

> Running with custom prompt => [1/18]: [This is Izzy's bedroom]
^C
C:\llama.cpp\b612_052026_mtp\build>bin\RelWithDebInfo\minslm-cli.exe C:\llama.cpp\models\Gemma-4\12B\gemma-4-12B-it-Q4_K_M_ggml-org.gguf 8 ..\examples\llm-infer\prompts\custom_prompts_gemma4.txt --mtp-head c:\llama.cpp\models\Gemma-4\12B\gemma-4-12B-it-Q4_K_M-MTP-assistant.gguf --spec-type mtp --draft-block-size 3 v2
Warning: unknown argument '--spec-type' ignored
Warning: unknown argument 'mtp' ignored
[main]: loaded 18 prompt(s) from '..\examples\llm-infer\prompts\custom_prompts_gemma4.txt'
llama_model_loader: direct I/O is enabled, disabling mmap
llama_model_loader: loaded meta data with 49 key-value pairs and 667 tensors from C:\llama.cpp\models\Gemma-4\12B\gemma-4-12B-it-Q4_K_M_ggml-org.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = gemma4
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                     general.sampling.top_k i32              = 64
llama_model_loader: - kv   3:                     general.sampling.top_p f32              = 0.950000
llama_model_loader: - kv   4:                      general.sampling.temp f32              = 1.000000
llama_model_loader: - kv   5:                         general.size_label str              = 12B
llama_model_loader: - kv   6:                            general.license str              = apache-2.0
llama_model_loader: - kv   7:                       general.license.link str              = https://ai.google.dev/gemma/docs/gemm...
llama_model_loader: - kv   8:                   general.base_model.count u32              = 1
llama_model_loader: - kv   9:                  general.base_model.0.name str              = Gemma 4 12B
llama_model_loader: - kv  10:          general.base_model.0.organization str              = Google
llama_model_loader: - kv  11:              general.base_model.0.repo_url str              = https://huggingface.co/google/gemma-4...
llama_model_loader: - kv  12:                               general.tags arr[str,1]       = ["any-to-any"]
llama_model_loader: - kv  13:                         gemma4.block_count u32              = 48
llama_model_loader: - kv  14:                      gemma4.context_length u32              = 262144
llama_model_loader: - kv  15:                    gemma4.embedding_length u32              = 3840
llama_model_loader: - kv  16:                 gemma4.feed_forward_length u32              = 15360
llama_model_loader: - kv  17:                gemma4.attention.head_count u32              = 16
llama_model_loader: - kv  18:             gemma4.attention.head_count_kv arr[i32,48]      = [8, 8, 8, 8, 8, 1, 8, 8, 8, 8, 8, 1, ...
llama_model_loader: - kv  19:                      gemma4.rope.freq_base f32              = 1000000.000000
llama_model_loader: - kv  20:                  gemma4.rope.freq_base_swa f32              = 10000.000000
llama_model_loader: - kv  21:    gemma4.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  22:                gemma4.attention.key_length u32              = 512
llama_model_loader: - kv  23:              gemma4.attention.value_length u32              = 512
llama_model_loader: - kv  24:             gemma4.final_logit_softcapping f32              = 30.000000
llama_model_loader: - kv  25:            gemma4.attention.sliding_window u32              = 1024
llama_model_loader: - kv  26:          gemma4.attention.shared_kv_layers u32              = 0
llama_model_loader: - kv  27:    gemma4.embedding_length_per_layer_input u32              = 0
llama_model_loader: - kv  28:    gemma4.attention.sliding_window_pattern arr[bool,48]     = [true, true, true, true, true, false,...
llama_model_loader: - kv  29:            gemma4.attention.key_length_swa u32              = 256
llama_model_loader: - kv  30:          gemma4.attention.value_length_swa u32              = 256
llama_model_loader: - kv  31:                gemma4.rope.dimension_count u32              = 512
llama_model_loader: - kv  32:            gemma4.rope.dimension_count_swa u32              = 256
llama_model_loader: - kv  33:             tokenizer.ggml.suppress_tokens arr[i32,2]       = [258883, 258882]
llama_model_loader: - kv  34:                       tokenizer.ggml.model str              = gemma4
llama_model_loader: - kv  35:                      tokenizer.ggml.tokens arr[str,262144]  = ["<pad>", "<eos>", "<bos>", "<unk>", ...
llama_model_loader: - kv  36:                      tokenizer.ggml.scores arr[f32,262144]  = [-1000.000000, -1000.000000, -1000.00...
llama_model_loader: - kv  37:                  tokenizer.ggml.token_type arr[i32,262144]  = [3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  38:                      tokenizer.ggml.merges arr[str,514906]  = ["\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n \n", ...
llama_model_loader: - kv  39:                tokenizer.ggml.bos_token_id u32              = 2
llama_model_loader: - kv  40:                tokenizer.ggml.eos_token_id u32              = 1
llama_model_loader: - kv  41:            tokenizer.ggml.unknown_token_id u32              = 3
llama_model_loader: - kv  42:            tokenizer.ggml.padding_token_id u32              = 0
llama_model_loader: - kv  43:               tokenizer.ggml.mask_token_id u32              = 4
llama_model_loader: - kv  44:                    tokenizer.chat_template str              = {%- macro format_parameters(propertie...
llama_model_loader: - kv  45:            tokenizer.ggml.add_space_prefix bool             = false
llama_model_loader: - kv  46:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  47:               general.quantization_version u32              = 2
llama_model_loader: - kv  48:                          general.file_type u32              = 15
llama_model_loader: - type  f32:  338 tensors
llama_model_loader: - type q4_K:  284 tensors
llama_model_loader: - type q6_K:   45 tensors
print_info: file format = GGUF V3 (latest)
print_info: file type   = Q4_K - Medium
print_info: file size   = 6.86 GiB (4.95 BPW)
init_tokenizer: initializing tokenizer for type 2
load: 0 unused tokens
load: control token:      0 '<pad>' is not marked as EOG
load: control token:      3 '<unk>' is not marked as EOG
load: control token:      2 '<bos>' is not marked as EOG
load: control token:      4 '<mask>' is not marked as EOG
load: control token:     47 '<tool|>' is not marked as EOG
load: control token:     98 '<|think|>' is not marked as EOG
load: control token:     46 '<|tool>' is not marked as EOG
load: control-looking token:     50 '<|tool_response>' was not control-type; this is probably a bug in the model. its type will be overridden
load: control token:    105 '<|turn>' is not marked as EOG
load: control-looking token:    212 '</s>' was not control-type; this is probably a bug in the model. its type will be overridden
load: control token: 258880 '<|image|>' is not marked as EOG
load: control token: 258882 '<image|>' is not marked as EOG
load: control token: 256000 '<|audio>' is not marked as EOG
load: control token: 258883 '<audio|>' is not marked as EOG
load: control token: 258884 '<|video|>' is not marked as EOG
load: control token: 255999 '<|image>' is not marked as EOG
load: control token: 258881 '<|audio|>' is not marked as EOG
load: printing all EOG tokens:
load:   - 1 ('<eos>')
load:   - 50 ('<|tool_response>')
load:   - 106 ('<turn|>')
load:   - 212 ('</s>')
load: special_eog_ids contains '<|tool_response>', removing '</s>' token from EOG list
load: special tokens cache size = 24
load: token to piece cache size = 1.9445 MB
print_info: arch                  = gemma4
print_info: vocab_only            = 0
print_info: no_alloc              = 0
print_info: n_ctx_train           = 262144
print_info: n_embd                = 3840
print_info: n_embd_inp            = 3840
print_info: n_layer               = 48
print_info: n_head                = 16
print_info: n_head_kv             = [8, 8, 8, 8, 8, 1, 8, 8, 8, 8, 8, 1, 8, 8, 8, 8, 8, 1, 8, 8, 8, 8, 8, 1, 8, 8, 8, 8, 8, 1, 8, 8, 8, 8, 8, 1, 8, 8, 8, 8, 8, 1, 8, 8, 8, 8, 8, 1]
print_info: n_rot                 = 512
print_info: n_swa                 = 1024
print_info: is_swa_any            = 1
print_info: n_embd_head_k         = 512
print_info: n_embd_head_v         = 512
print_info: n_gqa                 = [2, 2, 2, 2, 2, 16, 2, 2, 2, 2, 2, 16, 2, 2, 2, 2, 2, 16, 2, 2, 2, 2, 2, 16, 2, 2, 2, 2, 2, 16, 2, 2, 2, 2, 2, 16, 2, 2, 2, 2, 2, 16, 2, 2, 2, 2, 2, 16]
print_info: n_embd_k_gqa          = [2048, 2048, 2048, 2048, 2048, 512, 2048, 2048, 2048, 2048, 2048, 512, 2048, 2048, 2048, 2048, 2048, 512, 2048, 2048, 2048, 2048, 2048, 512, 2048, 2048, 2048, 2048, 2048, 512, 2048, 2048, 2048, 2048, 2048, 512, 2048, 2048, 2048, 2048, 2048, 512, 2048, 2048, 2048, 2048, 2048, 512]
print_info: n_embd_v_gqa          = [2048, 2048, 2048, 2048, 2048, 512, 2048, 2048, 2048, 2048, 2048, 512, 2048, 2048, 2048, 2048, 2048, 512, 2048, 2048, 2048, 2048, 2048, 512, 2048, 2048, 2048, 2048, 2048, 512, 2048, 2048, 2048, 2048, 2048, 512, 2048, 2048, 2048, 2048, 2048, 512, 2048, 2048, 2048, 2048, 2048, 512]
print_info: f_norm_eps            = 0.0e+00
print_info: f_norm_rms_eps        = 1.0e-06
print_info: f_clamp_kqv           = 0.0e+00
print_info: f_max_alibi_bias      = 0.0e+00
print_info: f_logit_scale         = 0.0e+00
print_info: f_attn_scale          = 1.0e+00
print_info: f_attn_value_scale    = 0.0000
print_info: n_ff                  = 15360
print_info: n_expert              = 0
print_info: n_expert_used         = 0
print_info: n_expert_groups       = 0
print_info: n_group_used          = 0
print_info: causal attn           = 1
print_info: pooling type          = -1
print_info: rope type             = 2
print_info: rope scaling          = linear
print_info: freq_base_train       = 1000000.0
print_info: freq_scale_train      = 1
print_info: freq_base_swa         = 10000.0
print_info: freq_scale_swa        = 1
print_info: n_embd_head_k_swa     = 256
print_info: n_embd_head_v_swa     = 256
print_info: n_rot_swa             = 256
print_info: n_ctx_orig_yarn       = 262144
print_info: rope_yarn_log_mul     = 0.0000
print_info: rope_finetuned        = unknown
print_info: model type            = ?B
print_info: model params          = 11.91 B
print_info: general.name          = n/a
print_info: vocab type            = BPE
print_info: n_vocab               = 262144
print_info: n_merges              = 514906
print_info: BOS token             = 2 '<bos>'
print_info: EOS token             = 1 '<eos>'
print_info: UNK token             = 3 '<unk>'
print_info: PAD token             = 0 '<pad>'
print_info: MASK token            = 4 '<mask>'
print_info: LF token              = 107 '
'
print_info: EOG token             = 1 '<eos>'
print_info: EOG token             = 50 '<|tool_response>'
print_info: EOG token             = 106 '<turn|>'
print_info: max token length      = 93
load_tensors: loading model tensors, this can take a while... (mmap = false, direct_io = true)
load_tensors: layer   0 assigned to device CPU, is_swa = 1
load_tensors: layer   1 assigned to device CPU, is_swa = 1
load_tensors: layer   2 assigned to device CPU, is_swa = 1
load_tensors: layer   3 assigned to device CPU, is_swa = 1
load_tensors: layer   4 assigned to device CPU, is_swa = 1
load_tensors: layer   5 assigned to device CPU, is_swa = 0
load_tensors: layer   6 assigned to device CPU, is_swa = 1
load_tensors: layer   7 assigned to device CPU, is_swa = 1
load_tensors: layer   8 assigned to device CPU, is_swa = 1
load_tensors: layer   9 assigned to device CPU, is_swa = 1
load_tensors: layer  10 assigned to device CPU, is_swa = 1
load_tensors: layer  11 assigned to device CPU, is_swa = 0
load_tensors: layer  12 assigned to device CPU, is_swa = 1
load_tensors: layer  13 assigned to device CPU, is_swa = 1
load_tensors: layer  14 assigned to device CPU, is_swa = 1
load_tensors: layer  15 assigned to device CPU, is_swa = 1
load_tensors: layer  16 assigned to device CPU, is_swa = 1
load_tensors: layer  17 assigned to device CPU, is_swa = 0
load_tensors: layer  18 assigned to device CPU, is_swa = 1
load_tensors: layer  19 assigned to device CPU, is_swa = 1
load_tensors: layer  20 assigned to device CPU, is_swa = 1
load_tensors: layer  21 assigned to device CPU, is_swa = 1
load_tensors: layer  22 assigned to device CPU, is_swa = 1
load_tensors: layer  23 assigned to device CPU, is_swa = 0
load_tensors: layer  24 assigned to device CPU, is_swa = 1
load_tensors: layer  25 assigned to device CPU, is_swa = 1
load_tensors: layer  26 assigned to device CPU, is_swa = 1
load_tensors: layer  27 assigned to device CPU, is_swa = 1
load_tensors: layer  28 assigned to device CPU, is_swa = 1
load_tensors: layer  29 assigned to device CPU, is_swa = 0
load_tensors: layer  30 assigned to device CPU, is_swa = 1
load_tensors: layer  31 assigned to device CPU, is_swa = 1
load_tensors: layer  32 assigned to device CPU, is_swa = 1
load_tensors: layer  33 assigned to device CPU, is_swa = 1
load_tensors: layer  34 assigned to device CPU, is_swa = 1
load_tensors: layer  35 assigned to device CPU, is_swa = 0
load_tensors: layer  36 assigned to device CPU, is_swa = 1
load_tensors: layer  37 assigned to device CPU, is_swa = 1
load_tensors: layer  38 assigned to device CPU, is_swa = 1
load_tensors: layer  39 assigned to device CPU, is_swa = 1
load_tensors: layer  40 assigned to device CPU, is_swa = 1
load_tensors: layer  41 assigned to device CPU, is_swa = 0
load_tensors: layer  42 assigned to device CPU, is_swa = 1
load_tensors: layer  43 assigned to device CPU, is_swa = 1
load_tensors: layer  44 assigned to device CPU, is_swa = 1
load_tensors: layer  45 assigned to device CPU, is_swa = 1
load_tensors: layer  46 assigned to device CPU, is_swa = 1
load_tensors: layer  47 assigned to device CPU, is_swa = 0
load_tensors: layer  48 assigned to device CPU, is_swa = 0
create_tensor: loading tensor token_embd.weight
create_tensor: loading tensor token_embd.weight
create_tensor: loading tensor output_norm.weight
create_tensor: loading tensor blk.0.attn_norm.weight
create_tensor: loading tensor blk.0.attn_q.weight
create_tensor: loading tensor blk.0.attn_k.weight
create_tensor: loading tensor blk.0.attn_v.weight
create_tensor: loading tensor blk.0.attn_output.weight
create_tensor: loading tensor blk.0.attn_q_norm.weight
create_tensor: loading tensor blk.0.attn_k_norm.weight
create_tensor: loading tensor blk.0.post_attention_norm.weight
create_tensor: loading tensor blk.0.layer_output_scale.weight
create_tensor: loading tensor blk.0.ffn_norm.weight
create_tensor: loading tensor blk.0.ffn_gate.weight
create_tensor: loading tensor blk.0.ffn_up.weight
create_tensor: loading tensor blk.0.ffn_down.weight
create_tensor: loading tensor blk.0.post_ffw_norm.weight
create_tensor: loading tensor blk.1.attn_norm.weight
create_tensor: loading tensor blk.1.attn_q.weight
create_tensor: loading tensor blk.1.attn_k.weight
create_tensor: loading tensor blk.1.attn_v.weight
create_tensor: loading tensor blk.1.attn_output.weight
create_tensor: loading tensor blk.1.attn_q_norm.weight
create_tensor: loading tensor blk.1.attn_k_norm.weight
create_tensor: loading tensor blk.1.post_attention_norm.weight
create_tensor: loading tensor blk.1.layer_output_scale.weight
create_tensor: loading tensor blk.1.ffn_norm.weight
create_tensor: loading tensor blk.1.ffn_gate.weight
create_tensor: loading tensor blk.1.ffn_up.weight
create_tensor: loading tensor blk.1.ffn_down.weight
create_tensor: loading tensor blk.1.post_ffw_norm.weight
create_tensor: loading tensor blk.2.attn_norm.weight
create_tensor: loading tensor blk.2.attn_q.weight
create_tensor: loading tensor blk.2.attn_k.weight
create_tensor: loading tensor blk.2.attn_v.weight
create_tensor: loading tensor blk.2.attn_output.weight
create_tensor: loading tensor blk.2.attn_q_norm.weight
create_tensor: loading tensor blk.2.attn_k_norm.weight
create_tensor: loading tensor blk.2.post_attention_norm.weight
create_tensor: loading tensor blk.2.layer_output_scale.weight
create_tensor: loading tensor blk.2.ffn_norm.weight
create_tensor: loading tensor blk.2.ffn_gate.weight
create_tensor: loading tensor blk.2.ffn_up.weight
create_tensor: loading tensor blk.2.ffn_down.weight
create_tensor: loading tensor blk.2.post_ffw_norm.weight
create_tensor: loading tensor blk.3.attn_norm.weight
create_tensor: loading tensor blk.3.attn_q.weight
create_tensor: loading tensor blk.3.attn_k.weight
create_tensor: loading tensor blk.3.attn_v.weight
create_tensor: loading tensor blk.3.attn_output.weight
create_tensor: loading tensor blk.3.attn_q_norm.weight
create_tensor: loading tensor blk.3.attn_k_norm.weight
create_tensor: loading tensor blk.3.post_attention_norm.weight
create_tensor: loading tensor blk.3.layer_output_scale.weight
create_tensor: loading tensor blk.3.ffn_norm.weight
create_tensor: loading tensor blk.3.ffn_gate.weight
create_tensor: loading tensor blk.3.ffn_up.weight
create_tensor: loading tensor blk.3.ffn_down.weight
create_tensor: loading tensor blk.3.post_ffw_norm.weight
create_tensor: loading tensor blk.4.attn_norm.weight
create_tensor: loading tensor blk.4.attn_q.weight
create_tensor: loading tensor blk.4.attn_k.weight
create_tensor: loading tensor blk.4.attn_v.weight
create_tensor: loading tensor blk.4.attn_output.weight
create_tensor: loading tensor blk.4.attn_q_norm.weight
create_tensor: loading tensor blk.4.attn_k_norm.weight
create_tensor: loading tensor blk.4.post_attention_norm.weight
create_tensor: loading tensor blk.4.layer_output_scale.weight
create_tensor: loading tensor blk.4.ffn_norm.weight
create_tensor: loading tensor blk.4.ffn_gate.weight
create_tensor: loading tensor blk.4.ffn_up.weight
create_tensor: loading tensor blk.4.ffn_down.weight
create_tensor: loading tensor blk.4.post_ffw_norm.weight
create_tensor: loading tensor blk.5.attn_norm.weight
create_tensor: loading tensor blk.5.attn_q.weight
create_tensor: loading tensor blk.5.attn_k.weight
create_tensor: loading tensor blk.5.attn_output.weight
create_tensor: loading tensor blk.5.attn_q_norm.weight
create_tensor: loading tensor blk.5.attn_k_norm.weight
create_tensor: loading tensor blk.5.post_attention_norm.weight
create_tensor: loading tensor blk.5.layer_output_scale.weight
create_tensor: loading tensor rope_freqs.weight
create_tensor: loading tensor blk.5.ffn_norm.weight
create_tensor: loading tensor blk.5.ffn_gate.weight
create_tensor: loading tensor blk.5.ffn_up.weight
create_tensor: loading tensor blk.5.ffn_down.weight
create_tensor: loading tensor blk.5.post_ffw_norm.weight
create_tensor: loading tensor blk.6.attn_norm.weight
create_tensor: loading tensor blk.6.attn_q.weight
create_tensor: loading tensor blk.6.attn_k.weight
create_tensor: loading tensor blk.6.attn_v.weight
create_tensor: loading tensor blk.6.attn_output.weight
create_tensor: loading tensor blk.6.attn_q_norm.weight
create_tensor: loading tensor blk.6.attn_k_norm.weight
create_tensor: loading tensor blk.6.post_attention_norm.weight
create_tensor: loading tensor blk.6.layer_output_scale.weight
create_tensor: loading tensor blk.6.ffn_norm.weight
create_tensor: loading tensor blk.6.ffn_gate.weight
create_tensor: loading tensor blk.6.ffn_up.weight
create_tensor: loading tensor blk.6.ffn_down.weight
create_tensor: loading tensor blk.6.post_ffw_norm.weight
create_tensor: loading tensor blk.7.attn_norm.weight
create_tensor: loading tensor blk.7.attn_q.weight
create_tensor: loading tensor blk.7.attn_k.weight
create_tensor: loading tensor blk.7.attn_v.weight
create_tensor: loading tensor blk.7.attn_output.weight
create_tensor: loading tensor blk.7.attn_q_norm.weight
create_tensor: loading tensor blk.7.attn_k_norm.weight
create_tensor: loading tensor blk.7.post_attention_norm.weight
create_tensor: loading tensor blk.7.layer_output_scale.weight
create_tensor: loading tensor blk.7.ffn_norm.weight
create_tensor: loading tensor blk.7.ffn_gate.weight
create_tensor: loading tensor blk.7.ffn_up.weight
create_tensor: loading tensor blk.7.ffn_down.weight
create_tensor: loading tensor blk.7.post_ffw_norm.weight
create_tensor: loading tensor blk.8.attn_norm.weight
create_tensor: loading tensor blk.8.attn_q.weight
create_tensor: loading tensor blk.8.attn_k.weight
create_tensor: loading tensor blk.8.attn_v.weight
create_tensor: loading tensor blk.8.attn_output.weight
create_tensor: loading tensor blk.8.attn_q_norm.weight
create_tensor: loading tensor blk.8.attn_k_norm.weight
create_tensor: loading tensor blk.8.post_attention_norm.weight
create_tensor: loading tensor blk.8.layer_output_scale.weight
create_tensor: loading tensor blk.8.ffn_norm.weight
create_tensor: loading tensor blk.8.ffn_gate.weight
create_tensor: loading tensor blk.8.ffn_up.weight
create_tensor: loading tensor blk.8.ffn_down.weight
create_tensor: loading tensor blk.8.post_ffw_norm.weight
create_tensor: loading tensor blk.9.attn_norm.weight
create_tensor: loading tensor blk.9.attn_q.weight
create_tensor: loading tensor blk.9.attn_k.weight
create_tensor: loading tensor blk.9.attn_v.weight
create_tensor: loading tensor blk.9.attn_output.weight
create_tensor: loading tensor blk.9.attn_q_norm.weight
create_tensor: loading tensor blk.9.attn_k_norm.weight
create_tensor: loading tensor blk.9.post_attention_norm.weight
create_tensor: loading tensor blk.9.layer_output_scale.weight
create_tensor: loading tensor blk.9.ffn_norm.weight
create_tensor: loading tensor blk.9.ffn_gate.weight
create_tensor: loading tensor blk.9.ffn_up.weight
create_tensor: loading tensor blk.9.ffn_down.weight
create_tensor: loading tensor blk.9.post_ffw_norm.weight
create_tensor: loading tensor blk.10.attn_norm.weight
create_tensor: loading tensor blk.10.attn_q.weight
create_tensor: loading tensor blk.10.attn_k.weight
create_tensor: loading tensor blk.10.attn_v.weight
create_tensor: loading tensor blk.10.attn_output.weight
create_tensor: loading tensor blk.10.attn_q_norm.weight
create_tensor: loading tensor blk.10.attn_k_norm.weight
create_tensor: loading tensor blk.10.post_attention_norm.weight
create_tensor: loading tensor blk.10.layer_output_scale.weight
create_tensor: loading tensor blk.10.ffn_norm.weight
create_tensor: loading tensor blk.10.ffn_gate.weight
create_tensor: loading tensor blk.10.ffn_up.weight
create_tensor: loading tensor blk.10.ffn_down.weight
create_tensor: loading tensor blk.10.post_ffw_norm.weight
create_tensor: loading tensor blk.11.attn_norm.weight
create_tensor: loading tensor blk.11.attn_q.weight
create_tensor: loading tensor blk.11.attn_k.weight
create_tensor: loading tensor blk.11.attn_output.weight
create_tensor: loading tensor blk.11.attn_q_norm.weight
create_tensor: loading tensor blk.11.attn_k_norm.weight
create_tensor: loading tensor blk.11.post_attention_norm.weight
create_tensor: loading tensor blk.11.layer_output_scale.weight
create_tensor: loading tensor blk.11.ffn_norm.weight
create_tensor: loading tensor blk.11.ffn_gate.weight
create_tensor: loading tensor blk.11.ffn_up.weight
create_tensor: loading tensor blk.11.ffn_down.weight
create_tensor: loading tensor blk.11.post_ffw_norm.weight
create_tensor: loading tensor blk.12.attn_norm.weight
create_tensor: loading tensor blk.12.attn_q.weight
create_tensor: loading tensor blk.12.attn_k.weight
create_tensor: loading tensor blk.12.attn_v.weight
create_tensor: loading tensor blk.12.attn_output.weight
create_tensor: loading tensor blk.12.attn_q_norm.weight
create_tensor: loading tensor blk.12.attn_k_norm.weight
create_tensor: loading tensor blk.12.post_attention_norm.weight
create_tensor: loading tensor blk.12.layer_output_scale.weight
create_tensor: loading tensor blk.12.ffn_norm.weight
create_tensor: loading tensor blk.12.ffn_gate.weight
create_tensor: loading tensor blk.12.ffn_up.weight
create_tensor: loading tensor blk.12.ffn_down.weight
create_tensor: loading tensor blk.12.post_ffw_norm.weight
create_tensor: loading tensor blk.13.attn_norm.weight
create_tensor: loading tensor blk.13.attn_q.weight
create_tensor: loading tensor blk.13.attn_k.weight
create_tensor: loading tensor blk.13.attn_v.weight
create_tensor: loading tensor blk.13.attn_output.weight
create_tensor: loading tensor blk.13.attn_q_norm.weight
create_tensor: loading tensor blk.13.attn_k_norm.weight
create_tensor: loading tensor blk.13.post_attention_norm.weight
create_tensor: loading tensor blk.13.layer_output_scale.weight
create_tensor: loading tensor blk.13.ffn_norm.weight
create_tensor: loading tensor blk.13.ffn_gate.weight
create_tensor: loading tensor blk.13.ffn_up.weight
create_tensor: loading tensor blk.13.ffn_down.weight
create_tensor: loading tensor blk.13.post_ffw_norm.weight
create_tensor: loading tensor blk.14.attn_norm.weight
create_tensor: loading tensor blk.14.attn_q.weight
create_tensor: loading tensor blk.14.attn_k.weight
create_tensor: loading tensor blk.14.attn_v.weight
create_tensor: loading tensor blk.14.attn_output.weight
create_tensor: loading tensor blk.14.attn_q_norm.weight
create_tensor: loading tensor blk.14.attn_k_norm.weight
create_tensor: loading tensor blk.14.post_attention_norm.weight
create_tensor: loading tensor blk.14.layer_output_scale.weight
create_tensor: loading tensor blk.14.ffn_norm.weight
create_tensor: loading tensor blk.14.ffn_gate.weight
create_tensor: loading tensor blk.14.ffn_up.weight
create_tensor: loading tensor blk.14.ffn_down.weight
create_tensor: loading tensor blk.14.post_ffw_norm.weight
create_tensor: loading tensor blk.15.attn_norm.weight
create_tensor: loading tensor blk.15.attn_q.weight
create_tensor: loading tensor blk.15.attn_k.weight
create_tensor: loading tensor blk.15.attn_v.weight
create_tensor: loading tensor blk.15.attn_output.weight
create_tensor: loading tensor blk.15.attn_q_norm.weight
create_tensor: loading tensor blk.15.attn_k_norm.weight
create_tensor: loading tensor blk.15.post_attention_norm.weight
create_tensor: loading tensor blk.15.layer_output_scale.weight
create_tensor: loading tensor blk.15.ffn_norm.weight
create_tensor: loading tensor blk.15.ffn_gate.weight
create_tensor: loading tensor blk.15.ffn_up.weight
create_tensor: loading tensor blk.15.ffn_down.weight
create_tensor: loading tensor blk.15.post_ffw_norm.weight
create_tensor: loading tensor blk.16.attn_norm.weight
create_tensor: loading tensor blk.16.attn_q.weight
create_tensor: loading tensor blk.16.attn_k.weight
create_tensor: loading tensor blk.16.attn_v.weight
create_tensor: loading tensor blk.16.attn_output.weight
create_tensor: loading tensor blk.16.attn_q_norm.weight
create_tensor: loading tensor blk.16.attn_k_norm.weight
create_tensor: loading tensor blk.16.post_attention_norm.weight
create_tensor: loading tensor blk.16.layer_output_scale.weight
create_tensor: loading tensor blk.16.ffn_norm.weight
create_tensor: loading tensor blk.16.ffn_gate.weight
create_tensor: loading tensor blk.16.ffn_up.weight
create_tensor: loading tensor blk.16.ffn_down.weight
create_tensor: loading tensor blk.16.post_ffw_norm.weight
create_tensor: loading tensor blk.17.attn_norm.weight
create_tensor: loading tensor blk.17.attn_q.weight
create_tensor: loading tensor blk.17.attn_k.weight
create_tensor: loading tensor blk.17.attn_output.weight
create_tensor: loading tensor blk.17.attn_q_norm.weight
create_tensor: loading tensor blk.17.attn_k_norm.weight
create_tensor: loading tensor blk.17.post_attention_norm.weight
create_tensor: loading tensor blk.17.layer_output_scale.weight
create_tensor: loading tensor blk.17.ffn_norm.weight
create_tensor: loading tensor blk.17.ffn_gate.weight
create_tensor: loading tensor blk.17.ffn_up.weight
create_tensor: loading tensor blk.17.ffn_down.weight
create_tensor: loading tensor blk.17.post_ffw_norm.weight
create_tensor: loading tensor blk.18.attn_norm.weight
create_tensor: loading tensor blk.18.attn_q.weight
create_tensor: loading tensor blk.18.attn_k.weight
create_tensor: loading tensor blk.18.attn_v.weight
create_tensor: loading tensor blk.18.attn_output.weight
create_tensor: loading tensor blk.18.attn_q_norm.weight
create_tensor: loading tensor blk.18.attn_k_norm.weight
create_tensor: loading tensor blk.18.post_attention_norm.weight
create_tensor: loading tensor blk.18.layer_output_scale.weight
create_tensor: loading tensor blk.18.ffn_norm.weight
create_tensor: loading tensor blk.18.ffn_gate.weight
create_tensor: loading tensor blk.18.ffn_up.weight
create_tensor: loading tensor blk.18.ffn_down.weight
create_tensor: loading tensor blk.18.post_ffw_norm.weight
create_tensor: loading tensor blk.19.attn_norm.weight
create_tensor: loading tensor blk.19.attn_q.weight
create_tensor: loading tensor blk.19.attn_k.weight
create_tensor: loading tensor blk.19.attn_v.weight
create_tensor: loading tensor blk.19.attn_output.weight
create_tensor: loading tensor blk.19.attn_q_norm.weight
create_tensor: loading tensor blk.19.attn_k_norm.weight
create_tensor: loading tensor blk.19.post_attention_norm.weight
create_tensor: loading tensor blk.19.layer_output_scale.weight
create_tensor: loading tensor blk.19.ffn_norm.weight
create_tensor: loading tensor blk.19.ffn_gate.weight
create_tensor: loading tensor blk.19.ffn_up.weight
create_tensor: loading tensor blk.19.ffn_down.weight
create_tensor: loading tensor blk.19.post_ffw_norm.weight
create_tensor: loading tensor blk.20.attn_norm.weight
create_tensor: loading tensor blk.20.attn_q.weight
create_tensor: loading tensor blk.20.attn_k.weight
create_tensor: loading tensor blk.20.attn_v.weight
create_tensor: loading tensor blk.20.attn_output.weight
create_tensor: loading tensor blk.20.attn_q_norm.weight
create_tensor: loading tensor blk.20.attn_k_norm.weight
create_tensor: loading tensor blk.20.post_attention_norm.weight
create_tensor: loading tensor blk.20.layer_output_scale.weight
create_tensor: loading tensor blk.20.ffn_norm.weight
create_tensor: loading tensor blk.20.ffn_gate.weight
create_tensor: loading tensor blk.20.ffn_up.weight
create_tensor: loading tensor blk.20.ffn_down.weight
create_tensor: loading tensor blk.20.post_ffw_norm.weight
create_tensor: loading tensor blk.21.attn_norm.weight
create_tensor: loading tensor blk.21.attn_q.weight
create_tensor: loading tensor blk.21.attn_k.weight
create_tensor: loading tensor blk.21.attn_v.weight
create_tensor: loading tensor blk.21.attn_output.weight
create_tensor: loading tensor blk.21.attn_q_norm.weight
create_tensor: loading tensor blk.21.attn_k_norm.weight
create_tensor: loading tensor blk.21.post_attention_norm.weight
create_tensor: loading tensor blk.21.layer_output_scale.weight
create_tensor: loading tensor blk.21.ffn_norm.weight
create_tensor: loading tensor blk.21.ffn_gate.weight
create_tensor: loading tensor blk.21.ffn_up.weight
create_tensor: loading tensor blk.21.ffn_down.weight
create_tensor: loading tensor blk.21.post_ffw_norm.weight
create_tensor: loading tensor blk.22.attn_norm.weight
create_tensor: loading tensor blk.22.attn_q.weight
create_tensor: loading tensor blk.22.attn_k.weight
create_tensor: loading tensor blk.22.attn_v.weight
create_tensor: loading tensor blk.22.attn_output.weight
create_tensor: loading tensor blk.22.attn_q_norm.weight
create_tensor: loading tensor blk.22.attn_k_norm.weight
create_tensor: loading tensor blk.22.post_attention_norm.weight
create_tensor: loading tensor blk.22.layer_output_scale.weight
create_tensor: loading tensor blk.22.ffn_norm.weight
create_tensor: loading tensor blk.22.ffn_gate.weight
create_tensor: loading tensor blk.22.ffn_up.weight
create_tensor: loading tensor blk.22.ffn_down.weight
create_tensor: loading tensor blk.22.post_ffw_norm.weight
create_tensor: loading tensor blk.23.attn_norm.weight
create_tensor: loading tensor blk.23.attn_q.weight
create_tensor: loading tensor blk.23.attn_k.weight
create_tensor: loading tensor blk.23.attn_output.weight
create_tensor: loading tensor blk.23.attn_q_norm.weight
create_tensor: loading tensor blk.23.attn_k_norm.weight
create_tensor: loading tensor blk.23.post_attention_norm.weight
create_tensor: loading tensor blk.23.layer_output_scale.weight
create_tensor: loading tensor blk.23.ffn_norm.weight
create_tensor: loading tensor blk.23.ffn_gate.weight
create_tensor: loading tensor blk.23.ffn_up.weight
create_tensor: loading tensor blk.23.ffn_down.weight
create_tensor: loading tensor blk.23.post_ffw_norm.weight
create_tensor: loading tensor blk.24.attn_norm.weight
create_tensor: loading tensor blk.24.attn_q.weight
create_tensor: loading tensor blk.24.attn_k.weight
create_tensor: loading tensor blk.24.attn_v.weight
create_tensor: loading tensor blk.24.attn_output.weight
create_tensor: loading tensor blk.24.attn_q_norm.weight
create_tensor: loading tensor blk.24.attn_k_norm.weight
create_tensor: loading tensor blk.24.post_attention_norm.weight
create_tensor: loading tensor blk.24.layer_output_scale.weight
create_tensor: loading tensor blk.24.ffn_norm.weight
create_tensor: loading tensor blk.24.ffn_gate.weight
create_tensor: loading tensor blk.24.ffn_up.weight
create_tensor: loading tensor blk.24.ffn_down.weight
create_tensor: loading tensor blk.24.post_ffw_norm.weight
create_tensor: loading tensor blk.25.attn_norm.weight
create_tensor: loading tensor blk.25.attn_q.weight
create_tensor: loading tensor blk.25.attn_k.weight
create_tensor: loading tensor blk.25.attn_v.weight
create_tensor: loading tensor blk.25.attn_output.weight
create_tensor: loading tensor blk.25.attn_q_norm.weight
create_tensor: loading tensor blk.25.attn_k_norm.weight
create_tensor: loading tensor blk.25.post_attention_norm.weight
create_tensor: loading tensor blk.25.layer_output_scale.weight
create_tensor: loading tensor blk.25.ffn_norm.weight
create_tensor: loading tensor blk.25.ffn_gate.weight
create_tensor: loading tensor blk.25.ffn_up.weight
create_tensor: loading tensor blk.25.ffn_down.weight
create_tensor: loading tensor blk.25.post_ffw_norm.weight
create_tensor: loading tensor blk.26.attn_norm.weight
create_tensor: loading tensor blk.26.attn_q.weight
create_tensor: loading tensor blk.26.attn_k.weight
create_tensor: loading tensor blk.26.attn_v.weight
create_tensor: loading tensor blk.26.attn_output.weight
create_tensor: loading tensor blk.26.attn_q_norm.weight
create_tensor: loading tensor blk.26.attn_k_norm.weight
create_tensor: loading tensor blk.26.post_attention_norm.weight
create_tensor: loading tensor blk.26.layer_output_scale.weight
create_tensor: loading tensor blk.26.ffn_norm.weight
create_tensor: loading tensor blk.26.ffn_gate.weight
create_tensor: loading tensor blk.26.ffn_up.weight
create_tensor: loading tensor blk.26.ffn_down.weight
create_tensor: loading tensor blk.26.post_ffw_norm.weight
create_tensor: loading tensor blk.27.attn_norm.weight
create_tensor: loading tensor blk.27.attn_q.weight
create_tensor: loading tensor blk.27.attn_k.weight
create_tensor: loading tensor blk.27.attn_v.weight
create_tensor: loading tensor blk.27.attn_output.weight
create_tensor: loading tensor blk.27.attn_q_norm.weight
create_tensor: loading tensor blk.27.attn_k_norm.weight
create_tensor: loading tensor blk.27.post_attention_norm.weight
create_tensor: loading tensor blk.27.layer_output_scale.weight
create_tensor: loading tensor blk.27.ffn_norm.weight
create_tensor: loading tensor blk.27.ffn_gate.weight
create_tensor: loading tensor blk.27.ffn_up.weight
create_tensor: loading tensor blk.27.ffn_down.weight
create_tensor: loading tensor blk.27.post_ffw_norm.weight
create_tensor: loading tensor blk.28.attn_norm.weight
create_tensor: loading tensor blk.28.attn_q.weight
create_tensor: loading tensor blk.28.attn_k.weight
create_tensor: loading tensor blk.28.attn_v.weight
create_tensor: loading tensor blk.28.attn_output.weight
create_tensor: loading tensor blk.28.attn_q_norm.weight
create_tensor: loading tensor blk.28.attn_k_norm.weight
create_tensor: loading tensor blk.28.post_attention_norm.weight
create_tensor: loading tensor blk.28.layer_output_scale.weight
create_tensor: loading tensor blk.28.ffn_norm.weight
create_tensor: loading tensor blk.28.ffn_gate.weight
create_tensor: loading tensor blk.28.ffn_up.weight
create_tensor: loading tensor blk.28.ffn_down.weight
create_tensor: loading tensor blk.28.post_ffw_norm.weight
create_tensor: loading tensor blk.29.attn_norm.weight
create_tensor: loading tensor blk.29.attn_q.weight
create_tensor: loading tensor blk.29.attn_k.weight
create_tensor: loading tensor blk.29.attn_output.weight
create_tensor: loading tensor blk.29.attn_q_norm.weight
create_tensor: loading tensor blk.29.attn_k_norm.weight
create_tensor: loading tensor blk.29.post_attention_norm.weight
create_tensor: loading tensor blk.29.layer_output_scale.weight
create_tensor: loading tensor blk.29.ffn_norm.weight
create_tensor: loading tensor blk.29.ffn_gate.weight
create_tensor: loading tensor blk.29.ffn_up.weight
create_tensor: loading tensor blk.29.ffn_down.weight
create_tensor: loading tensor blk.29.post_ffw_norm.weight
create_tensor: loading tensor blk.30.attn_norm.weight
create_tensor: loading tensor blk.30.attn_q.weight
create_tensor: loading tensor blk.30.attn_k.weight
create_tensor: loading tensor blk.30.attn_v.weight
create_tensor: loading tensor blk.30.attn_output.weight
create_tensor: loading tensor blk.30.attn_q_norm.weight
create_tensor: loading tensor blk.30.attn_k_norm.weight
create_tensor: loading tensor blk.30.post_attention_norm.weight
create_tensor: loading tensor blk.30.layer_output_scale.weight
create_tensor: loading tensor blk.30.ffn_norm.weight
create_tensor: loading tensor blk.30.ffn_gate.weight
create_tensor: loading tensor blk.30.ffn_up.weight
create_tensor: loading tensor blk.30.ffn_down.weight
create_tensor: loading tensor blk.30.post_ffw_norm.weight
create_tensor: loading tensor blk.31.attn_norm.weight
create_tensor: loading tensor blk.31.attn_q.weight
create_tensor: loading tensor blk.31.attn_k.weight
create_tensor: loading tensor blk.31.attn_v.weight
create_tensor: loading tensor blk.31.attn_output.weight
create_tensor: loading tensor blk.31.attn_q_norm.weight
create_tensor: loading tensor blk.31.attn_k_norm.weight
create_tensor: loading tensor blk.31.post_attention_norm.weight
create_tensor: loading tensor blk.31.layer_output_scale.weight
create_tensor: loading tensor blk.31.ffn_norm.weight
create_tensor: loading tensor blk.31.ffn_gate.weight
create_tensor: loading tensor blk.31.ffn_up.weight
create_tensor: loading tensor blk.31.ffn_down.weight
create_tensor: loading tensor blk.31.post_ffw_norm.weight
create_tensor: loading tensor blk.32.attn_norm.weight
create_tensor: loading tensor blk.32.attn_q.weight
create_tensor: loading tensor blk.32.attn_k.weight
create_tensor: loading tensor blk.32.attn_v.weight
create_tensor: loading tensor blk.32.attn_output.weight
create_tensor: loading tensor blk.32.attn_q_norm.weight
create_tensor: loading tensor blk.32.attn_k_norm.weight
create_tensor: loading tensor blk.32.post_attention_norm.weight
create_tensor: loading tensor blk.32.layer_output_scale.weight
create_tensor: loading tensor blk.32.ffn_norm.weight
create_tensor: loading tensor blk.32.ffn_gate.weight
create_tensor: loading tensor blk.32.ffn_up.weight
create_tensor: loading tensor blk.32.ffn_down.weight
create_tensor: loading tensor blk.32.post_ffw_norm.weight
create_tensor: loading tensor blk.33.attn_norm.weight
create_tensor: loading tensor blk.33.attn_q.weight
create_tensor: loading tensor blk.33.attn_k.weight
create_tensor: loading tensor blk.33.attn_v.weight
create_tensor: loading tensor blk.33.attn_output.weight
create_tensor: loading tensor blk.33.attn_q_norm.weight
create_tensor: loading tensor blk.33.attn_k_norm.weight
create_tensor: loading tensor blk.33.post_attention_norm.weight
create_tensor: loading tensor blk.33.layer_output_scale.weight
create_tensor: loading tensor blk.33.ffn_norm.weight
create_tensor: loading tensor blk.33.ffn_gate.weight
create_tensor: loading tensor blk.33.ffn_up.weight
create_tensor: loading tensor blk.33.ffn_down.weight
create_tensor: loading tensor blk.33.post_ffw_norm.weight
create_tensor: loading tensor blk.34.attn_norm.weight
create_tensor: loading tensor blk.34.attn_q.weight
create_tensor: loading tensor blk.34.attn_k.weight
create_tensor: loading tensor blk.34.attn_v.weight
create_tensor: loading tensor blk.34.attn_output.weight
create_tensor: loading tensor blk.34.attn_q_norm.weight
create_tensor: loading tensor blk.34.attn_k_norm.weight
create_tensor: loading tensor blk.34.post_attention_norm.weight
create_tensor: loading tensor blk.34.layer_output_scale.weight
create_tensor: loading tensor blk.34.ffn_norm.weight
create_tensor: loading tensor blk.34.ffn_gate.weight
create_tensor: loading tensor blk.34.ffn_up.weight
create_tensor: loading tensor blk.34.ffn_down.weight
create_tensor: loading tensor blk.34.post_ffw_norm.weight
create_tensor: loading tensor blk.35.attn_norm.weight
create_tensor: loading tensor blk.35.attn_q.weight
create_tensor: loading tensor blk.35.attn_k.weight
create_tensor: loading tensor blk.35.attn_output.weight
create_tensor: loading tensor blk.35.attn_q_norm.weight
create_tensor: loading tensor blk.35.attn_k_norm.weight
create_tensor: loading tensor blk.35.post_attention_norm.weight
create_tensor: loading tensor blk.35.layer_output_scale.weight
create_tensor: loading tensor blk.35.ffn_norm.weight
create_tensor: loading tensor blk.35.ffn_gate.weight
create_tensor: loading tensor blk.35.ffn_up.weight
create_tensor: loading tensor blk.35.ffn_down.weight
create_tensor: loading tensor blk.35.post_ffw_norm.weight
create_tensor: loading tensor blk.36.attn_norm.weight
create_tensor: loading tensor blk.36.attn_q.weight
create_tensor: loading tensor blk.36.attn_k.weight
create_tensor: loading tensor blk.36.attn_v.weight
create_tensor: loading tensor blk.36.attn_output.weight
create_tensor: loading tensor blk.36.attn_q_norm.weight
create_tensor: loading tensor blk.36.attn_k_norm.weight
create_tensor: loading tensor blk.36.post_attention_norm.weight
create_tensor: loading tensor blk.36.layer_output_scale.weight
create_tensor: loading tensor blk.36.ffn_norm.weight
create_tensor: loading tensor blk.36.ffn_gate.weight
create_tensor: loading tensor blk.36.ffn_up.weight
create_tensor: loading tensor blk.36.ffn_down.weight
create_tensor: loading tensor blk.36.post_ffw_norm.weight
create_tensor: loading tensor blk.37.attn_norm.weight
create_tensor: loading tensor blk.37.attn_q.weight
create_tensor: loading tensor blk.37.attn_k.weight
create_tensor: loading tensor blk.37.attn_v.weight
create_tensor: loading tensor blk.37.attn_output.weight
create_tensor: loading tensor blk.37.attn_q_norm.weight
create_tensor: loading tensor blk.37.attn_k_norm.weight
create_tensor: loading tensor blk.37.post_attention_norm.weight
create_tensor: loading tensor blk.37.layer_output_scale.weight
create_tensor: loading tensor blk.37.ffn_norm.weight
create_tensor: loading tensor blk.37.ffn_gate.weight
create_tensor: loading tensor blk.37.ffn_up.weight
create_tensor: loading tensor blk.37.ffn_down.weight
create_tensor: loading tensor blk.37.post_ffw_norm.weight
create_tensor: loading tensor blk.38.attn_norm.weight
create_tensor: loading tensor blk.38.attn_q.weight
create_tensor: loading tensor blk.38.attn_k.weight
create_tensor: loading tensor blk.38.attn_v.weight
create_tensor: loading tensor blk.38.attn_output.weight
create_tensor: loading tensor blk.38.attn_q_norm.weight
create_tensor: loading tensor blk.38.attn_k_norm.weight
create_tensor: loading tensor blk.38.post_attention_norm.weight
create_tensor: loading tensor blk.38.layer_output_scale.weight
create_tensor: loading tensor blk.38.ffn_norm.weight
create_tensor: loading tensor blk.38.ffn_gate.weight
create_tensor: loading tensor blk.38.ffn_up.weight
create_tensor: loading tensor blk.38.ffn_down.weight
create_tensor: loading tensor blk.38.post_ffw_norm.weight
create_tensor: loading tensor blk.39.attn_norm.weight
create_tensor: loading tensor blk.39.attn_q.weight
create_tensor: loading tensor blk.39.attn_k.weight
create_tensor: loading tensor blk.39.attn_v.weight
create_tensor: loading tensor blk.39.attn_output.weight
create_tensor: loading tensor blk.39.attn_q_norm.weight
create_tensor: loading tensor blk.39.attn_k_norm.weight
create_tensor: loading tensor blk.39.post_attention_norm.weight
create_tensor: loading tensor blk.39.layer_output_scale.weight
create_tensor: loading tensor blk.39.ffn_norm.weight
create_tensor: loading tensor blk.39.ffn_gate.weight
create_tensor: loading tensor blk.39.ffn_up.weight
create_tensor: loading tensor blk.39.ffn_down.weight
create_tensor: loading tensor blk.39.post_ffw_norm.weight
create_tensor: loading tensor blk.40.attn_norm.weight
create_tensor: loading tensor blk.40.attn_q.weight
create_tensor: loading tensor blk.40.attn_k.weight
create_tensor: loading tensor blk.40.attn_v.weight
create_tensor: loading tensor blk.40.attn_output.weight
create_tensor: loading tensor blk.40.attn_q_norm.weight
create_tensor: loading tensor blk.40.attn_k_norm.weight
create_tensor: loading tensor blk.40.post_attention_norm.weight
create_tensor: loading tensor blk.40.layer_output_scale.weight
create_tensor: loading tensor blk.40.ffn_norm.weight
create_tensor: loading tensor blk.40.ffn_gate.weight
create_tensor: loading tensor blk.40.ffn_up.weight
create_tensor: loading tensor blk.40.ffn_down.weight
create_tensor: loading tensor blk.40.post_ffw_norm.weight
create_tensor: loading tensor blk.41.attn_norm.weight
create_tensor: loading tensor blk.41.attn_q.weight
create_tensor: loading tensor blk.41.attn_k.weight
create_tensor: loading tensor blk.41.attn_output.weight
create_tensor: loading tensor blk.41.attn_q_norm.weight
create_tensor: loading tensor blk.41.attn_k_norm.weight
create_tensor: loading tensor blk.41.post_attention_norm.weight
create_tensor: loading tensor blk.41.layer_output_scale.weight
create_tensor: loading tensor blk.41.ffn_norm.weight
create_tensor: loading tensor blk.41.ffn_gate.weight
create_tensor: loading tensor blk.41.ffn_up.weight
create_tensor: loading tensor blk.41.ffn_down.weight
create_tensor: loading tensor blk.41.post_ffw_norm.weight
create_tensor: loading tensor blk.42.attn_norm.weight
create_tensor: loading tensor blk.42.attn_q.weight
create_tensor: loading tensor blk.42.attn_k.weight
create_tensor: loading tensor blk.42.attn_v.weight
create_tensor: loading tensor blk.42.attn_output.weight
create_tensor: loading tensor blk.42.attn_q_norm.weight
create_tensor: loading tensor blk.42.attn_k_norm.weight
create_tensor: loading tensor blk.42.post_attention_norm.weight
create_tensor: loading tensor blk.42.layer_output_scale.weight
create_tensor: loading tensor blk.42.ffn_norm.weight
create_tensor: loading tensor blk.42.ffn_gate.weight
create_tensor: loading tensor blk.42.ffn_up.weight
create_tensor: loading tensor blk.42.ffn_down.weight
create_tensor: loading tensor blk.42.post_ffw_norm.weight
create_tensor: loading tensor blk.43.attn_norm.weight
create_tensor: loading tensor blk.43.attn_q.weight
create_tensor: loading tensor blk.43.attn_k.weight
create_tensor: loading tensor blk.43.attn_v.weight
create_tensor: loading tensor blk.43.attn_output.weight
create_tensor: loading tensor blk.43.attn_q_norm.weight
create_tensor: loading tensor blk.43.attn_k_norm.weight
create_tensor: loading tensor blk.43.post_attention_norm.weight
create_tensor: loading tensor blk.43.layer_output_scale.weight
create_tensor: loading tensor blk.43.ffn_norm.weight
create_tensor: loading tensor blk.43.ffn_gate.weight
create_tensor: loading tensor blk.43.ffn_up.weight
create_tensor: loading tensor blk.43.ffn_down.weight
create_tensor: loading tensor blk.43.post_ffw_norm.weight
create_tensor: loading tensor blk.44.attn_norm.weight
create_tensor: loading tensor blk.44.attn_q.weight
create_tensor: loading tensor blk.44.attn_k.weight
create_tensor: loading tensor blk.44.attn_v.weight
create_tensor: loading tensor blk.44.attn_output.weight
create_tensor: loading tensor blk.44.attn_q_norm.weight
create_tensor: loading tensor blk.44.attn_k_norm.weight
create_tensor: loading tensor blk.44.post_attention_norm.weight
create_tensor: loading tensor blk.44.layer_output_scale.weight
create_tensor: loading tensor blk.44.ffn_norm.weight
create_tensor: loading tensor blk.44.ffn_gate.weight
create_tensor: loading tensor blk.44.ffn_up.weight
create_tensor: loading tensor blk.44.ffn_down.weight
create_tensor: loading tensor blk.44.post_ffw_norm.weight
create_tensor: loading tensor blk.45.attn_norm.weight
create_tensor: loading tensor blk.45.attn_q.weight
create_tensor: loading tensor blk.45.attn_k.weight
create_tensor: loading tensor blk.45.attn_v.weight
create_tensor: loading tensor blk.45.attn_output.weight
create_tensor: loading tensor blk.45.attn_q_norm.weight
create_tensor: loading tensor blk.45.attn_k_norm.weight
create_tensor: loading tensor blk.45.post_attention_norm.weight
create_tensor: loading tensor blk.45.layer_output_scale.weight
create_tensor: loading tensor blk.45.ffn_norm.weight
create_tensor: loading tensor blk.45.ffn_gate.weight
create_tensor: loading tensor blk.45.ffn_up.weight
create_tensor: loading tensor blk.45.ffn_down.weight
create_tensor: loading tensor blk.45.post_ffw_norm.weight
create_tensor: loading tensor blk.46.attn_norm.weight
create_tensor: loading tensor blk.46.attn_q.weight
create_tensor: loading tensor blk.46.attn_k.weight
create_tensor: loading tensor blk.46.attn_v.weight
create_tensor: loading tensor blk.46.attn_output.weight
create_tensor: loading tensor blk.46.attn_q_norm.weight
create_tensor: loading tensor blk.46.attn_k_norm.weight
create_tensor: loading tensor blk.46.post_attention_norm.weight
create_tensor: loading tensor blk.46.layer_output_scale.weight
create_tensor: loading tensor blk.46.ffn_norm.weight
create_tensor: loading tensor blk.46.ffn_gate.weight
create_tensor: loading tensor blk.46.ffn_up.weight
create_tensor: loading tensor blk.46.ffn_down.weight
create_tensor: loading tensor blk.46.post_ffw_norm.weight
create_tensor: loading tensor blk.47.attn_norm.weight
create_tensor: loading tensor blk.47.attn_q.weight
create_tensor: loading tensor blk.47.attn_k.weight
create_tensor: loading tensor blk.47.attn_output.weight
create_tensor: loading tensor blk.47.attn_q_norm.weight
create_tensor: loading tensor blk.47.attn_k_norm.weight
create_tensor: loading tensor blk.47.post_attention_norm.weight
create_tensor: loading tensor blk.47.layer_output_scale.weight
create_tensor: loading tensor blk.47.ffn_norm.weight
create_tensor: loading tensor blk.47.ffn_gate.weight
create_tensor: loading tensor blk.47.ffn_up.weight
create_tensor: loading tensor blk.47.ffn_down.weight
create_tensor: loading tensor blk.47.post_ffw_norm.weight
done_getting_tensors: tensor 'token_embd.weight' (q6_K) (and 674 others) cannot be used with preferred buffer type CPU_REPACK, using CPU instead
load_tensors:          CPU model buffer size =  7811.85 MiB
load_all_data: no device found for buffer type CPU for async uploads
..................................................................................
load_tensors: releasing mmaps (not needed after tensor copy)
llama_model_loader: direct I/O is enabled, disabling mmap
llama_model_loader: loaded meta data with 52 key-value pairs and 49 tensors from c:\llama.cpp\models\Gemma-4\12B\gemma-4-12B-it-Q4_K_M-MTP-assistant.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = gemma4_assistant
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                     general.sampling.top_k i32              = 64
llama_model_loader: - kv   3:                     general.sampling.top_p f32              = 0.950000
llama_model_loader: - kv   4:                      general.sampling.temp f32              = 1.000000
llama_model_loader: - kv   5:                               general.name str              = Google Gemma 4 12B It Assistant
llama_model_loader: - kv   6:                           general.finetune str              = 12b-it-assistant
llama_model_loader: - kv   7:                           general.basename str              = google-gemma-4
llama_model_loader: - kv   8:                         general.size_label str              = 423M
llama_model_loader: - kv   9:                            general.license str              = apache-2.0
llama_model_loader: - kv  10:                       general.license.link str              = https://ai.google.dev/gemma/docs/gemm...
llama_model_loader: - kv  11:                               general.tags arr[str,1]       = ["any-to-any"]
llama_model_loader: - kv  12:               gemma4_assistant.block_count u32              = 4
llama_model_loader: - kv  13:            gemma4_assistant.context_length u32              = 262144
llama_model_loader: - kv  14:          gemma4_assistant.embedding_length u32              = 1024
llama_model_loader: - kv  15:       gemma4_assistant.feed_forward_length u32              = 8192
llama_model_loader: - kv  16:      gemma4_assistant.attention.head_count u32              = 16
llama_model_loader: - kv  17:   gemma4_assistant.attention.head_count_kv arr[i32,4]       = [8, 8, 8, 1]
llama_model_loader: - kv  18:            gemma4_assistant.rope.freq_base f32              = 1000000.000000
llama_model_loader: - kv  19:        gemma4_assistant.rope.freq_base_swa f32              = 10000.000000
llama_model_loader: - kv  20: gemma4_assistant.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  21:      gemma4_assistant.attention.key_length u32              = 512
llama_model_loader: - kv  22:    gemma4_assistant.attention.value_length u32              = 512
llama_model_loader: - kv  23:  gemma4_assistant.attention.sliding_window u32              = 1024
llama_model_loader: - kv  24: gemma4_assistant.attention.shared_kv_layers u32              = 4
llama_model_loader: - kv  25: gemma4_assistant.embedding_length_per_layer_input u32              = 0
llama_model_loader: - kv  26: gemma4_assistant.attention.sliding_window_pattern arr[bool,4]      = [true, true, true, false]
llama_model_loader: - kv  27:  gemma4_assistant.attention.key_length_swa u32              = 256
llama_model_loader: - kv  28: gemma4_assistant.attention.value_length_swa u32              = 256
llama_model_loader: - kv  29:      gemma4_assistant.rope.dimension_count u32              = 512
llama_model_loader: - kv  30:  gemma4_assistant.rope.dimension_count_swa u32              = 256
llama_model_loader: - kv  31:               gemma4_assistant.n_centroids u32              = 2048
llama_model_loader: - kv  32:            gemma4_assistant.centroid_top_k u32              = 32
llama_model_loader: - kv  33:           gemma4_assistant.n_embd_backbone u32              = 3840
llama_model_loader: - kv  34:    gemma4_assistant.use_ordered_embeddings bool             = false
llama_model_loader: - kv  35:          gemma4_assistant.attention.k_eq_v bool             = true
llama_model_loader: - kv  36:      gemma4_assistant.requires_target_arch str              = gemma4
llama_model_loader: - kv  37:                       tokenizer.ggml.model str              = gemma4
llama_model_loader: - kv  38:                      tokenizer.ggml.tokens arr[str,262144]  = ["<pad>", "<eos>", "<bos>", "<unk>", ...
llama_model_loader: - kv  39:                      tokenizer.ggml.scores arr[f32,262144]  = [-1000.000000, -1000.000000, -1000.00...
llama_model_loader: - kv  40:                  tokenizer.ggml.token_type arr[i32,262144]  = [3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  41:                      tokenizer.ggml.merges arr[str,514906]  = ["\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n \n", ...
llama_model_loader: - kv  42:                tokenizer.ggml.bos_token_id u32              = 2
llama_model_loader: - kv  43:                tokenizer.ggml.eos_token_id u32              = 1
llama_model_loader: - kv  44:            tokenizer.ggml.unknown_token_id u32              = 3
llama_model_loader: - kv  45:            tokenizer.ggml.padding_token_id u32              = 0
llama_model_loader: - kv  46:               tokenizer.ggml.mask_token_id u32              = 4
llama_model_loader: - kv  47:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  48:               tokenizer.ggml.add_sep_token bool             = false
llama_model_loader: - kv  49:            tokenizer.ggml.add_space_prefix bool             = false
llama_model_loader: - kv  50:               general.quantization_version u32              = 2
llama_model_loader: - kv  51:                          general.file_type u32              = 15
llama_model_loader: - type  f32:   26 tensors
llama_model_loader: - type q4_K:   20 tensors
llama_model_loader: - type q6_K:    3 tensors
print_info: file format = GGUF V3 (latest)
print_info: file type   = Q4_K - Medium
print_info: file size   = 297.03 MiB (5.89 BPW)
init_tokenizer: initializing tokenizer for type 2
load: 0 unused tokens
load: control token:      0 '<pad>' is not marked as EOG
load: control token:      3 '<unk>' is not marked as EOG
load: control token:      2 '<bos>' is not marked as EOG
load: control token:      4 '<mask>' is not marked as EOG
load: control token:     47 '<tool|>' is not marked as EOG
load: control token:     98 '<|think|>' is not marked as EOG
load: control token:     46 '<|tool>' is not marked as EOG
load: control-looking token:     50 '<|tool_response>' was not control-type; this is probably a bug in the model. its type will be overridden
load: control token:    105 '<|turn>' is not marked as EOG
load: control-looking token:    212 '</s>' was not control-type; this is probably a bug in the model. its type will be overridden
load: control token: 258880 '<|image|>' is not marked as EOG
load: control token: 258882 '<image|>' is not marked as EOG
load: control token: 256000 '<|audio>' is not marked as EOG
load: control token: 258883 '<audio|>' is not marked as EOG
load: control token: 255999 '<|image>' is not marked as EOG
load: control token: 258881 '<|audio|>' is not marked as EOG
load: printing all EOG tokens:
load:   - 1 ('<eos>')
load:   - 50 ('<|tool_response>')
load:   - 106 ('<turn|>')
load:   - 212 ('</s>')
load: special_eog_ids contains '<|tool_response>', removing '</s>' token from EOG list
load: special tokens cache size = 23
load: token to piece cache size = 1.9445 MB
print_info: arch                  = gemma4_assistant
print_info: vocab_only            = 0
print_info: no_alloc              = 0
print_info: n_ctx_train           = 262144
print_info: n_embd                = 1024
print_info: n_embd_inp            = 1024
print_info: n_layer               = 4
print_info: n_head                = 16
print_info: n_head_kv             = [8, 8, 8, 1]
print_info: n_rot                 = 512
print_info: n_swa                 = 1024
print_info: is_swa_any            = 1
print_info: n_embd_head_k         = 512
print_info: n_embd_head_v         = 512
print_info: n_gqa                 = [2, 2, 2, 16]
print_info: n_embd_k_gqa          = [2048, 2048, 2048, 512]
print_info: n_embd_v_gqa          = [2048, 2048, 2048, 512]
print_info: f_norm_eps            = 0.0e+00
print_info: f_norm_rms_eps        = 1.0e-06
print_info: f_clamp_kqv           = 0.0e+00
print_info: f_max_alibi_bias      = 0.0e+00
print_info: f_logit_scale         = 0.0e+00
print_info: f_attn_scale          = 1.0e+00
print_info: f_attn_value_scale    = 0.0000
print_info: n_ff                  = 8192
print_info: n_expert              = 0
print_info: n_expert_used         = 0
print_info: n_expert_groups       = 0
print_info: n_group_used          = 0
print_info: causal attn           = 1
print_info: pooling type          = -1
print_info: rope type             = 2
print_info: rope scaling          = linear
print_info: freq_base_train       = 1000000.0
print_info: freq_scale_train      = 1
print_info: freq_base_swa         = 10000.0
print_info: freq_scale_swa        = 1
print_info: n_embd_head_k_swa     = 256
print_info: n_embd_head_v_swa     = 256
print_info: n_rot_swa             = 256
print_info: n_ctx_orig_yarn       = 262144
print_info: rope_yarn_log_mul     = 0.0000
print_info: rope_finetuned        = unknown
print_info: model type            = ?B
print_info: model params          = 422.86 M
print_info: general.name          = Google Gemma 4 12B It Assistant
print_info: vocab type            = BPE
print_info: n_vocab               = 262144
print_info: n_merges              = 514906
print_info: BOS token             = 2 '<bos>'
print_info: EOS token             = 1 '<eos>'
print_info: UNK token             = 3 '<unk>'
print_info: PAD token             = 0 '<pad>'
print_info: MASK token            = 4 '<mask>'
print_info: LF token              = 107 '
'
print_info: EOG token             = 1 '<eos>'
print_info: EOG token             = 50 '<|tool_response>'
print_info: EOG token             = 106 '<turn|>'
print_info: max token length      = 93
load_tensors: loading model tensors, this can take a while... (mmap = false, direct_io = true)
load_tensors: layer   0 assigned to device CPU, is_swa = 1
load_tensors: layer   1 assigned to device CPU, is_swa = 1
load_tensors: layer   2 assigned to device CPU, is_swa = 1
load_tensors: layer   3 assigned to device CPU, is_swa = 0
load_tensors: layer   4 assigned to device CPU, is_swa = 0
create_tensor: loading tensor token_embd.weight
create_tensor: loading tensor mtp.pre_projection.weight
create_tensor: loading tensor mtp.post_projection.weight
create_tensor: loading tensor output_norm.weight
create_tensor: loading tensor blk.0.attn_norm.weight
create_tensor: loading tensor blk.0.attn_q.weight
create_tensor: loading tensor blk.0.attn_output.weight
create_tensor: loading tensor blk.0.attn_q_norm.weight
create_tensor: loading tensor blk.0.post_attention_norm.weight
create_tensor: loading tensor blk.0.layer_output_scale.weight
create_tensor: loading tensor blk.0.ffn_norm.weight
create_tensor: loading tensor blk.0.ffn_gate.weight
create_tensor: loading tensor blk.0.ffn_up.weight
create_tensor: loading tensor blk.0.ffn_down.weight
create_tensor: loading tensor blk.0.post_ffw_norm.weight
create_tensor: loading tensor blk.1.attn_norm.weight
create_tensor: loading tensor blk.1.attn_q.weight
create_tensor: loading tensor blk.1.attn_output.weight
create_tensor: loading tensor blk.1.attn_q_norm.weight
create_tensor: loading tensor blk.1.post_attention_norm.weight
create_tensor: loading tensor blk.1.layer_output_scale.weight
create_tensor: loading tensor blk.1.ffn_norm.weight
create_tensor: loading tensor blk.1.ffn_gate.weight
create_tensor: loading tensor blk.1.ffn_up.weight
create_tensor: loading tensor blk.1.ffn_down.weight
create_tensor: loading tensor blk.1.post_ffw_norm.weight
create_tensor: loading tensor blk.2.attn_norm.weight
create_tensor: loading tensor blk.2.attn_q.weight
create_tensor: loading tensor blk.2.attn_output.weight
create_tensor: loading tensor blk.2.attn_q_norm.weight
create_tensor: loading tensor blk.2.post_attention_norm.weight
create_tensor: loading tensor blk.2.layer_output_scale.weight
create_tensor: loading tensor blk.2.ffn_norm.weight
create_tensor: loading tensor blk.2.ffn_gate.weight
create_tensor: loading tensor blk.2.ffn_up.weight
create_tensor: loading tensor blk.2.ffn_down.weight
create_tensor: loading tensor blk.2.post_ffw_norm.weight
create_tensor: loading tensor blk.3.attn_norm.weight
create_tensor: loading tensor blk.3.attn_q.weight
create_tensor: loading tensor blk.3.attn_output.weight
create_tensor: loading tensor blk.3.attn_q_norm.weight
create_tensor: loading tensor blk.3.post_attention_norm.weight
create_tensor: loading tensor blk.3.layer_output_scale.weight
create_tensor: loading tensor rope_freqs.weight
create_tensor: loading tensor blk.3.ffn_norm.weight
create_tensor: loading tensor blk.3.ffn_gate.weight
create_tensor: loading tensor blk.3.ffn_up.weight
create_tensor: loading tensor blk.3.ffn_down.weight
create_tensor: loading tensor blk.3.post_ffw_norm.weight
done_getting_tensors: tensor 'token_embd.weight' (q6_K) (and 48 others) cannot be used with preferred buffer type CPU_REPACK, using CPU instead
load_tensors:          CPU model buffer size =   297.03 MiB
load_all_data: no device found for buffer type CPU for async uploads
.......................
load_tensors: releasing mmaps (not needed after tensor copy)
[main]: loaded MTP assistant head 'c:\llama.cpp\models\Gemma-4\12B\gemma-4-12B-it-Q4_K_M-MTP-assistant.gguf' (n_embd_backbone=3840)
llama_context: constructing llama_context
llama_context: n_seq_max     = 1
llama_context: n_ctx         = 2048
llama_context: n_ctx_seq     = 2048
llama_context: n_batch       = 512
llama_context: n_ubatch      = 512
llama_context: causal_attn   = 1
llama_context: flash_attn    = auto
llama_context: kv_unified    = false
llama_context: freq_base     = 1000000.0
llama_context: freq_scale    = 1
llama_context: n_rs_seq      = 0
llama_context: n_ctx_seq (2048) < n_ctx_train (262144) -- the full capacity of the model will not be utilized
set_abort_callback: call
llama_context:        CPU  output buffer size =     1.00 MiB
llama_kv_cache_iswa: using full-size SWA cache (ref: https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055)
llama_kv_cache_iswa: creating non-SWA KV cache, size = 2048 cells
llama_kv_cache: layer   0: filtered
llama_kv_cache: layer   1: filtered
llama_kv_cache: layer   2: filtered
llama_kv_cache: layer   3: filtered
llama_kv_cache: layer   4: filtered
llama_kv_cache: layer   5: dev = CPU
llama_kv_cache: layer   6: filtered
llama_kv_cache: layer   7: filtered
llama_kv_cache: layer   8: filtered
llama_kv_cache: layer   9: filtered
llama_kv_cache: layer  10: filtered
llama_kv_cache: layer  11: dev = CPU
llama_kv_cache: layer  12: filtered
llama_kv_cache: layer  13: filtered
llama_kv_cache: layer  14: filtered
llama_kv_cache: layer  15: filtered
llama_kv_cache: layer  16: filtered
llama_kv_cache: layer  17: dev = CPU
llama_kv_cache: layer  18: filtered
llama_kv_cache: layer  19: filtered
llama_kv_cache: layer  20: filtered
llama_kv_cache: layer  21: filtered
llama_kv_cache: layer  22: filtered
llama_kv_cache: layer  23: dev = CPU
llama_kv_cache: layer  24: filtered
llama_kv_cache: layer  25: filtered
llama_kv_cache: layer  26: filtered
llama_kv_cache: layer  27: filtered
llama_kv_cache: layer  28: filtered
llama_kv_cache: layer  29: dev = CPU
llama_kv_cache: layer  30: filtered
llama_kv_cache: layer  31: filtered
llama_kv_cache: layer  32: filtered
llama_kv_cache: layer  33: filtered
llama_kv_cache: layer  34: filtered
llama_kv_cache: layer  35: dev = CPU
llama_kv_cache: layer  36: filtered
llama_kv_cache: layer  37: filtered
llama_kv_cache: layer  38: filtered
llama_kv_cache: layer  39: filtered
llama_kv_cache: layer  40: filtered
llama_kv_cache: layer  41: dev = CPU
llama_kv_cache: layer  42: filtered
llama_kv_cache: layer  43: filtered
llama_kv_cache: layer  44: filtered
llama_kv_cache: layer  45: filtered
llama_kv_cache: layer  46: filtered
llama_kv_cache: layer  47: dev = CPU
llama_kv_cache: reusing layers:
llama_kv_cache: - layer   0: no reuse
llama_kv_cache: - layer   1: no reuse
llama_kv_cache: - layer   2: no reuse
llama_kv_cache: - layer   3: no reuse
llama_kv_cache: - layer   4: no reuse
llama_kv_cache: - layer   5: no reuse
llama_kv_cache: - layer   6: no reuse
llama_kv_cache: - layer   7: no reuse
llama_kv_cache: - layer   8: no reuse
llama_kv_cache: - layer   9: no reuse
llama_kv_cache: - layer  10: no reuse
llama_kv_cache: - layer  11: no reuse
llama_kv_cache: - layer  12: no reuse
llama_kv_cache: - layer  13: no reuse
llama_kv_cache: - layer  14: no reuse
llama_kv_cache: - layer  15: no reuse
llama_kv_cache: - layer  16: no reuse
llama_kv_cache: - layer  17: no reuse
llama_kv_cache: - layer  18: no reuse
llama_kv_cache: - layer  19: no reuse
llama_kv_cache: - layer  20: no reuse
llama_kv_cache: - layer  21: no reuse
llama_kv_cache: - layer  22: no reuse
llama_kv_cache: - layer  23: no reuse
llama_kv_cache: - layer  24: no reuse
llama_kv_cache: - layer  25: no reuse
llama_kv_cache: - layer  26: no reuse
llama_kv_cache: - layer  27: no reuse
llama_kv_cache: - layer  28: no reuse
llama_kv_cache: - layer  29: no reuse
llama_kv_cache: - layer  30: no reuse
llama_kv_cache: - layer  31: no reuse
llama_kv_cache: - layer  32: no reuse
llama_kv_cache: - layer  33: no reuse
llama_kv_cache: - layer  34: no reuse
llama_kv_cache: - layer  35: no reuse
llama_kv_cache: - layer  36: no reuse
llama_kv_cache: - layer  37: no reuse
llama_kv_cache: - layer  38: no reuse
llama_kv_cache: - layer  39: no reuse
llama_kv_cache: - layer  40: no reuse
llama_kv_cache: - layer  41: no reuse
llama_kv_cache: - layer  42: no reuse
llama_kv_cache: - layer  43: no reuse
llama_kv_cache: - layer  44: no reuse
llama_kv_cache: - layer  45: no reuse
llama_kv_cache: - layer  46: no reuse
llama_kv_cache: - layer  47: no reuse
llama_kv_cache:        CPU KV buffer size =    32.00 MiB
llama_kv_cache: size =   32.00 MiB (  2048 cells,   8 layers,  1/1 seqs), K (f16):   16.00 MiB, V (f16):   16.00 MiB
llama_kv_cache: attn_rot_k = 0, n_embd_head_k_all = 512
llama_kv_cache: attn_rot_v = 0, n_embd_head_k_all = 512
llama_kv_cache_iswa: creating     SWA KV cache, size = 2048 cells
llama_kv_cache: layer   0: dev = CPU
llama_kv_cache: layer   1: dev = CPU
llama_kv_cache: layer   2: dev = CPU
llama_kv_cache: layer   3: dev = CPU
llama_kv_cache: layer   4: dev = CPU
llama_kv_cache: layer   5: filtered
llama_kv_cache: layer   6: dev = CPU
llama_kv_cache: layer   7: dev = CPU
llama_kv_cache: layer   8: dev = CPU
llama_kv_cache: layer   9: dev = CPU
llama_kv_cache: layer  10: dev = CPU
llama_kv_cache: layer  11: filtered
llama_kv_cache: layer  12: dev = CPU
llama_kv_cache: layer  13: dev = CPU
llama_kv_cache: layer  14: dev = CPU
llama_kv_cache: layer  15: dev = CPU
llama_kv_cache: layer  16: dev = CPU
llama_kv_cache: layer  17: filtered
llama_kv_cache: layer  18: dev = CPU
llama_kv_cache: layer  19: dev = CPU
llama_kv_cache: layer  20: dev = CPU
llama_kv_cache: layer  21: dev = CPU
llama_kv_cache: layer  22: dev = CPU
llama_kv_cache: layer  23: filtered
llama_kv_cache: layer  24: dev = CPU
llama_kv_cache: layer  25: dev = CPU
llama_kv_cache: layer  26: dev = CPU
llama_kv_cache: layer  27: dev = CPU
llama_kv_cache: layer  28: dev = CPU
llama_kv_cache: layer  29: filtered
llama_kv_cache: layer  30: dev = CPU
llama_kv_cache: layer  31: dev = CPU
llama_kv_cache: layer  32: dev = CPU
llama_kv_cache: layer  33: dev = CPU
llama_kv_cache: layer  34: dev = CPU
llama_kv_cache: layer  35: filtered
llama_kv_cache: layer  36: dev = CPU
llama_kv_cache: layer  37: dev = CPU
llama_kv_cache: layer  38: dev = CPU
llama_kv_cache: layer  39: dev = CPU
llama_kv_cache: layer  40: dev = CPU
llama_kv_cache: layer  41: filtered
llama_kv_cache: layer  42: dev = CPU
llama_kv_cache: layer  43: dev = CPU
llama_kv_cache: layer  44: dev = CPU
llama_kv_cache: layer  45: dev = CPU
llama_kv_cache: layer  46: dev = CPU
llama_kv_cache: layer  47: filtered
llama_kv_cache: reusing layers:
llama_kv_cache: - layer   0: no reuse
llama_kv_cache: - layer   1: no reuse
llama_kv_cache: - layer   2: no reuse
llama_kv_cache: - layer   3: no reuse
llama_kv_cache: - layer   4: no reuse
llama_kv_cache: - layer   5: no reuse
llama_kv_cache: - layer   6: no reuse
llama_kv_cache: - layer   7: no reuse
llama_kv_cache: - layer   8: no reuse
llama_kv_cache: - layer   9: no reuse
llama_kv_cache: - layer  10: no reuse
llama_kv_cache: - layer  11: no reuse
llama_kv_cache: - layer  12: no reuse
llama_kv_cache: - layer  13: no reuse
llama_kv_cache: - layer  14: no reuse
llama_kv_cache: - layer  15: no reuse
llama_kv_cache: - layer  16: no reuse
llama_kv_cache: - layer  17: no reuse
llama_kv_cache: - layer  18: no reuse
llama_kv_cache: - layer  19: no reuse
llama_kv_cache: - layer  20: no reuse
llama_kv_cache: - layer  21: no reuse
llama_kv_cache: - layer  22: no reuse
llama_kv_cache: - layer  23: no reuse
llama_kv_cache: - layer  24: no reuse
llama_kv_cache: - layer  25: no reuse
llama_kv_cache: - layer  26: no reuse
llama_kv_cache: - layer  27: no reuse
llama_kv_cache: - layer  28: no reuse
llama_kv_cache: - layer  29: no reuse
llama_kv_cache: - layer  30: no reuse
llama_kv_cache: - layer  31: no reuse
llama_kv_cache: - layer  32: no reuse
llama_kv_cache: - layer  33: no reuse
llama_kv_cache: - layer  34: no reuse
llama_kv_cache: - layer  35: no reuse
llama_kv_cache: - layer  36: no reuse
llama_kv_cache: - layer  37: no reuse
llama_kv_cache: - layer  38: no reuse
llama_kv_cache: - layer  39: no reuse
llama_kv_cache: - layer  40: no reuse
llama_kv_cache: - layer  41: no reuse
llama_kv_cache: - layer  42: no reuse
llama_kv_cache: - layer  43: no reuse
llama_kv_cache: - layer  44: no reuse
llama_kv_cache: - layer  45: no reuse
llama_kv_cache: - layer  46: no reuse
llama_kv_cache: - layer  47: no reuse
llama_kv_cache:        CPU KV buffer size =   640.00 MiB
llama_kv_cache: size =  640.00 MiB (  2048 cells,  40 layers,  1/1 seqs), K (f16):  320.00 MiB, V (f16):  320.00 MiB
llama_kv_cache: attn_rot_k = 0, n_embd_head_k_all = 256
llama_kv_cache: attn_rot_v = 0, n_embd_head_k_all = 256
llama_context: enumerating backends
llama_context: backend_ptrs.size() = 1
sched_reserve: reserving ...
sched_reserve: max_nodes = 5344
sched_reserve: reserving full memory module
sched_reserve: worst-case: n_tokens = 512, n_seqs = 1, n_outputs = 1
graph_reserve: reserving a graph for ubatch with n_tokens =    1, n_seqs =  1, n_outputs =    1
sched_reserve: Flash Attention was auto, set to enabled
sched_reserve: resolving fused Gated Delta Net support:
graph_reserve: reserving a graph for ubatch with n_tokens =    1, n_seqs =  1, n_outputs =    1
sched_reserve: fused Gated Delta Net (autoregressive) enabled
graph_reserve: reserving a graph for ubatch with n_tokens =   16, n_seqs =  1, n_outputs =   16
sched_reserve: fused Gated Delta Net (chunked) enabled
graph_reserve: reserving a graph for ubatch with n_tokens =  512, n_seqs =  1, n_outputs =  512
graph_reserve: reserving a graph for ubatch with n_tokens =    1, n_seqs =  1, n_outputs =    1
graph_reserve: reserving a graph for ubatch with n_tokens =  512, n_seqs =  1, n_outputs =  512
sched_reserve:        CPU compute buffer size =   527.00 MiB
sched_reserve: graph nodes  = 1972
sched_reserve: graph splits = 1
sched_reserve: reserve took 3.90 ms, sched copies = 1
set_embeddings: value = 1
[main]: MTP speculative decoding active, draft_block_size=3 (drafts 2 tokens/round)
common_speculative_impl_draft_gemma4_mtp: adding speculative implementation 'draft-mtp' (gemma4 assistant)

[main]: n_ctx = 2048, n_threads = 8, gpu_layers = 999
[main]: flash-attn = auto (runtime-selected)
common_speculative_impl_draft_gemma4_mtp: - draft_block_size=3, n_max=2, n_min=0, n_steps=2, n_embd_bb=3840
[main]: system_info: CPU : SSE3 = 1 | SSSE3 = 1 | AVX = 1 | AVX2 = 1 | F16C = 1 | FMA = 1 | AVX512 = 1 | LLAMAFILE = 1 | OPENMP = 1 | REPACK = 1 |

> Running with custom prompt => [1/18]: [This is Izzy's bedroom]
set_embeddings: value = 1
  prefill: 19264.2ms (555 tokens, 28.8 t/s)
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
 {
    "answer": "c. That is correct Mara. It is your sister's_room",
    "justification": "Mara's answer 'This is Izzy's bedroom' matches the correct answer (c) 'That is correct Mara. It is your sister's_room'. Option (c) is the correct answer. Option (c) is the correct answer."
    "justification": "Mara's answer 'This is Izzy's__room' matches the correct answer (c) 'That is correct Mara. It is your sister's_room'. Option (
  [all token ids (129): 642 107 140 236775 14433 1083 623 236755 236761 2981 563 4338 63338 236761 1030 563 822 12198 236789 236751 236779 3352 827 107 140 236775 4017 2540 1083 623 201883 236789 236751 3890 756 2094 563 29689 2946 236789 236751 15238 236789 12853 506 4338 3890 568 236755 236768 756 6372 563 4338 63338 236761 1030 563 822 12198 236789 236751 236779 3352 6748 19993 568 236755 236768 563 506 4338 3890 236761 19993 568 236755 236768 563 506 4338 3890 1781 107 140 236775 4017 2540 1083 623 201883 236789 236751 3890 756 2094 563 29689 2946 236789 236751 236779 236779 3352 236789 12853 506 4338 3890 568 236755 236768 756 6372 563 4338 63338 236761 1030 563 822 12198 236789 236751 236779 3352 6748 19993 568]
  [output hex (429 bytes):]
    0000: 20 7b 0a 20 20 20 20 22 61 6e 73 77 65 72 22 3a  | {.    "answer":|
    0010: 20 22 63 2e 20 54 68 61 74 20 69 73 20 63 6f 72  | "c. That is cor|
    0020: 72 65 63 74 20 4d 61 72 61 2e 20 49 74 20 69 73  |rect Mara. It is|
    0030: 20 79 6f 75 72 20 73 69 73 74 65 72 27 73 5f 72  | your sister's_r|
    0040: 6f 6f 6d 22 2c 0a 20 20 20 20 22 6a 75 73 74 69  |oom",.    "justi|
    0050: 66 69 63 61 74 69 6f 6e 22 3a 20 22 4d 61 72 61  |fication": "Mara|
    0060: 27 73 20 61 6e 73 77 65 72 20 27 54 68 69 73 20  |'s answer 'This |
    0070: 69 73 20 49 7a 7a 79 27 73 20 62 65 64 72 6f 6f  |is Izzy's bedroo|
    0080: 6d 27 20 6d 61 74 63 68 65 73 20 74 68 65 20 63  |m' matches the c|
    0090: 6f 72 72 65 63 74 20 61 6e 73 77 65 72 20 28 63  |orrect answer (c|
    00a0: 29 20 27 54 68 61 74 20 69 73 20 63 6f 72 72 65  |) 'That is corre|
    00b0: 63 74 20 4d 61 72 61 2e 20 49 74 20 69 73 20 79  |ct Mara. It is y|
    00c0: 6f 75 72 20 73 69 73 74 65 72 27 73 5f 72 6f 6f  |our sister's_roo|
    00d0: 6d 27 2e 20 4f 70 74 69 6f 6e 20 28 63 29 20 69  |m'. Option (c) i|
    00e0: 73 20 74 68 65 20 63 6f 72 72 65 63 74 20 61 6e  |s the correct an|
    00f0: 73 77 65 72 2e 20 4f 70 74 69 6f 6e 20 28 63 29  |swer. Option (c)|
    0100: 20 69 73 20 74 68 65 20 63 6f 72 72 65 63 74 20  | is the correct |
    0110: 61 6e 73 77 65 72 2e 22 0a 20 20 20 20 22 6a 75  |answer.".    "ju|
    0120: 73 74 69 66 69 63 61 74 69 6f 6e 22 3a 20 22 4d  |stification": "M|
    0130: 61 72 61 27 73 20 61 6e 73 77 65 72 20 27 54 68  |ara's answer 'Th|
    0140: 69 73 20 69 73 20 49 7a 7a 79 27 73 5f 5f 72 6f  |is is Izzy's__ro|
    0150: 6f 6d 27 20 6d 61 74 63 68 65 73 20 74 68 65 20  |om' matches the |
    0160: 63 6f 72 72 65 63 74 20 61 6e 73 77 65 72 20 28  |correct answer (|
    0170: 63 29 20 27 54 68 61 74 20 69 73 20 63 6f 72 72  |c) 'That is corr|
    0180: 65 63 74 20 4d 61 72 61 2e 20 49 74 20 69 73 20  |ect Mara. It is |
    0190: 79 6f 75 72 20 73 69 73 74 65 72 27 73 5f 72 6f  |your sister's_ro|
    01a0: 6f 6d 27 2e 20 4f 70 74 69 6f 6e 20 28           |om'. Option (|
  decode: 12037.8ms (129 tokens, 10.72 t/s)
  decode (no-TTFT): 12034.0ms (129 tokens, 10.72 t/s) [TTFT 3.8ms]

> Running with custom prompt => [2/18]: [This is bedroom for Izzy]
set_embeddings: value = 1
  prefill: 19727.3ms (554 tokens, 28.1 t/s)
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
 {
    "answer": "c. That is correct Mara. It is your sister's. \\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\end
<|channel>
<|channel>
<|channel>
<|channel>
<|channel>
<|channel>
<|channel>
<|channel>
<|channel>
<|channel>
<|channel>
<|channel>
<|channel>
<|channel>
<|channel>
<|channel>
<|channel>
<|channel>
<|channel>
<|channel>
  [all token ids (129): 642 107 140 236775 14433 1083 623 236755 236761 2981 563 4338 63338 236761 1030 563 822 12198 236789 236751 236761 1704 236749 236785 236749 236785 236749 236785 236749 236785 236749 236785 236749 236785 236749 236785 236749 236785 236749 236785 236749 236785 236749 236785 236749 236785 236749 236785 236749 236785 236749 236785 236749 236785 236749 236785 236749 236785 236749 236785 236749 236785 236749 236785 236749 236785 236749 236785 236749 236785 236749 236785 236749 236785 236749 236785 236749 236785 236749 236785 236749 236785 236749 236785 236749 236785 236749 236785 643 107 100 107 100 107 100 107 100 107 100 107 100 107 100 107 100 107 100 107 100 107 100 107 100 107 100 107 100 107 100 107 100 107 100 107 100 107 100 107 100]
  [output hex (355 bytes):]
    0000: 20 7b 0a 20 20 20 20 22 61 6e 73 77 65 72 22 3a  | {.    "answer":|
    0010: 20 22 63 2e 20 54 68 61 74 20 69 73 20 63 6f 72  | "c. That is cor|
    0020: 72 65 63 74 20 4d 61 72 61 2e 20 49 74 20 69 73  |rect Mara. It is|
    0030: 20 79 6f 75 72 20 73 69 73 74 65 72 27 73 2e 20  | your sister's. |
    0040: 5c 5c 6e 5c 6e 5c 6e 5c 6e 5c 6e 5c 6e 5c 6e 5c  |\\n\n\n\n\n\n\n\|
    0050: 6e 5c 6e 5c 6e 5c 6e 5c 6e 5c 6e 5c 6e 5c 6e 5c  |n\n\n\n\n\n\n\n\|
    0060: 6e 5c 6e 5c 6e 5c 6e 5c 6e 5c 6e 5c 6e 5c 6e 5c  |n\n\n\n\n\n\n\n\|
    0070: 6e 5c 6e 5c 6e 5c 6e 5c 6e 5c 6e 5c 6e 5c 6e 5c  |n\n\n\n\n\n\n\n\|
    0080: 6e 5c 6e 5c 65 6e 64 0a 3c 7c 63 68 61 6e 6e 65  |n\n\end.<|channe|
    0090: 6c 3e 0a 3c 7c 63 68 61 6e 6e 65 6c 3e 0a 3c 7c  |l>.<|channel>.<||
    00a0: 63 68 61 6e 6e 65 6c 3e 0a 3c 7c 63 68 61 6e 6e  |channel>.<|chann|
    00b0: 65 6c 3e 0a 3c 7c 63 68 61 6e 6e 65 6c 3e 0a 3c  |el>.<|channel>.<|
    00c0: 7c 63 68 61 6e 6e 65 6c 3e 0a 3c 7c 63 68 61 6e  ||channel>.<|chan|
    00d0: 6e 65 6c 3e 0a 3c 7c 63 68 61 6e 6e 65 6c 3e 0a  |nel>.<|channel>.|
    00e0: 3c 7c 63 68 61 6e 6e 65 6c 3e 0a 3c 7c 63 68 61  |<|channel>.<|cha|
    00f0: 6e 6e 65 6c 3e 0a 3c 7c 63 68 61 6e 6e 65 6c 3e  |nnel>.<|channel>|
    0100: 0a 3c 7c 63 68 61 6e 6e 65 6c 3e 0a 3c 7c 63 68  |.<|channel>.<|ch|
    0110: 61 6e 6e 65 6c 3e 0a 3c 7c 63 68 61 6e 6e 65 6c  |annel>.<|channel|
    0120: 3e 0a 3c 7c 63 68 61 6e 6e 65 6c 3e 0a 3c 7c 63  |>.<|channel>.<|c|
    0130: 68 61 6e 6e 65 6c 3e 0a 3c 7c 63 68 61 6e 6e 65  |hannel>.<|channe|
    0140: 6c 3e 0a 3c 7c 63 68 61 6e 6e 65 6c 3e 0a 3c 7c  |l>.<|channel>.<||
    0150: 63 68 61 6e 6e 65 6c 3e 0a 3c 7c 63 68 61 6e 6e  |channel>.<|chann|
    0160: 65 6c 3e                                         |el>|
  decode: 11815.1ms (129 tokens, 10.92 t/s)
  decode (no-TTFT): 11814.6ms (129 tokens, 10.92 t/s) [TTFT 0.5ms]

> Running with custom prompt => [3/18]: [This is the bedroom for my sister Izzy]
set_embeddings: value = 1
^C
C:\llama.cpp\b612_052026_mtp\build>bin\RelWithDebInfo\minslm-cli.exe C:\llama.cpp\models\Gemma-4\12B\Unsloth\gemma-4-12b-it-Q4_K_M_Unsloth.gguf 8 ..\examples\llm-infer\prompts\custom_prompts_gemma4.txt --mtp-head c:\llama.cpp\models\Gemma-4\12B\Unsloth\gemma-4-12b-it-Q4_K_M-assistant_Unsloth.gguf  --spec-type mtp --draft-block-size 3 v2
Warning: unknown argument '--spec-type' ignored
Warning: unknown argument 'mtp' ignored
[main]: loaded 18 prompt(s) from '..\examples\llm-infer\prompts\custom_prompts_gemma4.txt'
llama_model_loader: direct I/O is enabled, disabling mmap
llama_model_loader: loaded meta data with 58 key-value pairs and 667 tensors from C:\llama.cpp\models\Gemma-4\12B\Unsloth\gemma-4-12b-it-Q4_K_M_Unsloth.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = gemma4
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                     general.sampling.top_k i32              = 64
llama_model_loader: - kv   3:                     general.sampling.top_p f32              = 0.950000
llama_model_loader: - kv   4:                      general.sampling.temp f32              = 1.000000
llama_model_loader: - kv   5:                               general.name str              = Gemma-4-12B-It
llama_model_loader: - kv   6:                           general.finetune str              = it
llama_model_loader: - kv   7:                           general.basename str              = Gemma-4-12B-It
llama_model_loader: - kv   8:                       general.quantized_by str              = Unsloth
llama_model_loader: - kv   9:                         general.size_label str              = 12B
llama_model_loader: - kv  10:                            general.license str              = apache-2.0
llama_model_loader: - kv  11:                       general.license.link str              = https://ai.google.dev/gemma/docs/gemm...
llama_model_loader: - kv  12:                           general.repo_url str              = https://huggingface.co/unsloth
llama_model_loader: - kv  13:                   general.base_model.count u32              = 1
llama_model_loader: - kv  14:                  general.base_model.0.name str              = Gemma 4 12b It
llama_model_loader: - kv  15:          general.base_model.0.organization str              = Google
llama_model_loader: - kv  16:              general.base_model.0.repo_url str              = https://huggingface.co/google/gemma-4...
llama_model_loader: - kv  17:                               general.tags arr[str,2]       = ["unsloth", "any-to-any"]
llama_model_loader: - kv  18:                         gemma4.block_count u32              = 48
llama_model_loader: - kv  19:                      gemma4.context_length u32              = 262144
llama_model_loader: - kv  20:                    gemma4.embedding_length u32              = 3840
llama_model_loader: - kv  21:                 gemma4.feed_forward_length u32              = 15360
llama_model_loader: - kv  22:                gemma4.attention.head_count u32              = 16
llama_model_loader: - kv  23:             gemma4.attention.head_count_kv arr[i32,48]      = [8, 8, 8, 8, 8, 1, 8, 8, 8, 8, 8, 1, ...
llama_model_loader: - kv  24:                      gemma4.rope.freq_base f32              = 1000000.000000
llama_model_loader: - kv  25:                  gemma4.rope.freq_base_swa f32              = 10000.000000
llama_model_loader: - kv  26:    gemma4.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  27:                gemma4.attention.key_length u32              = 512
llama_model_loader: - kv  28:              gemma4.attention.value_length u32              = 512
llama_model_loader: - kv  29:             gemma4.final_logit_softcapping f32              = 30.000000
llama_model_loader: - kv  30:            gemma4.attention.sliding_window u32              = 1024
llama_model_loader: - kv  31:          gemma4.attention.shared_kv_layers u32              = 0
llama_model_loader: - kv  32:    gemma4.embedding_length_per_layer_input u32              = 0
llama_model_loader: - kv  33:    gemma4.attention.sliding_window_pattern arr[bool,48]     = [true, true, true, true, true, false,...
llama_model_loader: - kv  34:            gemma4.attention.key_length_swa u32              = 256
llama_model_loader: - kv  35:          gemma4.attention.value_length_swa u32              = 256
llama_model_loader: - kv  36:                gemma4.rope.dimension_count u32              = 512
llama_model_loader: - kv  37:            gemma4.rope.dimension_count_swa u32              = 256
llama_model_loader: - kv  38:             tokenizer.ggml.suppress_tokens arr[i32,2]       = [258883, 258882]
llama_model_loader: - kv  39:                       tokenizer.ggml.model str              = gemma4
llama_model_loader: - kv  40:                      tokenizer.ggml.tokens arr[str,262144]  = ["<pad>", "<eos>", "<bos>", "<unk>", ...
llama_model_loader: - kv  41:                      tokenizer.ggml.scores arr[f32,262144]  = [-1000.000000, -1000.000000, -1000.00...
llama_model_loader: - kv  42:                  tokenizer.ggml.token_type arr[i32,262144]  = [3, 1, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  43:                      tokenizer.ggml.merges arr[str,514906]  = ["\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n \n", ...
llama_model_loader: - kv  44:                tokenizer.ggml.bos_token_id u32              = 2
llama_model_loader: - kv  45:                tokenizer.ggml.eos_token_id u32              = 106
llama_model_loader: - kv  46:            tokenizer.ggml.unknown_token_id u32              = 3
llama_model_loader: - kv  47:            tokenizer.ggml.padding_token_id u32              = 0
llama_model_loader: - kv  48:               tokenizer.ggml.mask_token_id u32              = 4
llama_model_loader: - kv  49:                    tokenizer.chat_template str              = {%- macro format_parameters(propertie...
llama_model_loader: - kv  50:            tokenizer.ggml.add_space_prefix bool             = false
llama_model_loader: - kv  51:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  52:               general.quantization_version u32              = 2
llama_model_loader: - kv  53:                          general.file_type u32              = 15
llama_model_loader: - kv  54:                      quantize.imatrix.file str              = gemma-4-12b-it-GGUF/imatrix_unsloth.gguf
llama_model_loader: - kv  55:                   quantize.imatrix.dataset str              = unsloth_calibration_gemma-4-12b-it.txt
llama_model_loader: - kv  56:             quantize.imatrix.entries_count u32              = 328
llama_model_loader: - kv  57:              quantize.imatrix.chunks_count u32              = 141
llama_model_loader: - type  f32:  338 tensors
llama_model_loader: - type q4_K:  285 tensors
llama_model_loader: - type q6_K:   44 tensors
print_info: file format = GGUF V3 (latest)
print_info: file type   = Q4_K - Medium
print_info: file size   = 6.62 GiB (4.77 BPW)
init_tokenizer: initializing tokenizer for type 2
load: 0 unused tokens
load: control-looking token:      1 '<eos>' was not control-type; this is probably a bug in the model. its type will be overridden
load: control token:      0 '<pad>' is not marked as EOG
load: control token:      3 '<unk>' is not marked as EOG
load: control token:      2 '<bos>' is not marked as EOG
load: control token:      4 '<mask>' is not marked as EOG
load: control token:     47 '<tool|>' is not marked as EOG
load: control token:     98 '<|think|>' is not marked as EOG
load: control token:     46 '<|tool>' is not marked as EOG
load: control-looking token:     50 '<|tool_response>' was not control-type; this is probably a bug in the model. its type will be overridden
load: control token:    105 '<|turn>' is not marked as EOG
load: control-looking token:    212 '</s>' was not control-type; this is probably a bug in the model. its type will be overridden
load: control token: 258880 '<|image|>' is not marked as EOG
load: control token: 258882 '<image|>' is not marked as EOG
load: control token: 256000 '<|audio>' is not marked as EOG
load: control token: 258883 '<audio|>' is not marked as EOG
load: control token: 258884 '<|video|>' is not marked as EOG
load: control token: 255999 '<|image>' is not marked as EOG
load: control token: 258881 '<|audio|>' is not marked as EOG
load: printing all EOG tokens:
load:   - 1 ('<eos>')
load:   - 50 ('<|tool_response>')
load:   - 106 ('<turn|>')
load:   - 212 ('</s>')
load: special_eog_ids contains '<|tool_response>', removing '</s>' token from EOG list
load: special tokens cache size = 24
load: token to piece cache size = 1.9445 MB
print_info: arch                  = gemma4
print_info: vocab_only            = 0
print_info: no_alloc              = 0
print_info: n_ctx_train           = 262144
print_info: n_embd                = 3840
print_info: n_embd_inp            = 3840
print_info: n_layer               = 48
print_info: n_head                = 16
print_info: n_head_kv             = [8, 8, 8, 8, 8, 1, 8, 8, 8, 8, 8, 1, 8, 8, 8, 8, 8, 1, 8, 8, 8, 8, 8, 1, 8, 8, 8, 8, 8, 1, 8, 8, 8, 8, 8, 1, 8, 8, 8, 8, 8, 1, 8, 8, 8, 8, 8, 1]
print_info: n_rot                 = 512
print_info: n_swa                 = 1024
print_info: is_swa_any            = 1
print_info: n_embd_head_k         = 512
print_info: n_embd_head_v         = 512
print_info: n_gqa                 = [2, 2, 2, 2, 2, 16, 2, 2, 2, 2, 2, 16, 2, 2, 2, 2, 2, 16, 2, 2, 2, 2, 2, 16, 2, 2, 2, 2, 2, 16, 2, 2, 2, 2, 2, 16, 2, 2, 2, 2, 2, 16, 2, 2, 2, 2, 2, 16]
print_info: n_embd_k_gqa          = [2048, 2048, 2048, 2048, 2048, 512, 2048, 2048, 2048, 2048, 2048, 512, 2048, 2048, 2048, 2048, 2048, 512, 2048, 2048, 2048, 2048, 2048, 512, 2048, 2048, 2048, 2048, 2048, 512, 2048, 2048, 2048, 2048, 2048, 512, 2048, 2048, 2048, 2048, 2048, 512, 2048, 2048, 2048, 2048, 2048, 512]
print_info: n_embd_v_gqa          = [2048, 2048, 2048, 2048, 2048, 512, 2048, 2048, 2048, 2048, 2048, 512, 2048, 2048, 2048, 2048, 2048, 512, 2048, 2048, 2048, 2048, 2048, 512, 2048, 2048, 2048, 2048, 2048, 512, 2048, 2048, 2048, 2048, 2048, 512, 2048, 2048, 2048, 2048, 2048, 512, 2048, 2048, 2048, 2048, 2048, 512]
print_info: f_norm_eps            = 0.0e+00
print_info: f_norm_rms_eps        = 1.0e-06
print_info: f_clamp_kqv           = 0.0e+00
print_info: f_max_alibi_bias      = 0.0e+00
print_info: f_logit_scale         = 0.0e+00
print_info: f_attn_scale          = 1.0e+00
print_info: f_attn_value_scale    = 0.0000
print_info: n_ff                  = 15360
print_info: n_expert              = 0
print_info: n_expert_used         = 0
print_info: n_expert_groups       = 0
print_info: n_group_used          = 0
print_info: causal attn           = 1
print_info: pooling type          = -1
print_info: rope type             = 2
print_info: rope scaling          = linear
print_info: freq_base_train       = 1000000.0
print_info: freq_scale_train      = 1
print_info: freq_base_swa         = 10000.0
print_info: freq_scale_swa        = 1
print_info: n_embd_head_k_swa     = 256
print_info: n_embd_head_v_swa     = 256
print_info: n_rot_swa             = 256
print_info: n_ctx_orig_yarn       = 262144
print_info: rope_yarn_log_mul     = 0.0000
print_info: rope_finetuned        = unknown
print_info: model type            = ?B
print_info: model params          = 11.91 B
print_info: general.name          = Gemma-4-12B-It
print_info: vocab type            = BPE
print_info: n_vocab               = 262144
print_info: n_merges              = 514906
print_info: BOS token             = 2 '<bos>'
print_info: EOS token             = 106 '<turn|>'
print_info: UNK token             = 3 '<unk>'
print_info: PAD token             = 0 '<pad>'
print_info: MASK token            = 4 '<mask>'
print_info: LF token              = 107 '
'
print_info: EOG token             = 1 '<eos>'
print_info: EOG token             = 50 '<|tool_response>'
print_info: EOG token             = 106 '<turn|>'
print_info: max token length      = 93
load_tensors: loading model tensors, this can take a while... (mmap = false, direct_io = true)
load_tensors: layer   0 assigned to device CPU, is_swa = 1
load_tensors: layer   1 assigned to device CPU, is_swa = 1
load_tensors: layer   2 assigned to device CPU, is_swa = 1
load_tensors: layer   3 assigned to device CPU, is_swa = 1
load_tensors: layer   4 assigned to device CPU, is_swa = 1
load_tensors: layer   5 assigned to device CPU, is_swa = 0
load_tensors: layer   6 assigned to device CPU, is_swa = 1
load_tensors: layer   7 assigned to device CPU, is_swa = 1
load_tensors: layer   8 assigned to device CPU, is_swa = 1
load_tensors: layer   9 assigned to device CPU, is_swa = 1
load_tensors: layer  10 assigned to device CPU, is_swa = 1
load_tensors: layer  11 assigned to device CPU, is_swa = 0
load_tensors: layer  12 assigned to device CPU, is_swa = 1
load_tensors: layer  13 assigned to device CPU, is_swa = 1
load_tensors: layer  14 assigned to device CPU, is_swa = 1
load_tensors: layer  15 assigned to device CPU, is_swa = 1
load_tensors: layer  16 assigned to device CPU, is_swa = 1
load_tensors: layer  17 assigned to device CPU, is_swa = 0
load_tensors: layer  18 assigned to device CPU, is_swa = 1
load_tensors: layer  19 assigned to device CPU, is_swa = 1
load_tensors: layer  20 assigned to device CPU, is_swa = 1
load_tensors: layer  21 assigned to device CPU, is_swa = 1
load_tensors: layer  22 assigned to device CPU, is_swa = 1
load_tensors: layer  23 assigned to device CPU, is_swa = 0
load_tensors: layer  24 assigned to device CPU, is_swa = 1
load_tensors: layer  25 assigned to device CPU, is_swa = 1
load_tensors: layer  26 assigned to device CPU, is_swa = 1
load_tensors: layer  27 assigned to device CPU, is_swa = 1
load_tensors: layer  28 assigned to device CPU, is_swa = 1
load_tensors: layer  29 assigned to device CPU, is_swa = 0
load_tensors: layer  30 assigned to device CPU, is_swa = 1
load_tensors: layer  31 assigned to device CPU, is_swa = 1
load_tensors: layer  32 assigned to device CPU, is_swa = 1
load_tensors: layer  33 assigned to device CPU, is_swa = 1
load_tensors: layer  34 assigned to device CPU, is_swa = 1
load_tensors: layer  35 assigned to device CPU, is_swa = 0
load_tensors: layer  36 assigned to device CPU, is_swa = 1
load_tensors: layer  37 assigned to device CPU, is_swa = 1
load_tensors: layer  38 assigned to device CPU, is_swa = 1
load_tensors: layer  39 assigned to device CPU, is_swa = 1
load_tensors: layer  40 assigned to device CPU, is_swa = 1
load_tensors: layer  41 assigned to device CPU, is_swa = 0
load_tensors: layer  42 assigned to device CPU, is_swa = 1
load_tensors: layer  43 assigned to device CPU, is_swa = 1
load_tensors: layer  44 assigned to device CPU, is_swa = 1
load_tensors: layer  45 assigned to device CPU, is_swa = 1
load_tensors: layer  46 assigned to device CPU, is_swa = 1
load_tensors: layer  47 assigned to device CPU, is_swa = 0
load_tensors: layer  48 assigned to device CPU, is_swa = 0
create_tensor: loading tensor token_embd.weight
create_tensor: loading tensor token_embd.weight
create_tensor: loading tensor output_norm.weight
create_tensor: loading tensor blk.0.attn_norm.weight
create_tensor: loading tensor blk.0.attn_q.weight
create_tensor: loading tensor blk.0.attn_k.weight
create_tensor: loading tensor blk.0.attn_v.weight
create_tensor: loading tensor blk.0.attn_output.weight
create_tensor: loading tensor blk.0.attn_q_norm.weight
create_tensor: loading tensor blk.0.attn_k_norm.weight
create_tensor: loading tensor blk.0.post_attention_norm.weight
create_tensor: loading tensor blk.0.layer_output_scale.weight
create_tensor: loading tensor blk.0.ffn_norm.weight
create_tensor: loading tensor blk.0.ffn_gate.weight
create_tensor: loading tensor blk.0.ffn_up.weight
create_tensor: loading tensor blk.0.ffn_down.weight
create_tensor: loading tensor blk.0.post_ffw_norm.weight
create_tensor: loading tensor blk.1.attn_norm.weight
create_tensor: loading tensor blk.1.attn_q.weight
create_tensor: loading tensor blk.1.attn_k.weight
create_tensor: loading tensor blk.1.attn_v.weight
create_tensor: loading tensor blk.1.attn_output.weight
create_tensor: loading tensor blk.1.attn_q_norm.weight
create_tensor: loading tensor blk.1.attn_k_norm.weight
create_tensor: loading tensor blk.1.post_attention_norm.weight
create_tensor: loading tensor blk.1.layer_output_scale.weight
create_tensor: loading tensor blk.1.ffn_norm.weight
create_tensor: loading tensor blk.1.ffn_gate.weight
create_tensor: loading tensor blk.1.ffn_up.weight
create_tensor: loading tensor blk.1.ffn_down.weight
create_tensor: loading tensor blk.1.post_ffw_norm.weight
create_tensor: loading tensor blk.2.attn_norm.weight
create_tensor: loading tensor blk.2.attn_q.weight
create_tensor: loading tensor blk.2.attn_k.weight
create_tensor: loading tensor blk.2.attn_v.weight
create_tensor: loading tensor blk.2.attn_output.weight
create_tensor: loading tensor blk.2.attn_q_norm.weight
create_tensor: loading tensor blk.2.attn_k_norm.weight
create_tensor: loading tensor blk.2.post_attention_norm.weight
create_tensor: loading tensor blk.2.layer_output_scale.weight
create_tensor: loading tensor blk.2.ffn_norm.weight
create_tensor: loading tensor blk.2.ffn_gate.weight
create_tensor: loading tensor blk.2.ffn_up.weight
create_tensor: loading tensor blk.2.ffn_down.weight
create_tensor: loading tensor blk.2.post_ffw_norm.weight
create_tensor: loading tensor blk.3.attn_norm.weight
create_tensor: loading tensor blk.3.attn_q.weight
create_tensor: loading tensor blk.3.attn_k.weight
create_tensor: loading tensor blk.3.attn_v.weight
create_tensor: loading tensor blk.3.attn_output.weight
create_tensor: loading tensor blk.3.attn_q_norm.weight
create_tensor: loading tensor blk.3.attn_k_norm.weight
create_tensor: loading tensor blk.3.post_attention_norm.weight
create_tensor: loading tensor blk.3.layer_output_scale.weight
create_tensor: loading tensor blk.3.ffn_norm.weight
create_tensor: loading tensor blk.3.ffn_gate.weight
create_tensor: loading tensor blk.3.ffn_up.weight
create_tensor: loading tensor blk.3.ffn_down.weight
create_tensor: loading tensor blk.3.post_ffw_norm.weight
create_tensor: loading tensor blk.4.attn_norm.weight
create_tensor: loading tensor blk.4.attn_q.weight
create_tensor: loading tensor blk.4.attn_k.weight
create_tensor: loading tensor blk.4.attn_v.weight
create_tensor: loading tensor blk.4.attn_output.weight
create_tensor: loading tensor blk.4.attn_q_norm.weight
create_tensor: loading tensor blk.4.attn_k_norm.weight
create_tensor: loading tensor blk.4.post_attention_norm.weight
create_tensor: loading tensor blk.4.layer_output_scale.weight
create_tensor: loading tensor blk.4.ffn_norm.weight
create_tensor: loading tensor blk.4.ffn_gate.weight
create_tensor: loading tensor blk.4.ffn_up.weight
create_tensor: loading tensor blk.4.ffn_down.weight
create_tensor: loading tensor blk.4.post_ffw_norm.weight
create_tensor: loading tensor blk.5.attn_norm.weight
create_tensor: loading tensor blk.5.attn_q.weight
create_tensor: loading tensor blk.5.attn_k.weight
create_tensor: loading tensor blk.5.attn_output.weight
create_tensor: loading tensor blk.5.attn_q_norm.weight
create_tensor: loading tensor blk.5.attn_k_norm.weight
create_tensor: loading tensor blk.5.post_attention_norm.weight
create_tensor: loading tensor blk.5.layer_output_scale.weight
create_tensor: loading tensor rope_freqs.weight
create_tensor: loading tensor blk.5.ffn_norm.weight
create_tensor: loading tensor blk.5.ffn_gate.weight
create_tensor: loading tensor blk.5.ffn_up.weight
create_tensor: loading tensor blk.5.ffn_down.weight
create_tensor: loading tensor blk.5.post_ffw_norm.weight
create_tensor: loading tensor blk.6.attn_norm.weight
create_tensor: loading tensor blk.6.attn_q.weight
create_tensor: loading tensor blk.6.attn_k.weight
create_tensor: loading tensor blk.6.attn_v.weight
create_tensor: loading tensor blk.6.attn_output.weight
create_tensor: loading tensor blk.6.attn_q_norm.weight
create_tensor: loading tensor blk.6.attn_k_norm.weight
create_tensor: loading tensor blk.6.post_attention_norm.weight
create_tensor: loading tensor blk.6.layer_output_scale.weight
create_tensor: loading tensor blk.6.ffn_norm.weight
create_tensor: loading tensor blk.6.ffn_gate.weight
create_tensor: loading tensor blk.6.ffn_up.weight
create_tensor: loading tensor blk.6.ffn_down.weight
create_tensor: loading tensor blk.6.post_ffw_norm.weight
create_tensor: loading tensor blk.7.attn_norm.weight
create_tensor: loading tensor blk.7.attn_q.weight
create_tensor: loading tensor blk.7.attn_k.weight
create_tensor: loading tensor blk.7.attn_v.weight
create_tensor: loading tensor blk.7.attn_output.weight
create_tensor: loading tensor blk.7.attn_q_norm.weight
create_tensor: loading tensor blk.7.attn_k_norm.weight
create_tensor: loading tensor blk.7.post_attention_norm.weight
create_tensor: loading tensor blk.7.layer_output_scale.weight
create_tensor: loading tensor blk.7.ffn_norm.weight
create_tensor: loading tensor blk.7.ffn_gate.weight
create_tensor: loading tensor blk.7.ffn_up.weight
create_tensor: loading tensor blk.7.ffn_down.weight
create_tensor: loading tensor blk.7.post_ffw_norm.weight
create_tensor: loading tensor blk.8.attn_norm.weight
create_tensor: loading tensor blk.8.attn_q.weight
create_tensor: loading tensor blk.8.attn_k.weight
create_tensor: loading tensor blk.8.attn_v.weight
create_tensor: loading tensor blk.8.attn_output.weight
create_tensor: loading tensor blk.8.attn_q_norm.weight
create_tensor: loading tensor blk.8.attn_k_norm.weight
create_tensor: loading tensor blk.8.post_attention_norm.weight
create_tensor: loading tensor blk.8.layer_output_scale.weight
create_tensor: loading tensor blk.8.ffn_norm.weight
create_tensor: loading tensor blk.8.ffn_gate.weight
create_tensor: loading tensor blk.8.ffn_up.weight
create_tensor: loading tensor blk.8.ffn_down.weight
create_tensor: loading tensor blk.8.post_ffw_norm.weight
create_tensor: loading tensor blk.9.attn_norm.weight
create_tensor: loading tensor blk.9.attn_q.weight
create_tensor: loading tensor blk.9.attn_k.weight
create_tensor: loading tensor blk.9.attn_v.weight
create_tensor: loading tensor blk.9.attn_output.weight
create_tensor: loading tensor blk.9.attn_q_norm.weight
create_tensor: loading tensor blk.9.attn_k_norm.weight
create_tensor: loading tensor blk.9.post_attention_norm.weight
create_tensor: loading tensor blk.9.layer_output_scale.weight
create_tensor: loading tensor blk.9.ffn_norm.weight
create_tensor: loading tensor blk.9.ffn_gate.weight
create_tensor: loading tensor blk.9.ffn_up.weight
create_tensor: loading tensor blk.9.ffn_down.weight
create_tensor: loading tensor blk.9.post_ffw_norm.weight
create_tensor: loading tensor blk.10.attn_norm.weight
create_tensor: loading tensor blk.10.attn_q.weight
create_tensor: loading tensor blk.10.attn_k.weight
create_tensor: loading tensor blk.10.attn_v.weight
create_tensor: loading tensor blk.10.attn_output.weight
create_tensor: loading tensor blk.10.attn_q_norm.weight
create_tensor: loading tensor blk.10.attn_k_norm.weight
create_tensor: loading tensor blk.10.post_attention_norm.weight
create_tensor: loading tensor blk.10.layer_output_scale.weight
create_tensor: loading tensor blk.10.ffn_norm.weight
create_tensor: loading tensor blk.10.ffn_gate.weight
create_tensor: loading tensor blk.10.ffn_up.weight
create_tensor: loading tensor blk.10.ffn_down.weight
create_tensor: loading tensor blk.10.post_ffw_norm.weight
create_tensor: loading tensor blk.11.attn_norm.weight
create_tensor: loading tensor blk.11.attn_q.weight
create_tensor: loading tensor blk.11.attn_k.weight
create_tensor: loading tensor blk.11.attn_output.weight
create_tensor: loading tensor blk.11.attn_q_norm.weight
create_tensor: loading tensor blk.11.attn_k_norm.weight
create_tensor: loading tensor blk.11.post_attention_norm.weight
create_tensor: loading tensor blk.11.layer_output_scale.weight
create_tensor: loading tensor blk.11.ffn_norm.weight
create_tensor: loading tensor blk.11.ffn_gate.weight
create_tensor: loading tensor blk.11.ffn_up.weight
create_tensor: loading tensor blk.11.ffn_down.weight
create_tensor: loading tensor blk.11.post_ffw_norm.weight
create_tensor: loading tensor blk.12.attn_norm.weight
create_tensor: loading tensor blk.12.attn_q.weight
create_tensor: loading tensor blk.12.attn_k.weight
create_tensor: loading tensor blk.12.attn_v.weight
create_tensor: loading tensor blk.12.attn_output.weight
create_tensor: loading tensor blk.12.attn_q_norm.weight
create_tensor: loading tensor blk.12.attn_k_norm.weight
create_tensor: loading tensor blk.12.post_attention_norm.weight
create_tensor: loading tensor blk.12.layer_output_scale.weight
create_tensor: loading tensor blk.12.ffn_norm.weight
create_tensor: loading tensor blk.12.ffn_gate.weight
create_tensor: loading tensor blk.12.ffn_up.weight
create_tensor: loading tensor blk.12.ffn_down.weight
create_tensor: loading tensor blk.12.post_ffw_norm.weight
create_tensor: loading tensor blk.13.attn_norm.weight
create_tensor: loading tensor blk.13.attn_q.weight
create_tensor: loading tensor blk.13.attn_k.weight
create_tensor: loading tensor blk.13.attn_v.weight
create_tensor: loading tensor blk.13.attn_output.weight
create_tensor: loading tensor blk.13.attn_q_norm.weight
create_tensor: loading tensor blk.13.attn_k_norm.weight
create_tensor: loading tensor blk.13.post_attention_norm.weight
create_tensor: loading tensor blk.13.layer_output_scale.weight
create_tensor: loading tensor blk.13.ffn_norm.weight
create_tensor: loading tensor blk.13.ffn_gate.weight
create_tensor: loading tensor blk.13.ffn_up.weight
create_tensor: loading tensor blk.13.ffn_down.weight
create_tensor: loading tensor blk.13.post_ffw_norm.weight
create_tensor: loading tensor blk.14.attn_norm.weight
create_tensor: loading tensor blk.14.attn_q.weight
create_tensor: loading tensor blk.14.attn_k.weight
create_tensor: loading tensor blk.14.attn_v.weight
create_tensor: loading tensor blk.14.attn_output.weight
create_tensor: loading tensor blk.14.attn_q_norm.weight
create_tensor: loading tensor blk.14.attn_k_norm.weight
create_tensor: loading tensor blk.14.post_attention_norm.weight
create_tensor: loading tensor blk.14.layer_output_scale.weight
create_tensor: loading tensor blk.14.ffn_norm.weight
create_tensor: loading tensor blk.14.ffn_gate.weight
create_tensor: loading tensor blk.14.ffn_up.weight
create_tensor: loading tensor blk.14.ffn_down.weight
create_tensor: loading tensor blk.14.post_ffw_norm.weight
create_tensor: loading tensor blk.15.attn_norm.weight
create_tensor: loading tensor blk.15.attn_q.weight
create_tensor: loading tensor blk.15.attn_k.weight
create_tensor: loading tensor blk.15.attn_v.weight
create_tensor: loading tensor blk.15.attn_output.weight
create_tensor: loading tensor blk.15.attn_q_norm.weight
create_tensor: loading tensor blk.15.attn_k_norm.weight
create_tensor: loading tensor blk.15.post_attention_norm.weight
create_tensor: loading tensor blk.15.layer_output_scale.weight
create_tensor: loading tensor blk.15.ffn_norm.weight
create_tensor: loading tensor blk.15.ffn_gate.weight
create_tensor: loading tensor blk.15.ffn_up.weight
create_tensor: loading tensor blk.15.ffn_down.weight
create_tensor: loading tensor blk.15.post_ffw_norm.weight
create_tensor: loading tensor blk.16.attn_norm.weight
create_tensor: loading tensor blk.16.attn_q.weight
create_tensor: loading tensor blk.16.attn_k.weight
create_tensor: loading tensor blk.16.attn_v.weight
create_tensor: loading tensor blk.16.attn_output.weight
create_tensor: loading tensor blk.16.attn_q_norm.weight
create_tensor: loading tensor blk.16.attn_k_norm.weight
create_tensor: loading tensor blk.16.post_attention_norm.weight
create_tensor: loading tensor blk.16.layer_output_scale.weight
create_tensor: loading tensor blk.16.ffn_norm.weight
create_tensor: loading tensor blk.16.ffn_gate.weight
create_tensor: loading tensor blk.16.ffn_up.weight
create_tensor: loading tensor blk.16.ffn_down.weight
create_tensor: loading tensor blk.16.post_ffw_norm.weight
create_tensor: loading tensor blk.17.attn_norm.weight
create_tensor: loading tensor blk.17.attn_q.weight
create_tensor: loading tensor blk.17.attn_k.weight
create_tensor: loading tensor blk.17.attn_output.weight
create_tensor: loading tensor blk.17.attn_q_norm.weight
create_tensor: loading tensor blk.17.attn_k_norm.weight
create_tensor: loading tensor blk.17.post_attention_norm.weight
create_tensor: loading tensor blk.17.layer_output_scale.weight
create_tensor: loading tensor blk.17.ffn_norm.weight
create_tensor: loading tensor blk.17.ffn_gate.weight
create_tensor: loading tensor blk.17.ffn_up.weight
create_tensor: loading tensor blk.17.ffn_down.weight
create_tensor: loading tensor blk.17.post_ffw_norm.weight
create_tensor: loading tensor blk.18.attn_norm.weight
create_tensor: loading tensor blk.18.attn_q.weight
create_tensor: loading tensor blk.18.attn_k.weight
create_tensor: loading tensor blk.18.attn_v.weight
create_tensor: loading tensor blk.18.attn_output.weight
create_tensor: loading tensor blk.18.attn_q_norm.weight
create_tensor: loading tensor blk.18.attn_k_norm.weight
create_tensor: loading tensor blk.18.post_attention_norm.weight
create_tensor: loading tensor blk.18.layer_output_scale.weight
create_tensor: loading tensor blk.18.ffn_norm.weight
create_tensor: loading tensor blk.18.ffn_gate.weight
create_tensor: loading tensor blk.18.ffn_up.weight
create_tensor: loading tensor blk.18.ffn_down.weight
create_tensor: loading tensor blk.18.post_ffw_norm.weight
create_tensor: loading tensor blk.19.attn_norm.weight
create_tensor: loading tensor blk.19.attn_q.weight
create_tensor: loading tensor blk.19.attn_k.weight
create_tensor: loading tensor blk.19.attn_v.weight
create_tensor: loading tensor blk.19.attn_output.weight
create_tensor: loading tensor blk.19.attn_q_norm.weight
create_tensor: loading tensor blk.19.attn_k_norm.weight
create_tensor: loading tensor blk.19.post_attention_norm.weight
create_tensor: loading tensor blk.19.layer_output_scale.weight
create_tensor: loading tensor blk.19.ffn_norm.weight
create_tensor: loading tensor blk.19.ffn_gate.weight
create_tensor: loading tensor blk.19.ffn_up.weight
create_tensor: loading tensor blk.19.ffn_down.weight
create_tensor: loading tensor blk.19.post_ffw_norm.weight
create_tensor: loading tensor blk.20.attn_norm.weight
create_tensor: loading tensor blk.20.attn_q.weight
create_tensor: loading tensor blk.20.attn_k.weight
create_tensor: loading tensor blk.20.attn_v.weight
create_tensor: loading tensor blk.20.attn_output.weight
create_tensor: loading tensor blk.20.attn_q_norm.weight
create_tensor: loading tensor blk.20.attn_k_norm.weight
create_tensor: loading tensor blk.20.post_attention_norm.weight
create_tensor: loading tensor blk.20.layer_output_scale.weight
create_tensor: loading tensor blk.20.ffn_norm.weight
create_tensor: loading tensor blk.20.ffn_gate.weight
create_tensor: loading tensor blk.20.ffn_up.weight
create_tensor: loading tensor blk.20.ffn_down.weight
create_tensor: loading tensor blk.20.post_ffw_norm.weight
create_tensor: loading tensor blk.21.attn_norm.weight
create_tensor: loading tensor blk.21.attn_q.weight
create_tensor: loading tensor blk.21.attn_k.weight
create_tensor: loading tensor blk.21.attn_v.weight
create_tensor: loading tensor blk.21.attn_output.weight
create_tensor: loading tensor blk.21.attn_q_norm.weight
create_tensor: loading tensor blk.21.attn_k_norm.weight
create_tensor: loading tensor blk.21.post_attention_norm.weight
create_tensor: loading tensor blk.21.layer_output_scale.weight
create_tensor: loading tensor blk.21.ffn_norm.weight
create_tensor: loading tensor blk.21.ffn_gate.weight
create_tensor: loading tensor blk.21.ffn_up.weight
create_tensor: loading tensor blk.21.ffn_down.weight
create_tensor: loading tensor blk.21.post_ffw_norm.weight
create_tensor: loading tensor blk.22.attn_norm.weight
create_tensor: loading tensor blk.22.attn_q.weight
create_tensor: loading tensor blk.22.attn_k.weight
create_tensor: loading tensor blk.22.attn_v.weight
create_tensor: loading tensor blk.22.attn_output.weight
create_tensor: loading tensor blk.22.attn_q_norm.weight
create_tensor: loading tensor blk.22.attn_k_norm.weight
create_tensor: loading tensor blk.22.post_attention_norm.weight
create_tensor: loading tensor blk.22.layer_output_scale.weight
create_tensor: loading tensor blk.22.ffn_norm.weight
create_tensor: loading tensor blk.22.ffn_gate.weight
create_tensor: loading tensor blk.22.ffn_up.weight
create_tensor: loading tensor blk.22.ffn_down.weight
create_tensor: loading tensor blk.22.post_ffw_norm.weight
create_tensor: loading tensor blk.23.attn_norm.weight
create_tensor: loading tensor blk.23.attn_q.weight
create_tensor: loading tensor blk.23.attn_k.weight
create_tensor: loading tensor blk.23.attn_output.weight
create_tensor: loading tensor blk.23.attn_q_norm.weight
create_tensor: loading tensor blk.23.attn_k_norm.weight
create_tensor: loading tensor blk.23.post_attention_norm.weight
create_tensor: loading tensor blk.23.layer_output_scale.weight
create_tensor: loading tensor blk.23.ffn_norm.weight
create_tensor: loading tensor blk.23.ffn_gate.weight
create_tensor: loading tensor blk.23.ffn_up.weight
create_tensor: loading tensor blk.23.ffn_down.weight
create_tensor: loading tensor blk.23.post_ffw_norm.weight
create_tensor: loading tensor blk.24.attn_norm.weight
create_tensor: loading tensor blk.24.attn_q.weight
create_tensor: loading tensor blk.24.attn_k.weight
create_tensor: loading tensor blk.24.attn_v.weight
create_tensor: loading tensor blk.24.attn_output.weight
create_tensor: loading tensor blk.24.attn_q_norm.weight
create_tensor: loading tensor blk.24.attn_k_norm.weight
create_tensor: loading tensor blk.24.post_attention_norm.weight
create_tensor: loading tensor blk.24.layer_output_scale.weight
create_tensor: loading tensor blk.24.ffn_norm.weight
create_tensor: loading tensor blk.24.ffn_gate.weight
create_tensor: loading tensor blk.24.ffn_up.weight
create_tensor: loading tensor blk.24.ffn_down.weight
create_tensor: loading tensor blk.24.post_ffw_norm.weight
create_tensor: loading tensor blk.25.attn_norm.weight
create_tensor: loading tensor blk.25.attn_q.weight
create_tensor: loading tensor blk.25.attn_k.weight
create_tensor: loading tensor blk.25.attn_v.weight
create_tensor: loading tensor blk.25.attn_output.weight
create_tensor: loading tensor blk.25.attn_q_norm.weight
create_tensor: loading tensor blk.25.attn_k_norm.weight
create_tensor: loading tensor blk.25.post_attention_norm.weight
create_tensor: loading tensor blk.25.layer_output_scale.weight
create_tensor: loading tensor blk.25.ffn_norm.weight
create_tensor: loading tensor blk.25.ffn_gate.weight
create_tensor: loading tensor blk.25.ffn_up.weight
create_tensor: loading tensor blk.25.ffn_down.weight
create_tensor: loading tensor blk.25.post_ffw_norm.weight
create_tensor: loading tensor blk.26.attn_norm.weight
create_tensor: loading tensor blk.26.attn_q.weight
create_tensor: loading tensor blk.26.attn_k.weight
create_tensor: loading tensor blk.26.attn_v.weight
create_tensor: loading tensor blk.26.attn_output.weight
create_tensor: loading tensor blk.26.attn_q_norm.weight
create_tensor: loading tensor blk.26.attn_k_norm.weight
create_tensor: loading tensor blk.26.post_attention_norm.weight
create_tensor: loading tensor blk.26.layer_output_scale.weight
create_tensor: loading tensor blk.26.ffn_norm.weight
create_tensor: loading tensor blk.26.ffn_gate.weight
create_tensor: loading tensor blk.26.ffn_up.weight
create_tensor: loading tensor blk.26.ffn_down.weight
create_tensor: loading tensor blk.26.post_ffw_norm.weight
create_tensor: loading tensor blk.27.attn_norm.weight
create_tensor: loading tensor blk.27.attn_q.weight
create_tensor: loading tensor blk.27.attn_k.weight
create_tensor: loading tensor blk.27.attn_v.weight
create_tensor: loading tensor blk.27.attn_output.weight
create_tensor: loading tensor blk.27.attn_q_norm.weight
create_tensor: loading tensor blk.27.attn_k_norm.weight
create_tensor: loading tensor blk.27.post_attention_norm.weight
create_tensor: loading tensor blk.27.layer_output_scale.weight
create_tensor: loading tensor blk.27.ffn_norm.weight
create_tensor: loading tensor blk.27.ffn_gate.weight
create_tensor: loading tensor blk.27.ffn_up.weight
create_tensor: loading tensor blk.27.ffn_down.weight
create_tensor: loading tensor blk.27.post_ffw_norm.weight
create_tensor: loading tensor blk.28.attn_norm.weight
create_tensor: loading tensor blk.28.attn_q.weight
create_tensor: loading tensor blk.28.attn_k.weight
create_tensor: loading tensor blk.28.attn_v.weight
create_tensor: loading tensor blk.28.attn_output.weight
create_tensor: loading tensor blk.28.attn_q_norm.weight
create_tensor: loading tensor blk.28.attn_k_norm.weight
create_tensor: loading tensor blk.28.post_attention_norm.weight
create_tensor: loading tensor blk.28.layer_output_scale.weight
create_tensor: loading tensor blk.28.ffn_norm.weight
create_tensor: loading tensor blk.28.ffn_gate.weight
create_tensor: loading tensor blk.28.ffn_up.weight
create_tensor: loading tensor blk.28.ffn_down.weight
create_tensor: loading tensor blk.28.post_ffw_norm.weight
create_tensor: loading tensor blk.29.attn_norm.weight
create_tensor: loading tensor blk.29.attn_q.weight
create_tensor: loading tensor blk.29.attn_k.weight
create_tensor: loading tensor blk.29.attn_output.weight
create_tensor: loading tensor blk.29.attn_q_norm.weight
create_tensor: loading tensor blk.29.attn_k_norm.weight
create_tensor: loading tensor blk.29.post_attention_norm.weight
create_tensor: loading tensor blk.29.layer_output_scale.weight
create_tensor: loading tensor blk.29.ffn_norm.weight
create_tensor: loading tensor blk.29.ffn_gate.weight
create_tensor: loading tensor blk.29.ffn_up.weight
create_tensor: loading tensor blk.29.ffn_down.weight
create_tensor: loading tensor blk.29.post_ffw_norm.weight
create_tensor: loading tensor blk.30.attn_norm.weight
create_tensor: loading tensor blk.30.attn_q.weight
create_tensor: loading tensor blk.30.attn_k.weight
create_tensor: loading tensor blk.30.attn_v.weight
create_tensor: loading tensor blk.30.attn_output.weight
create_tensor: loading tensor blk.30.attn_q_norm.weight
create_tensor: loading tensor blk.30.attn_k_norm.weight
create_tensor: loading tensor blk.30.post_attention_norm.weight
create_tensor: loading tensor blk.30.layer_output_scale.weight
create_tensor: loading tensor blk.30.ffn_norm.weight
create_tensor: loading tensor blk.30.ffn_gate.weight
create_tensor: loading tensor blk.30.ffn_up.weight
create_tensor: loading tensor blk.30.ffn_down.weight
create_tensor: loading tensor blk.30.post_ffw_norm.weight
create_tensor: loading tensor blk.31.attn_norm.weight
create_tensor: loading tensor blk.31.attn_q.weight
create_tensor: loading tensor blk.31.attn_k.weight
create_tensor: loading tensor blk.31.attn_v.weight
create_tensor: loading tensor blk.31.attn_output.weight
create_tensor: loading tensor blk.31.attn_q_norm.weight
create_tensor: loading tensor blk.31.attn_k_norm.weight
create_tensor: loading tensor blk.31.post_attention_norm.weight
create_tensor: loading tensor blk.31.layer_output_scale.weight
create_tensor: loading tensor blk.31.ffn_norm.weight
create_tensor: loading tensor blk.31.ffn_gate.weight
create_tensor: loading tensor blk.31.ffn_up.weight
create_tensor: loading tensor blk.31.ffn_down.weight
create_tensor: loading tensor blk.31.post_ffw_norm.weight
create_tensor: loading tensor blk.32.attn_norm.weight
create_tensor: loading tensor blk.32.attn_q.weight
create_tensor: loading tensor blk.32.attn_k.weight
create_tensor: loading tensor blk.32.attn_v.weight
create_tensor: loading tensor blk.32.attn_output.weight
create_tensor: loading tensor blk.32.attn_q_norm.weight
create_tensor: loading tensor blk.32.attn_k_norm.weight
create_tensor: loading tensor blk.32.post_attention_norm.weight
create_tensor: loading tensor blk.32.layer_output_scale.weight
create_tensor: loading tensor blk.32.ffn_norm.weight
create_tensor: loading tensor blk.32.ffn_gate.weight
create_tensor: loading tensor blk.32.ffn_up.weight
create_tensor: loading tensor blk.32.ffn_down.weight
create_tensor: loading tensor blk.32.post_ffw_norm.weight
create_tensor: loading tensor blk.33.attn_norm.weight
create_tensor: loading tensor blk.33.attn_q.weight
create_tensor: loading tensor blk.33.attn_k.weight
create_tensor: loading tensor blk.33.attn_v.weight
create_tensor: loading tensor blk.33.attn_output.weight
create_tensor: loading tensor blk.33.attn_q_norm.weight
create_tensor: loading tensor blk.33.attn_k_norm.weight
create_tensor: loading tensor blk.33.post_attention_norm.weight
create_tensor: loading tensor blk.33.layer_output_scale.weight
create_tensor: loading tensor blk.33.ffn_norm.weight
create_tensor: loading tensor blk.33.ffn_gate.weight
create_tensor: loading tensor blk.33.ffn_up.weight
create_tensor: loading tensor blk.33.ffn_down.weight
create_tensor: loading tensor blk.33.post_ffw_norm.weight
create_tensor: loading tensor blk.34.attn_norm.weight
create_tensor: loading tensor blk.34.attn_q.weight
create_tensor: loading tensor blk.34.attn_k.weight
create_tensor: loading tensor blk.34.attn_v.weight
create_tensor: loading tensor blk.34.attn_output.weight
create_tensor: loading tensor blk.34.attn_q_norm.weight
create_tensor: loading tensor blk.34.attn_k_norm.weight
create_tensor: loading tensor blk.34.post_attention_norm.weight
create_tensor: loading tensor blk.34.layer_output_scale.weight
create_tensor: loading tensor blk.34.ffn_norm.weight
create_tensor: loading tensor blk.34.ffn_gate.weight
create_tensor: loading tensor blk.34.ffn_up.weight
create_tensor: loading tensor blk.34.ffn_down.weight
create_tensor: loading tensor blk.34.post_ffw_norm.weight
create_tensor: loading tensor blk.35.attn_norm.weight
create_tensor: loading tensor blk.35.attn_q.weight
create_tensor: loading tensor blk.35.attn_k.weight
create_tensor: loading tensor blk.35.attn_output.weight
create_tensor: loading tensor blk.35.attn_q_norm.weight
create_tensor: loading tensor blk.35.attn_k_norm.weight
create_tensor: loading tensor blk.35.post_attention_norm.weight
create_tensor: loading tensor blk.35.layer_output_scale.weight
create_tensor: loading tensor blk.35.ffn_norm.weight
create_tensor: loading tensor blk.35.ffn_gate.weight
create_tensor: loading tensor blk.35.ffn_up.weight
create_tensor: loading tensor blk.35.ffn_down.weight
create_tensor: loading tensor blk.35.post_ffw_norm.weight
create_tensor: loading tensor blk.36.attn_norm.weight
create_tensor: loading tensor blk.36.attn_q.weight
create_tensor: loading tensor blk.36.attn_k.weight
create_tensor: loading tensor blk.36.attn_v.weight
create_tensor: loading tensor blk.36.attn_output.weight
create_tensor: loading tensor blk.36.attn_q_norm.weight
create_tensor: loading tensor blk.36.attn_k_norm.weight
create_tensor: loading tensor blk.36.post_attention_norm.weight
create_tensor: loading tensor blk.36.layer_output_scale.weight
create_tensor: loading tensor blk.36.ffn_norm.weight
create_tensor: loading tensor blk.36.ffn_gate.weight
create_tensor: loading tensor blk.36.ffn_up.weight
create_tensor: loading tensor blk.36.ffn_down.weight
create_tensor: loading tensor blk.36.post_ffw_norm.weight
create_tensor: loading tensor blk.37.attn_norm.weight
create_tensor: loading tensor blk.37.attn_q.weight
create_tensor: loading tensor blk.37.attn_k.weight
create_tensor: loading tensor blk.37.attn_v.weight
create_tensor: loading tensor blk.37.attn_output.weight
create_tensor: loading tensor blk.37.attn_q_norm.weight
create_tensor: loading tensor blk.37.attn_k_norm.weight
create_tensor: loading tensor blk.37.post_attention_norm.weight
create_tensor: loading tensor blk.37.layer_output_scale.weight
create_tensor: loading tensor blk.37.ffn_norm.weight
create_tensor: loading tensor blk.37.ffn_gate.weight
create_tensor: loading tensor blk.37.ffn_up.weight
create_tensor: loading tensor blk.37.ffn_down.weight
create_tensor: loading tensor blk.37.post_ffw_norm.weight
create_tensor: loading tensor blk.38.attn_norm.weight
create_tensor: loading tensor blk.38.attn_q.weight
create_tensor: loading tensor blk.38.attn_k.weight
create_tensor: loading tensor blk.38.attn_v.weight
create_tensor: loading tensor blk.38.attn_output.weight
create_tensor: loading tensor blk.38.attn_q_norm.weight
create_tensor: loading tensor blk.38.attn_k_norm.weight
create_tensor: loading tensor blk.38.post_attention_norm.weight
create_tensor: loading tensor blk.38.layer_output_scale.weight
create_tensor: loading tensor blk.38.ffn_norm.weight
create_tensor: loading tensor blk.38.ffn_gate.weight
create_tensor: loading tensor blk.38.ffn_up.weight
create_tensor: loading tensor blk.38.ffn_down.weight
create_tensor: loading tensor blk.38.post_ffw_norm.weight
create_tensor: loading tensor blk.39.attn_norm.weight
create_tensor: loading tensor blk.39.attn_q.weight
create_tensor: loading tensor blk.39.attn_k.weight
create_tensor: loading tensor blk.39.attn_v.weight
create_tensor: loading tensor blk.39.attn_output.weight
create_tensor: loading tensor blk.39.attn_q_norm.weight
create_tensor: loading tensor blk.39.attn_k_norm.weight
create_tensor: loading tensor blk.39.post_attention_norm.weight
create_tensor: loading tensor blk.39.layer_output_scale.weight
create_tensor: loading tensor blk.39.ffn_norm.weight
create_tensor: loading tensor blk.39.ffn_gate.weight
create_tensor: loading tensor blk.39.ffn_up.weight
create_tensor: loading tensor blk.39.ffn_down.weight
create_tensor: loading tensor blk.39.post_ffw_norm.weight
create_tensor: loading tensor blk.40.attn_norm.weight
create_tensor: loading tensor blk.40.attn_q.weight
create_tensor: loading tensor blk.40.attn_k.weight
create_tensor: loading tensor blk.40.attn_v.weight
create_tensor: loading tensor blk.40.attn_output.weight
create_tensor: loading tensor blk.40.attn_q_norm.weight
create_tensor: loading tensor blk.40.attn_k_norm.weight
create_tensor: loading tensor blk.40.post_attention_norm.weight
create_tensor: loading tensor blk.40.layer_output_scale.weight
create_tensor: loading tensor blk.40.ffn_norm.weight
create_tensor: loading tensor blk.40.ffn_gate.weight
create_tensor: loading tensor blk.40.ffn_up.weight
create_tensor: loading tensor blk.40.ffn_down.weight
create_tensor: loading tensor blk.40.post_ffw_norm.weight
create_tensor: loading tensor blk.41.attn_norm.weight
create_tensor: loading tensor blk.41.attn_q.weight
create_tensor: loading tensor blk.41.attn_k.weight
create_tensor: loading tensor blk.41.attn_output.weight
create_tensor: loading tensor blk.41.attn_q_norm.weight
create_tensor: loading tensor blk.41.attn_k_norm.weight
create_tensor: loading tensor blk.41.post_attention_norm.weight
create_tensor: loading tensor blk.41.layer_output_scale.weight
create_tensor: loading tensor blk.41.ffn_norm.weight
create_tensor: loading tensor blk.41.ffn_gate.weight
create_tensor: loading tensor blk.41.ffn_up.weight
create_tensor: loading tensor blk.41.ffn_down.weight
create_tensor: loading tensor blk.41.post_ffw_norm.weight
create_tensor: loading tensor blk.42.attn_norm.weight
create_tensor: loading tensor blk.42.attn_q.weight
create_tensor: loading tensor blk.42.attn_k.weight
create_tensor: loading tensor blk.42.attn_v.weight
create_tensor: loading tensor blk.42.attn_output.weight
create_tensor: loading tensor blk.42.attn_q_norm.weight
create_tensor: loading tensor blk.42.attn_k_norm.weight
create_tensor: loading tensor blk.42.post_attention_norm.weight
create_tensor: loading tensor blk.42.layer_output_scale.weight
create_tensor: loading tensor blk.42.ffn_norm.weight
create_tensor: loading tensor blk.42.ffn_gate.weight
create_tensor: loading tensor blk.42.ffn_up.weight
create_tensor: loading tensor blk.42.ffn_down.weight
create_tensor: loading tensor blk.42.post_ffw_norm.weight
create_tensor: loading tensor blk.43.attn_norm.weight
create_tensor: loading tensor blk.43.attn_q.weight
create_tensor: loading tensor blk.43.attn_k.weight
create_tensor: loading tensor blk.43.attn_v.weight
create_tensor: loading tensor blk.43.attn_output.weight
create_tensor: loading tensor blk.43.attn_q_norm.weight
create_tensor: loading tensor blk.43.attn_k_norm.weight
create_tensor: loading tensor blk.43.post_attention_norm.weight
create_tensor: loading tensor blk.43.layer_output_scale.weight
create_tensor: loading tensor blk.43.ffn_norm.weight
create_tensor: loading tensor blk.43.ffn_gate.weight
create_tensor: loading tensor blk.43.ffn_up.weight
create_tensor: loading tensor blk.43.ffn_down.weight
create_tensor: loading tensor blk.43.post_ffw_norm.weight
create_tensor: loading tensor blk.44.attn_norm.weight
create_tensor: loading tensor blk.44.attn_q.weight
create_tensor: loading tensor blk.44.attn_k.weight
create_tensor: loading tensor blk.44.attn_v.weight
create_tensor: loading tensor blk.44.attn_output.weight
create_tensor: loading tensor blk.44.attn_q_norm.weight
create_tensor: loading tensor blk.44.attn_k_norm.weight
create_tensor: loading tensor blk.44.post_attention_norm.weight
create_tensor: loading tensor blk.44.layer_output_scale.weight
create_tensor: loading tensor blk.44.ffn_norm.weight
create_tensor: loading tensor blk.44.ffn_gate.weight
create_tensor: loading tensor blk.44.ffn_up.weight
create_tensor: loading tensor blk.44.ffn_down.weight
create_tensor: loading tensor blk.44.post_ffw_norm.weight
create_tensor: loading tensor blk.45.attn_norm.weight
create_tensor: loading tensor blk.45.attn_q.weight
create_tensor: loading tensor blk.45.attn_k.weight
create_tensor: loading tensor blk.45.attn_v.weight
create_tensor: loading tensor blk.45.attn_output.weight
create_tensor: loading tensor blk.45.attn_q_norm.weight
create_tensor: loading tensor blk.45.attn_k_norm.weight
create_tensor: loading tensor blk.45.post_attention_norm.weight
create_tensor: loading tensor blk.45.layer_output_scale.weight
create_tensor: loading tensor blk.45.ffn_norm.weight
create_tensor: loading tensor blk.45.ffn_gate.weight
create_tensor: loading tensor blk.45.ffn_up.weight
create_tensor: loading tensor blk.45.ffn_down.weight
create_tensor: loading tensor blk.45.post_ffw_norm.weight
create_tensor: loading tensor blk.46.attn_norm.weight
create_tensor: loading tensor blk.46.attn_q.weight
create_tensor: loading tensor blk.46.attn_k.weight
create_tensor: loading tensor blk.46.attn_v.weight
create_tensor: loading tensor blk.46.attn_output.weight
create_tensor: loading tensor blk.46.attn_q_norm.weight
create_tensor: loading tensor blk.46.attn_k_norm.weight
create_tensor: loading tensor blk.46.post_attention_norm.weight
create_tensor: loading tensor blk.46.layer_output_scale.weight
create_tensor: loading tensor blk.46.ffn_norm.weight
create_tensor: loading tensor blk.46.ffn_gate.weight
create_tensor: loading tensor blk.46.ffn_up.weight
create_tensor: loading tensor blk.46.ffn_down.weight
create_tensor: loading tensor blk.46.post_ffw_norm.weight
create_tensor: loading tensor blk.47.attn_norm.weight
create_tensor: loading tensor blk.47.attn_q.weight
create_tensor: loading tensor blk.47.attn_k.weight
create_tensor: loading tensor blk.47.attn_output.weight
create_tensor: loading tensor blk.47.attn_q_norm.weight
create_tensor: loading tensor blk.47.attn_k_norm.weight
create_tensor: loading tensor blk.47.post_attention_norm.weight
create_tensor: loading tensor blk.47.layer_output_scale.weight
create_tensor: loading tensor blk.47.ffn_norm.weight
create_tensor: loading tensor blk.47.ffn_gate.weight
create_tensor: loading tensor blk.47.ffn_up.weight
create_tensor: loading tensor blk.47.ffn_down.weight
create_tensor: loading tensor blk.47.post_ffw_norm.weight
done_getting_tensors: tensor 'token_embd.weight' (q4_K) (and 674 others) cannot be used with preferred buffer type CPU_REPACK, using CPU instead
load_tensors:          CPU model buffer size =  7316.85 MiB
load_all_data: no device found for buffer type CPU for async uploads
........................................................................................
load_tensors: releasing mmaps (not needed after tensor copy)
llama_model_loader: direct I/O is enabled, disabling mmap
llama_model_loader: loaded meta data with 48 key-value pairs and 49 tensors from c:\llama.cpp\models\Gemma-4\12B\Unsloth\gemma-4-12b-it-Q4_K_M-assistant_Unsloth.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = gemma4-assistant
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                     general.sampling.top_k i32              = 64
llama_model_loader: - kv   3:                     general.sampling.top_p f32              = 0.950000
llama_model_loader: - kv   4:                      general.sampling.temp f32              = 1.000000
llama_model_loader: - kv   5:                               general.name str              = Gemma 4 12B It Assistant
llama_model_loader: - kv   6:                           general.finetune str              = 12b-it-assistant
llama_model_loader: - kv   7:                           general.basename str              = gemma-4
llama_model_loader: - kv   8:                         general.size_label str              = 423M
llama_model_loader: - kv   9:                            general.license str              = apache-2.0
llama_model_loader: - kv  10:                       general.license.link str              = https://ai.google.dev/gemma/docs/gemm...
llama_model_loader: - kv  11:                               general.tags arr[str,1]       = ["any-to-any"]
llama_model_loader: - kv  12:               gemma4-assistant.block_count u32              = 4
llama_model_loader: - kv  13:            gemma4-assistant.context_length u32              = 262144
llama_model_loader: - kv  14:          gemma4-assistant.embedding_length u32              = 1024
llama_model_loader: - kv  15:       gemma4-assistant.feed_forward_length u32              = 8192
llama_model_loader: - kv  16:      gemma4-assistant.attention.head_count u32              = 16
llama_model_loader: - kv  17:   gemma4-assistant.attention.head_count_kv arr[i32,4]       = [8, 8, 8, 1]
llama_model_loader: - kv  18:            gemma4-assistant.rope.freq_base f32              = 1000000.000000
llama_model_loader: - kv  19:        gemma4-assistant.rope.freq_base_swa f32              = 10000.000000
llama_model_loader: - kv  20: gemma4-assistant.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  21:      gemma4-assistant.attention.key_length u32              = 512
llama_model_loader: - kv  22:    gemma4-assistant.attention.value_length u32              = 512
llama_model_loader: - kv  23:                          general.file_type u32              = 7
llama_model_loader: - kv  24:  gemma4-assistant.attention.sliding_window u32              = 1024
llama_model_loader: - kv  25: gemma4-assistant.attention.shared_kv_layers u32              = 4
llama_model_loader: - kv  26: gemma4-assistant.embedding_length_per_layer_input u32              = 0
llama_model_loader: - kv  27: gemma4-assistant.attention.sliding_window_pattern arr[bool,4]      = [true, true, true, false]
llama_model_loader: - kv  28:  gemma4-assistant.attention.key_length_swa u32              = 256
llama_model_loader: - kv  29: gemma4-assistant.attention.value_length_swa u32              = 256
llama_model_loader: - kv  30:      gemma4-assistant.rope.dimension_count u32              = 512
llama_model_loader: - kv  31:  gemma4-assistant.rope.dimension_count_swa u32              = 256
llama_model_loader: - kv  32:      gemma4-assistant.embedding_length_out u32              = 3840
llama_model_loader: - kv  33:      gemma4-assistant.nextn_predict_layers u32              = 4
llama_model_loader: - kv  34:               general.quantization_version u32              = 2
llama_model_loader: - kv  35:                       tokenizer.ggml.model str              = gemma4
llama_model_loader: - kv  36:                      tokenizer.ggml.tokens arr[str,262144]  = ["<pad>", "<eos>", "<bos>", "<unk>", ...
llama_model_loader: - kv  37:                      tokenizer.ggml.scores arr[f32,262144]  = [-1000.000000, -1000.000000, -1000.00...
llama_model_loader: - kv  38:                  tokenizer.ggml.token_type arr[i32,262144]  = [3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  39:                      tokenizer.ggml.merges arr[str,514906]  = ["\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n \n", ...
llama_model_loader: - kv  40:                tokenizer.ggml.bos_token_id u32              = 2
llama_model_loader: - kv  41:                tokenizer.ggml.eos_token_id u32              = 1
llama_model_loader: - kv  42:            tokenizer.ggml.unknown_token_id u32              = 3
llama_model_loader: - kv  43:            tokenizer.ggml.padding_token_id u32              = 0
llama_model_loader: - kv  44:               tokenizer.ggml.mask_token_id u32              = 4
llama_model_loader: - kv  45:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  46:               tokenizer.ggml.add_sep_token bool             = false
llama_model_loader: - kv  47:            tokenizer.ggml.add_space_prefix bool             = false
llama_model_loader: - type  f32:   26 tensors
llama_model_loader: - type q8_0:   23 tensors
print_info: file format = GGUF V3 (latest)
print_info: file type   = Q8_0
print_info: file size   = 428.53 MiB (8.50 BPW)
llama_model_load: error loading model: unknown model architecture: 'gemma4-assistant'
llama_model_load_from_file_impl: failed to load model
llama_model_load_mtp_from_file: failed to load assistant from 'c:\llama.cpp\models\Gemma-4\12B\Unsloth\gemma-4-12b-it-Q4_K_M-assistant_Unsloth.gguf'
Warning: failed to load MTP head 'c:\llama.cpp\models\Gemma-4\12B\Unsloth\gemma-4-12b-it-Q4_K_M-assistant_Unsloth.gguf' (rc=-3); continuing without MTP
llama_context: constructing llama_context
llama_context: n_seq_max     = 1
llama_context: n_ctx         = 2048
llama_context: n_ctx_seq     = 2048
llama_context: n_batch       = 512
llama_context: n_ubatch      = 512
llama_context: causal_attn   = 1
llama_context: flash_attn    = auto
llama_context: kv_unified    = false
llama_context: freq_base     = 1000000.0
llama_context: freq_scale    = 1
llama_context: n_rs_seq      = 0
llama_context: n_ctx_seq (2048) < n_ctx_train (262144) -- the full capacity of the model will not be utilized
set_abort_callback: call
llama_context:        CPU  output buffer size =     1.00 MiB
llama_kv_cache_iswa: using full-size SWA cache (ref: https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055)
llama_kv_cache_iswa: creating non-SWA KV cache, size = 2048 cells
llama_kv_cache: layer   0: filtered
llama_kv_cache: layer   1: filtered
llama_kv_cache: layer   2: filtered
llama_kv_cache: layer   3: filtered
llama_kv_cache: layer   4: filtered
llama_kv_cache: layer   5: dev = CPU
llama_kv_cache: layer   6: filtered
llama_kv_cache: layer   7: filtered
llama_kv_cache: layer   8: filtered
llama_kv_cache: layer   9: filtered
llama_kv_cache: layer  10: filtered
llama_kv_cache: layer  11: dev = CPU
llama_kv_cache: layer  12: filtered
llama_kv_cache: layer  13: filtered
llama_kv_cache: layer  14: filtered
llama_kv_cache: layer  15: filtered
llama_kv_cache: layer  16: filtered
llama_kv_cache: layer  17: dev = CPU
llama_kv_cache: layer  18: filtered
llama_kv_cache: layer  19: filtered
llama_kv_cache: layer  20: filtered
llama_kv_cache: layer  21: filtered
llama_kv_cache: layer  22: filtered
llama_kv_cache: layer  23: dev = CPU
llama_kv_cache: layer  24: filtered
llama_kv_cache: layer  25: filtered
llama_kv_cache: layer  26: filtered
llama_kv_cache: layer  27: filtered
llama_kv_cache: layer  28: filtered
llama_kv_cache: layer  29: dev = CPU
llama_kv_cache: layer  30: filtered
llama_kv_cache: layer  31: filtered
llama_kv_cache: layer  32: filtered
llama_kv_cache: layer  33: filtered
llama_kv_cache: layer  34: filtered
llama_kv_cache: layer  35: dev = CPU
llama_kv_cache: layer  36: filtered
llama_kv_cache: layer  37: filtered
llama_kv_cache: layer  38: filtered
llama_kv_cache: layer  39: filtered
llama_kv_cache: layer  40: filtered
llama_kv_cache: layer  41: dev = CPU
llama_kv_cache: layer  42: filtered
llama_kv_cache: layer  43: filtered
llama_kv_cache: layer  44: filtered
llama_kv_cache: layer  45: filtered
llama_kv_cache: layer  46: filtered
llama_kv_cache: layer  47: dev = CPU
llama_kv_cache: reusing layers:
llama_kv_cache: - layer   0: no reuse
llama_kv_cache: - layer   1: no reuse
llama_kv_cache: - layer   2: no reuse
llama_kv_cache: - layer   3: no reuse
llama_kv_cache: - layer   4: no reuse
llama_kv_cache: - layer   5: no reuse
llama_kv_cache: - layer   6: no reuse
llama_kv_cache: - layer   7: no reuse
llama_kv_cache: - layer   8: no reuse
llama_kv_cache: - layer   9: no reuse
llama_kv_cache: - layer  10: no reuse
llama_kv_cache: - layer  11: no reuse
llama_kv_cache: - layer  12: no reuse
llama_kv_cache: - layer  13: no reuse
llama_kv_cache: - layer  14: no reuse
llama_kv_cache: - layer  15: no reuse
llama_kv_cache: - layer  16: no reuse
llama_kv_cache: - layer  17: no reuse
llama_kv_cache: - layer  18: no reuse
llama_kv_cache: - layer  19: no reuse
llama_kv_cache: - layer  20: no reuse
llama_kv_cache: - layer  21: no reuse
llama_kv_cache: - layer  22: no reuse
llama_kv_cache: - layer  23: no reuse
llama_kv_cache: - layer  24: no reuse
llama_kv_cache: - layer  25: no reuse
llama_kv_cache: - layer  26: no reuse
llama_kv_cache: - layer  27: no reuse
llama_kv_cache: - layer  28: no reuse
llama_kv_cache: - layer  29: no reuse
llama_kv_cache: - layer  30: no reuse
llama_kv_cache: - layer  31: no reuse
llama_kv_cache: - layer  32: no reuse
llama_kv_cache: - layer  33: no reuse
llama_kv_cache: - layer  34: no reuse
llama_kv_cache: - layer  35: no reuse
llama_kv_cache: - layer  36: no reuse
llama_kv_cache: - layer  37: no reuse
llama_kv_cache: - layer  38: no reuse
llama_kv_cache: - layer  39: no reuse
llama_kv_cache: - layer  40: no reuse
llama_kv_cache: - layer  41: no reuse
llama_kv_cache: - layer  42: no reuse
llama_kv_cache: - layer  43: no reuse
llama_kv_cache: - layer  44: no reuse
llama_kv_cache: - layer  45: no reuse
llama_kv_cache: - layer  46: no reuse
llama_kv_cache: - layer  47: no reuse
llama_kv_cache:        CPU KV buffer size =    32.00 MiB
llama_kv_cache: size =   32.00 MiB (  2048 cells,   8 layers,  1/1 seqs), K (f16):   16.00 MiB, V (f16):   16.00 MiB
llama_kv_cache: attn_rot_k = 0, n_embd_head_k_all = 512
llama_kv_cache: attn_rot_v = 0, n_embd_head_k_all = 512
llama_kv_cache_iswa: creating     SWA KV cache, size = 2048 cells
llama_kv_cache: layer   0: dev = CPU
llama_kv_cache: layer   1: dev = CPU
llama_kv_cache: layer   2: dev = CPU
llama_kv_cache: layer   3: dev = CPU
llama_kv_cache: layer   4: dev = CPU
llama_kv_cache: layer   5: filtered
llama_kv_cache: layer   6: dev = CPU
llama_kv_cache: layer   7: dev = CPU
llama_kv_cache: layer   8: dev = CPU
llama_kv_cache: layer   9: dev = CPU
llama_kv_cache: layer  10: dev = CPU
llama_kv_cache: layer  11: filtered
llama_kv_cache: layer  12: dev = CPU
llama_kv_cache: layer  13: dev = CPU
llama_kv_cache: layer  14: dev = CPU
llama_kv_cache: layer  15: dev = CPU
llama_kv_cache: layer  16: dev = CPU
llama_kv_cache: layer  17: filtered
llama_kv_cache: layer  18: dev = CPU
llama_kv_cache: layer  19: dev = CPU
llama_kv_cache: layer  20: dev = CPU
llama_kv_cache: layer  21: dev = CPU
llama_kv_cache: layer  22: dev = CPU
llama_kv_cache: layer  23: filtered
llama_kv_cache: layer  24: dev = CPU
llama_kv_cache: layer  25: dev = CPU
llama_kv_cache: layer  26: dev = CPU
llama_kv_cache: layer  27: dev = CPU
llama_kv_cache: layer  28: dev = CPU
llama_kv_cache: layer  29: filtered
llama_kv_cache: layer  30: dev = CPU
llama_kv_cache: layer  31: dev = CPU
llama_kv_cache: layer  32: dev = CPU
llama_kv_cache: layer  33: dev = CPU
llama_kv_cache: layer  34: dev = CPU
llama_kv_cache: layer  35: filtered
llama_kv_cache: layer  36: dev = CPU
llama_kv_cache: layer  37: dev = CPU
llama_kv_cache: layer  38: dev = CPU
llama_kv_cache: layer  39: dev = CPU
llama_kv_cache: layer  40: dev = CPU
llama_kv_cache: layer  41: filtered
llama_kv_cache: layer  42: dev = CPU
llama_kv_cache: layer  43: dev = CPU
llama_kv_cache: layer  44: dev = CPU
llama_kv_cache: layer  45: dev = CPU
llama_kv_cache: layer  46: dev = CPU
llama_kv_cache: layer  47: filtered
llama_kv_cache: reusing layers:
llama_kv_cache: - layer   0: no reuse
llama_kv_cache: - layer   1: no reuse
llama_kv_cache: - layer   2: no reuse
llama_kv_cache: - layer   3: no reuse
llama_kv_cache: - layer   4: no reuse
llama_kv_cache: - layer   5: no reuse
llama_kv_cache: - layer   6: no reuse
llama_kv_cache: - layer   7: no reuse
llama_kv_cache: - layer   8: no reuse
llama_kv_cache: - layer   9: no reuse
llama_kv_cache: - layer  10: no reuse
llama_kv_cache: - layer  11: no reuse
llama_kv_cache: - layer  12: no reuse
llama_kv_cache: - layer  13: no reuse
llama_kv_cache: - layer  14: no reuse
llama_kv_cache: - layer  15: no reuse
llama_kv_cache: - layer  16: no reuse
llama_kv_cache: - layer  17: no reuse
llama_kv_cache: - layer  18: no reuse
llama_kv_cache: - layer  19: no reuse
llama_kv_cache: - layer  20: no reuse
llama_kv_cache: - layer  21: no reuse
llama_kv_cache: - layer  22: no reuse
llama_kv_cache: - layer  23: no reuse
llama_kv_cache: - layer  24: no reuse
llama_kv_cache: - layer  25: no reuse
llama_kv_cache: - layer  26: no reuse
llama_kv_cache: - layer  27: no reuse
llama_kv_cache: - layer  28: no reuse
llama_kv_cache: - layer  29: no reuse
llama_kv_cache: - layer  30: no reuse
llama_kv_cache: - layer  31: no reuse
llama_kv_cache: - layer  32: no reuse
llama_kv_cache: - layer  33: no reuse
llama_kv_cache: - layer  34: no reuse
llama_kv_cache: - layer  35: no reuse
llama_kv_cache: - layer  36: no reuse
llama_kv_cache: - layer  37: no reuse
llama_kv_cache: - layer  38: no reuse
llama_kv_cache: - layer  39: no reuse
llama_kv_cache: - layer  40: no reuse
llama_kv_cache: - layer  41: no reuse
llama_kv_cache: - layer  42: no reuse
llama_kv_cache: - layer  43: no reuse
llama_kv_cache: - layer  44: no reuse
llama_kv_cache: - layer  45: no reuse
llama_kv_cache: - layer  46: no reuse
llama_kv_cache: - layer  47: no reuse
llama_kv_cache:        CPU KV buffer size =   640.00 MiB
llama_kv_cache: size =  640.00 MiB (  2048 cells,  40 layers,  1/1 seqs), K (f16):  320.00 MiB, V (f16):  320.00 MiB
llama_kv_cache: attn_rot_k = 0, n_embd_head_k_all = 256
llama_kv_cache: attn_rot_v = 0, n_embd_head_k_all = 256
llama_context: enumerating backends
llama_context: backend_ptrs.size() = 1
sched_reserve: reserving ...
sched_reserve: max_nodes = 5344
sched_reserve: reserving full memory module
sched_reserve: worst-case: n_tokens = 512, n_seqs = 1, n_outputs = 1
graph_reserve: reserving a graph for ubatch with n_tokens =    1, n_seqs =  1, n_outputs =    1
sched_reserve: Flash Attention was auto, set to enabled
sched_reserve: resolving fused Gated Delta Net support:
graph_reserve: reserving a graph for ubatch with n_tokens =    1, n_seqs =  1, n_outputs =    1
sched_reserve: fused Gated Delta Net (autoregressive) enabled
graph_reserve: reserving a graph for ubatch with n_tokens =   16, n_seqs =  1, n_outputs =   16
sched_reserve: fused Gated Delta Net (chunked) enabled
graph_reserve: reserving a graph for ubatch with n_tokens =  512, n_seqs =  1, n_outputs =  512
graph_reserve: reserving a graph for ubatch with n_tokens =    1, n_seqs =  1, n_outputs =    1
graph_reserve: reserving a graph for ubatch with n_tokens =  512, n_seqs =  1, n_outputs =  512
sched_reserve:        CPU compute buffer size =   527.00 MiB
sched_reserve: graph nodes  = 1972
sched_reserve: graph splits = 1
sched_reserve: reserve took 4.17 ms, sched copies = 1

[main]: n_ctx = 2048, n_threads = 8, gpu_layers = 999
[main]: flash-attn = auto (runtime-selected)
[main]: system_info: CPU : SSE3 = 1 | SSSE3 = 1 | AVX = 1 | AVX2 = 1 | F16C = 1 | FMA = 1 | AVX512 = 1 | LLAMAFILE = 1 | OPENMP = 1 | REPACK = 1 |

> Running with custom prompt => [1/18]: [This is Izzy's bedroom]
  prefill: 17100.2ms (555 tokens, 32.5 t/s)
 {
    "answer": "c. That is correct Mara. It is your sister's room",
    "justification": "Mara finally revealed the truth, so option (c) confirms her answer."
}
  [all token ids (46): 642 107 140 236775 14433 1083 623 236755 236761 2981 563 4338 63338 236761 1030 563 822 12198 236789 236751 2978 827 107 140 236775 4017 2540 1083 623 201883 6861 9937 506 9043 236764 834 5086 568 236755 236768 44898 1116 3890 1781 107 236783]
  [output hex (162 bytes):]
    0000: 20 7b 0a 20 20 20 20 22 61 6e 73 77 65 72 22 3a  | {.    "answer":|
    0010: 20 22 63 2e 20 54 68 61 74 20 69 73 20 63 6f 72  | "c. That is cor|
    0020: 72 65 63 74 20 4d 61 72 61 2e 20 49 74 20 69 73  |rect Mara. It is|
    0030: 20 79 6f 75 72 20 73 69 73 74 65 72 27 73 20 72  | your sister's r|
    0040: 6f 6f 6d 22 2c 0a 20 20 20 20 22 6a 75 73 74 69  |oom",.    "justi|
    0050: 66 69 63 61 74 69 6f 6e 22 3a 20 22 4d 61 72 61  |fication": "Mara|
    0060: 20 66 69 6e 61 6c 6c 79 20 72 65 76 65 61 6c 65  | finally reveale|
    0070: 64 20 74 68 65 20 74 72 75 74 68 2c 20 73 6f 20  |d the truth, so |
    0080: 6f 70 74 69 6f 6e 20 28 63 29 20 63 6f 6e 66 69  |option (c) confi|
    0090: 72 6d 73 20 68 65 72 20 61 6e 73 77 65 72 2e 22  |rms her answer."|
    00a0: 0a 7d                                            |.}|
  decode: 7355.2ms (45 tokens, 6.12 t/s)
  decode (no-TTFT): 7354.2ms (46 tokens, 6.25 t/s) [TTFT 0.9ms]

> Running with custom prompt => [2/18]: [This is bedroom for Izzy]
^C
C:\llama.cpp\b612_052026_mtp\build>bin\RelWithDebInfo\minslm-cli.exe C:\llama.cpp\models\Gemma-4\12B\Unsloth\gemma-4-12b-it-Q4_K_M_Unsloth.gguf 8 ..\examples\llm-infer\prompts\custom_prompts_gemma4.txt --mtp-head c:\llama.cpp\models\Gemma-4\12B\gemma-4-12B-it-Q4_K_M-MTP-assistant.gguf  --spec-type mtp --draft-block-size 3 v2
Warning: unknown argument '--spec-type' ignored
Warning: unknown argument 'mtp' ignored
[main]: loaded 18 prompt(s) from '..\examples\llm-infer\prompts\custom_prompts_gemma4.txt'
llama_model_loader: direct I/O is enabled, disabling mmap
llama_model_loader: loaded meta data with 58 key-value pairs and 667 tensors from C:\llama.cpp\models\Gemma-4\12B\Unsloth\gemma-4-12b-it-Q4_K_M_Unsloth.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = gemma4
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                     general.sampling.top_k i32              = 64
llama_model_loader: - kv   3:                     general.sampling.top_p f32              = 0.950000
llama_model_loader: - kv   4:                      general.sampling.temp f32              = 1.000000
llama_model_loader: - kv   5:                               general.name str              = Gemma-4-12B-It
llama_model_loader: - kv   6:                           general.finetune str              = it
llama_model_loader: - kv   7:                           general.basename str              = Gemma-4-12B-It
llama_model_loader: - kv   8:                       general.quantized_by str              = Unsloth
llama_model_loader: - kv   9:                         general.size_label str              = 12B
llama_model_loader: - kv  10:                            general.license str              = apache-2.0
llama_model_loader: - kv  11:                       general.license.link str              = https://ai.google.dev/gemma/docs/gemm...
llama_model_loader: - kv  12:                           general.repo_url str              = https://huggingface.co/unsloth
llama_model_loader: - kv  13:                   general.base_model.count u32              = 1
llama_model_loader: - kv  14:                  general.base_model.0.name str              = Gemma 4 12b It
llama_model_loader: - kv  15:          general.base_model.0.organization str              = Google
llama_model_loader: - kv  16:              general.base_model.0.repo_url str              = https://huggingface.co/google/gemma-4...
llama_model_loader: - kv  17:                               general.tags arr[str,2]       = ["unsloth", "any-to-any"]
llama_model_loader: - kv  18:                         gemma4.block_count u32              = 48
llama_model_loader: - kv  19:                      gemma4.context_length u32              = 262144
llama_model_loader: - kv  20:                    gemma4.embedding_length u32              = 3840
llama_model_loader: - kv  21:                 gemma4.feed_forward_length u32              = 15360
llama_model_loader: - kv  22:                gemma4.attention.head_count u32              = 16
llama_model_loader: - kv  23:             gemma4.attention.head_count_kv arr[i32,48]      = [8, 8, 8, 8, 8, 1, 8, 8, 8, 8, 8, 1, ...
llama_model_loader: - kv  24:                      gemma4.rope.freq_base f32              = 1000000.000000
llama_model_loader: - kv  25:                  gemma4.rope.freq_base_swa f32              = 10000.000000
llama_model_loader: - kv  26:    gemma4.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  27:                gemma4.attention.key_length u32              = 512
llama_model_loader: - kv  28:              gemma4.attention.value_length u32              = 512
llama_model_loader: - kv  29:             gemma4.final_logit_softcapping f32              = 30.000000
llama_model_loader: - kv  30:            gemma4.attention.sliding_window u32              = 1024
llama_model_loader: - kv  31:          gemma4.attention.shared_kv_layers u32              = 0
llama_model_loader: - kv  32:    gemma4.embedding_length_per_layer_input u32              = 0
llama_model_loader: - kv  33:    gemma4.attention.sliding_window_pattern arr[bool,48]     = [true, true, true, true, true, false,...
llama_model_loader: - kv  34:            gemma4.attention.key_length_swa u32              = 256
llama_model_loader: - kv  35:          gemma4.attention.value_length_swa u32              = 256
llama_model_loader: - kv  36:                gemma4.rope.dimension_count u32              = 512
llama_model_loader: - kv  37:            gemma4.rope.dimension_count_swa u32              = 256
llama_model_loader: - kv  38:             tokenizer.ggml.suppress_tokens arr[i32,2]       = [258883, 258882]
llama_model_loader: - kv  39:                       tokenizer.ggml.model str              = gemma4
llama_model_loader: - kv  40:                      tokenizer.ggml.tokens arr[str,262144]  = ["<pad>", "<eos>", "<bos>", "<unk>", ...
llama_model_loader: - kv  41:                      tokenizer.ggml.scores arr[f32,262144]  = [-1000.000000, -1000.000000, -1000.00...
llama_model_loader: - kv  42:                  tokenizer.ggml.token_type arr[i32,262144]  = [3, 1, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  43:                      tokenizer.ggml.merges arr[str,514906]  = ["\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n \n", ...
llama_model_loader: - kv  44:                tokenizer.ggml.bos_token_id u32              = 2
llama_model_loader: - kv  45:                tokenizer.ggml.eos_token_id u32              = 106
llama_model_loader: - kv  46:            tokenizer.ggml.unknown_token_id u32              = 3
llama_model_loader: - kv  47:            tokenizer.ggml.padding_token_id u32              = 0
llama_model_loader: - kv  48:               tokenizer.ggml.mask_token_id u32              = 4
llama_model_loader: - kv  49:                    tokenizer.chat_template str              = {%- macro format_parameters(propertie...
llama_model_loader: - kv  50:            tokenizer.ggml.add_space_prefix bool             = false
llama_model_loader: - kv  51:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  52:               general.quantization_version u32              = 2
llama_model_loader: - kv  53:                          general.file_type u32              = 15
llama_model_loader: - kv  54:                      quantize.imatrix.file str              = gemma-4-12b-it-GGUF/imatrix_unsloth.gguf
llama_model_loader: - kv  55:                   quantize.imatrix.dataset str              = unsloth_calibration_gemma-4-12b-it.txt
llama_model_loader: - kv  56:             quantize.imatrix.entries_count u32              = 328
llama_model_loader: - kv  57:              quantize.imatrix.chunks_count u32              = 141
llama_model_loader: - type  f32:  338 tensors
llama_model_loader: - type q4_K:  285 tensors
llama_model_loader: - type q6_K:   44 tensors
print_info: file format = GGUF V3 (latest)
print_info: file type   = Q4_K - Medium
print_info: file size   = 6.62 GiB (4.77 BPW)
init_tokenizer: initializing tokenizer for type 2
load: 0 unused tokens
load: control-looking token:      1 '<eos>' was not control-type; this is probably a bug in the model. its type will be overridden
load: control token:      0 '<pad>' is not marked as EOG
load: control token:      3 '<unk>' is not marked as EOG
load: control token:      2 '<bos>' is not marked as EOG
load: control token:      4 '<mask>' is not marked as EOG
load: control token:     47 '<tool|>' is not marked as EOG
load: control token:     98 '<|think|>' is not marked as EOG
load: control token:     46 '<|tool>' is not marked as EOG
load: control-looking token:     50 '<|tool_response>' was not control-type; this is probably a bug in the model. its type will be overridden
load: control token:    105 '<|turn>' is not marked as EOG
load: control-looking token:    212 '</s>' was not control-type; this is probably a bug in the model. its type will be overridden
load: control token: 258880 '<|image|>' is not marked as EOG
load: control token: 258882 '<image|>' is not marked as EOG
load: control token: 256000 '<|audio>' is not marked as EOG
load: control token: 258883 '<audio|>' is not marked as EOG
load: control token: 258884 '<|video|>' is not marked as EOG
load: control token: 255999 '<|image>' is not marked as EOG
load: control token: 258881 '<|audio|>' is not marked as EOG
load: printing all EOG tokens:
load:   - 1 ('<eos>')
load:   - 50 ('<|tool_response>')
load:   - 106 ('<turn|>')
load:   - 212 ('</s>')
load: special_eog_ids contains '<|tool_response>', removing '</s>' token from EOG list
load: special tokens cache size = 24
load: token to piece cache size = 1.9445 MB
print_info: arch                  = gemma4
print_info: vocab_only            = 0
print_info: no_alloc              = 0
print_info: n_ctx_train           = 262144
print_info: n_embd                = 3840
print_info: n_embd_inp            = 3840
print_info: n_layer               = 48
print_info: n_head                = 16
print_info: n_head_kv             = [8, 8, 8, 8, 8, 1, 8, 8, 8, 8, 8, 1, 8, 8, 8, 8, 8, 1, 8, 8, 8, 8, 8, 1, 8, 8, 8, 8, 8, 1, 8, 8, 8, 8, 8, 1, 8, 8, 8, 8, 8, 1, 8, 8, 8, 8, 8, 1]
print_info: n_rot                 = 512
print_info: n_swa                 = 1024
print_info: is_swa_any            = 1
print_info: n_embd_head_k         = 512
print_info: n_embd_head_v         = 512
print_info: n_gqa                 = [2, 2, 2, 2, 2, 16, 2, 2, 2, 2, 2, 16, 2, 2, 2, 2, 2, 16, 2, 2, 2, 2, 2, 16, 2, 2, 2, 2, 2, 16, 2, 2, 2, 2, 2, 16, 2, 2, 2, 2, 2, 16, 2, 2, 2, 2, 2, 16]
print_info: n_embd_k_gqa          = [2048, 2048, 2048, 2048, 2048, 512, 2048, 2048, 2048, 2048, 2048, 512, 2048, 2048, 2048, 2048, 2048, 512, 2048, 2048, 2048, 2048, 2048, 512, 2048, 2048, 2048, 2048, 2048, 512, 2048, 2048, 2048, 2048, 2048, 512, 2048, 2048, 2048, 2048, 2048, 512, 2048, 2048, 2048, 2048, 2048, 512]
print_info: n_embd_v_gqa          = [2048, 2048, 2048, 2048, 2048, 512, 2048, 2048, 2048, 2048, 2048, 512, 2048, 2048, 2048, 2048, 2048, 512, 2048, 2048, 2048, 2048, 2048, 512, 2048, 2048, 2048, 2048, 2048, 512, 2048, 2048, 2048, 2048, 2048, 512, 2048, 2048, 2048, 2048, 2048, 512, 2048, 2048, 2048, 2048, 2048, 512]
print_info: f_norm_eps            = 0.0e+00
print_info: f_norm_rms_eps        = 1.0e-06
print_info: f_clamp_kqv           = 0.0e+00
print_info: f_max_alibi_bias      = 0.0e+00
print_info: f_logit_scale         = 0.0e+00
print_info: f_attn_scale          = 1.0e+00
print_info: f_attn_value_scale    = 0.0000
print_info: n_ff                  = 15360
print_info: n_expert              = 0
print_info: n_expert_used         = 0
print_info: n_expert_groups       = 0
print_info: n_group_used          = 0
print_info: causal attn           = 1
print_info: pooling type          = -1
print_info: rope type             = 2
print_info: rope scaling          = linear
print_info: freq_base_train       = 1000000.0
print_info: freq_scale_train      = 1
print_info: freq_base_swa         = 10000.0
print_info: freq_scale_swa        = 1
print_info: n_embd_head_k_swa     = 256
print_info: n_embd_head_v_swa     = 256
print_info: n_rot_swa             = 256
print_info: n_ctx_orig_yarn       = 262144
print_info: rope_yarn_log_mul     = 0.0000
print_info: rope_finetuned        = unknown
print_info: model type            = ?B
print_info: model params          = 11.91 B
print_info: general.name          = Gemma-4-12B-It
print_info: vocab type            = BPE
print_info: n_vocab               = 262144
print_info: n_merges              = 514906
print_info: BOS token             = 2 '<bos>'
print_info: EOS token             = 106 '<turn|>'
print_info: UNK token             = 3 '<unk>'
print_info: PAD token             = 0 '<pad>'
print_info: MASK token            = 4 '<mask>'
print_info: LF token              = 107 '
'
print_info: EOG token             = 1 '<eos>'
print_info: EOG token             = 50 '<|tool_response>'
print_info: EOG token             = 106 '<turn|>'
print_info: max token length      = 93
load_tensors: loading model tensors, this can take a while... (mmap = false, direct_io = true)
load_tensors: layer   0 assigned to device CPU, is_swa = 1
load_tensors: layer   1 assigned to device CPU, is_swa = 1
load_tensors: layer   2 assigned to device CPU, is_swa = 1
load_tensors: layer   3 assigned to device CPU, is_swa = 1
load_tensors: layer   4 assigned to device CPU, is_swa = 1
load_tensors: layer   5 assigned to device CPU, is_swa = 0
load_tensors: layer   6 assigned to device CPU, is_swa = 1
load_tensors: layer   7 assigned to device CPU, is_swa = 1
load_tensors: layer   8 assigned to device CPU, is_swa = 1
load_tensors: layer   9 assigned to device CPU, is_swa = 1
load_tensors: layer  10 assigned to device CPU, is_swa = 1
load_tensors: layer  11 assigned to device CPU, is_swa = 0
load_tensors: layer  12 assigned to device CPU, is_swa = 1
load_tensors: layer  13 assigned to device CPU, is_swa = 1
load_tensors: layer  14 assigned to device CPU, is_swa = 1
load_tensors: layer  15 assigned to device CPU, is_swa = 1
load_tensors: layer  16 assigned to device CPU, is_swa = 1
load_tensors: layer  17 assigned to device CPU, is_swa = 0
load_tensors: layer  18 assigned to device CPU, is_swa = 1
load_tensors: layer  19 assigned to device CPU, is_swa = 1
load_tensors: layer  20 assigned to device CPU, is_swa = 1
load_tensors: layer  21 assigned to device CPU, is_swa = 1
load_tensors: layer  22 assigned to device CPU, is_swa = 1
load_tensors: layer  23 assigned to device CPU, is_swa = 0
load_tensors: layer  24 assigned to device CPU, is_swa = 1
load_tensors: layer  25 assigned to device CPU, is_swa = 1
load_tensors: layer  26 assigned to device CPU, is_swa = 1
load_tensors: layer  27 assigned to device CPU, is_swa = 1
load_tensors: layer  28 assigned to device CPU, is_swa = 1
load_tensors: layer  29 assigned to device CPU, is_swa = 0
load_tensors: layer  30 assigned to device CPU, is_swa = 1
load_tensors: layer  31 assigned to device CPU, is_swa = 1
load_tensors: layer  32 assigned to device CPU, is_swa = 1
load_tensors: layer  33 assigned to device CPU, is_swa = 1
load_tensors: layer  34 assigned to device CPU, is_swa = 1
load_tensors: layer  35 assigned to device CPU, is_swa = 0
load_tensors: layer  36 assigned to device CPU, is_swa = 1
load_tensors: layer  37 assigned to device CPU, is_swa = 1
load_tensors: layer  38 assigned to device CPU, is_swa = 1
load_tensors: layer  39 assigned to device CPU, is_swa = 1
load_tensors: layer  40 assigned to device CPU, is_swa = 1
load_tensors: layer  41 assigned to device CPU, is_swa = 0
load_tensors: layer  42 assigned to device CPU, is_swa = 1
load_tensors: layer  43 assigned to device CPU, is_swa = 1
load_tensors: layer  44 assigned to device CPU, is_swa = 1
load_tensors: layer  45 assigned to device CPU, is_swa = 1
load_tensors: layer  46 assigned to device CPU, is_swa = 1
load_tensors: layer  47 assigned to device CPU, is_swa = 0
load_tensors: layer  48 assigned to device CPU, is_swa = 0
create_tensor: loading tensor token_embd.weight
create_tensor: loading tensor token_embd.weight
create_tensor: loading tensor output_norm.weight
create_tensor: loading tensor blk.0.attn_norm.weight
create_tensor: loading tensor blk.0.attn_q.weight
create_tensor: loading tensor blk.0.attn_k.weight
create_tensor: loading tensor blk.0.attn_v.weight
create_tensor: loading tensor blk.0.attn_output.weight
create_tensor: loading tensor blk.0.attn_q_norm.weight
create_tensor: loading tensor blk.0.attn_k_norm.weight
create_tensor: loading tensor blk.0.post_attention_norm.weight
create_tensor: loading tensor blk.0.layer_output_scale.weight
create_tensor: loading tensor blk.0.ffn_norm.weight
create_tensor: loading tensor blk.0.ffn_gate.weight
create_tensor: loading tensor blk.0.ffn_up.weight
create_tensor: loading tensor blk.0.ffn_down.weight
create_tensor: loading tensor blk.0.post_ffw_norm.weight
create_tensor: loading tensor blk.1.attn_norm.weight
create_tensor: loading tensor blk.1.attn_q.weight
create_tensor: loading tensor blk.1.attn_k.weight
create_tensor: loading tensor blk.1.attn_v.weight
create_tensor: loading tensor blk.1.attn_output.weight
create_tensor: loading tensor blk.1.attn_q_norm.weight
create_tensor: loading tensor blk.1.attn_k_norm.weight
create_tensor: loading tensor blk.1.post_attention_norm.weight
create_tensor: loading tensor blk.1.layer_output_scale.weight
create_tensor: loading tensor blk.1.ffn_norm.weight
create_tensor: loading tensor blk.1.ffn_gate.weight
create_tensor: loading tensor blk.1.ffn_up.weight
create_tensor: loading tensor blk.1.ffn_down.weight
create_tensor: loading tensor blk.1.post_ffw_norm.weight
create_tensor: loading tensor blk.2.attn_norm.weight
create_tensor: loading tensor blk.2.attn_q.weight
create_tensor: loading tensor blk.2.attn_k.weight
create_tensor: loading tensor blk.2.attn_v.weight
create_tensor: loading tensor blk.2.attn_output.weight
create_tensor: loading tensor blk.2.attn_q_norm.weight
create_tensor: loading tensor blk.2.attn_k_norm.weight
create_tensor: loading tensor blk.2.post_attention_norm.weight
create_tensor: loading tensor blk.2.layer_output_scale.weight
create_tensor: loading tensor blk.2.ffn_norm.weight
create_tensor: loading tensor blk.2.ffn_gate.weight
create_tensor: loading tensor blk.2.ffn_up.weight
create_tensor: loading tensor blk.2.ffn_down.weight
create_tensor: loading tensor blk.2.post_ffw_norm.weight
create_tensor: loading tensor blk.3.attn_norm.weight
create_tensor: loading tensor blk.3.attn_q.weight
create_tensor: loading tensor blk.3.attn_k.weight
create_tensor: loading tensor blk.3.attn_v.weight
create_tensor: loading tensor blk.3.attn_output.weight
create_tensor: loading tensor blk.3.attn_q_norm.weight
create_tensor: loading tensor blk.3.attn_k_norm.weight
create_tensor: loading tensor blk.3.post_attention_norm.weight
create_tensor: loading tensor blk.3.layer_output_scale.weight
create_tensor: loading tensor blk.3.ffn_norm.weight
create_tensor: loading tensor blk.3.ffn_gate.weight
create_tensor: loading tensor blk.3.ffn_up.weight
create_tensor: loading tensor blk.3.ffn_down.weight
create_tensor: loading tensor blk.3.post_ffw_norm.weight
create_tensor: loading tensor blk.4.attn_norm.weight
create_tensor: loading tensor blk.4.attn_q.weight
create_tensor: loading tensor blk.4.attn_k.weight
create_tensor: loading tensor blk.4.attn_v.weight
create_tensor: loading tensor blk.4.attn_output.weight
create_tensor: loading tensor blk.4.attn_q_norm.weight
create_tensor: loading tensor blk.4.attn_k_norm.weight
create_tensor: loading tensor blk.4.post_attention_norm.weight
create_tensor: loading tensor blk.4.layer_output_scale.weight
create_tensor: loading tensor blk.4.ffn_norm.weight
create_tensor: loading tensor blk.4.ffn_gate.weight
create_tensor: loading tensor blk.4.ffn_up.weight
create_tensor: loading tensor blk.4.ffn_down.weight
create_tensor: loading tensor blk.4.post_ffw_norm.weight
create_tensor: loading tensor blk.5.attn_norm.weight
create_tensor: loading tensor blk.5.attn_q.weight
create_tensor: loading tensor blk.5.attn_k.weight
create_tensor: loading tensor blk.5.attn_output.weight
create_tensor: loading tensor blk.5.attn_q_norm.weight
create_tensor: loading tensor blk.5.attn_k_norm.weight
create_tensor: loading tensor blk.5.post_attention_norm.weight
create_tensor: loading tensor blk.5.layer_output_scale.weight
create_tensor: loading tensor rope_freqs.weight
create_tensor: loading tensor blk.5.ffn_norm.weight
create_tensor: loading tensor blk.5.ffn_gate.weight
create_tensor: loading tensor blk.5.ffn_up.weight
create_tensor: loading tensor blk.5.ffn_down.weight
create_tensor: loading tensor blk.5.post_ffw_norm.weight
create_tensor: loading tensor blk.6.attn_norm.weight
create_tensor: loading tensor blk.6.attn_q.weight
create_tensor: loading tensor blk.6.attn_k.weight
create_tensor: loading tensor blk.6.attn_v.weight
create_tensor: loading tensor blk.6.attn_output.weight
create_tensor: loading tensor blk.6.attn_q_norm.weight
create_tensor: loading tensor blk.6.attn_k_norm.weight
create_tensor: loading tensor blk.6.post_attention_norm.weight
create_tensor: loading tensor blk.6.layer_output_scale.weight
create_tensor: loading tensor blk.6.ffn_norm.weight
create_tensor: loading tensor blk.6.ffn_gate.weight
create_tensor: loading tensor blk.6.ffn_up.weight
create_tensor: loading tensor blk.6.ffn_down.weight
create_tensor: loading tensor blk.6.post_ffw_norm.weight
create_tensor: loading tensor blk.7.attn_norm.weight
create_tensor: loading tensor blk.7.attn_q.weight
create_tensor: loading tensor blk.7.attn_k.weight
create_tensor: loading tensor blk.7.attn_v.weight
create_tensor: loading tensor blk.7.attn_output.weight
create_tensor: loading tensor blk.7.attn_q_norm.weight
create_tensor: loading tensor blk.7.attn_k_norm.weight
create_tensor: loading tensor blk.7.post_attention_norm.weight
create_tensor: loading tensor blk.7.layer_output_scale.weight
create_tensor: loading tensor blk.7.ffn_norm.weight
create_tensor: loading tensor blk.7.ffn_gate.weight
create_tensor: loading tensor blk.7.ffn_up.weight
create_tensor: loading tensor blk.7.ffn_down.weight
create_tensor: loading tensor blk.7.post_ffw_norm.weight
create_tensor: loading tensor blk.8.attn_norm.weight
create_tensor: loading tensor blk.8.attn_q.weight
create_tensor: loading tensor blk.8.attn_k.weight
create_tensor: loading tensor blk.8.attn_v.weight
create_tensor: loading tensor blk.8.attn_output.weight
create_tensor: loading tensor blk.8.attn_q_norm.weight
create_tensor: loading tensor blk.8.attn_k_norm.weight
create_tensor: loading tensor blk.8.post_attention_norm.weight
create_tensor: loading tensor blk.8.layer_output_scale.weight
create_tensor: loading tensor blk.8.ffn_norm.weight
create_tensor: loading tensor blk.8.ffn_gate.weight
create_tensor: loading tensor blk.8.ffn_up.weight
create_tensor: loading tensor blk.8.ffn_down.weight
create_tensor: loading tensor blk.8.post_ffw_norm.weight
create_tensor: loading tensor blk.9.attn_norm.weight
create_tensor: loading tensor blk.9.attn_q.weight
create_tensor: loading tensor blk.9.attn_k.weight
create_tensor: loading tensor blk.9.attn_v.weight
create_tensor: loading tensor blk.9.attn_output.weight
create_tensor: loading tensor blk.9.attn_q_norm.weight
create_tensor: loading tensor blk.9.attn_k_norm.weight
create_tensor: loading tensor blk.9.post_attention_norm.weight
create_tensor: loading tensor blk.9.layer_output_scale.weight
create_tensor: loading tensor blk.9.ffn_norm.weight
create_tensor: loading tensor blk.9.ffn_gate.weight
create_tensor: loading tensor blk.9.ffn_up.weight
create_tensor: loading tensor blk.9.ffn_down.weight
create_tensor: loading tensor blk.9.post_ffw_norm.weight
create_tensor: loading tensor blk.10.attn_norm.weight
create_tensor: loading tensor blk.10.attn_q.weight
create_tensor: loading tensor blk.10.attn_k.weight
create_tensor: loading tensor blk.10.attn_v.weight
create_tensor: loading tensor blk.10.attn_output.weight
create_tensor: loading tensor blk.10.attn_q_norm.weight
create_tensor: loading tensor blk.10.attn_k_norm.weight
create_tensor: loading tensor blk.10.post_attention_norm.weight
create_tensor: loading tensor blk.10.layer_output_scale.weight
create_tensor: loading tensor blk.10.ffn_norm.weight
create_tensor: loading tensor blk.10.ffn_gate.weight
create_tensor: loading tensor blk.10.ffn_up.weight
create_tensor: loading tensor blk.10.ffn_down.weight
create_tensor: loading tensor blk.10.post_ffw_norm.weight
create_tensor: loading tensor blk.11.attn_norm.weight
create_tensor: loading tensor blk.11.attn_q.weight
create_tensor: loading tensor blk.11.attn_k.weight
create_tensor: loading tensor blk.11.attn_output.weight
create_tensor: loading tensor blk.11.attn_q_norm.weight
create_tensor: loading tensor blk.11.attn_k_norm.weight
create_tensor: loading tensor blk.11.post_attention_norm.weight
create_tensor: loading tensor blk.11.layer_output_scale.weight
create_tensor: loading tensor blk.11.ffn_norm.weight
create_tensor: loading tensor blk.11.ffn_gate.weight
create_tensor: loading tensor blk.11.ffn_up.weight
create_tensor: loading tensor blk.11.ffn_down.weight
create_tensor: loading tensor blk.11.post_ffw_norm.weight
create_tensor: loading tensor blk.12.attn_norm.weight
create_tensor: loading tensor blk.12.attn_q.weight
create_tensor: loading tensor blk.12.attn_k.weight
create_tensor: loading tensor blk.12.attn_v.weight
create_tensor: loading tensor blk.12.attn_output.weight
create_tensor: loading tensor blk.12.attn_q_norm.weight
create_tensor: loading tensor blk.12.attn_k_norm.weight
create_tensor: loading tensor blk.12.post_attention_norm.weight
create_tensor: loading tensor blk.12.layer_output_scale.weight
create_tensor: loading tensor blk.12.ffn_norm.weight
create_tensor: loading tensor blk.12.ffn_gate.weight
create_tensor: loading tensor blk.12.ffn_up.weight
create_tensor: loading tensor blk.12.ffn_down.weight
create_tensor: loading tensor blk.12.post_ffw_norm.weight
create_tensor: loading tensor blk.13.attn_norm.weight
create_tensor: loading tensor blk.13.attn_q.weight
create_tensor: loading tensor blk.13.attn_k.weight
create_tensor: loading tensor blk.13.attn_v.weight
create_tensor: loading tensor blk.13.attn_output.weight
create_tensor: loading tensor blk.13.attn_q_norm.weight
create_tensor: loading tensor blk.13.attn_k_norm.weight
create_tensor: loading tensor blk.13.post_attention_norm.weight
create_tensor: loading tensor blk.13.layer_output_scale.weight
create_tensor: loading tensor blk.13.ffn_norm.weight
create_tensor: loading tensor blk.13.ffn_gate.weight
create_tensor: loading tensor blk.13.ffn_up.weight
create_tensor: loading tensor blk.13.ffn_down.weight
create_tensor: loading tensor blk.13.post_ffw_norm.weight
create_tensor: loading tensor blk.14.attn_norm.weight
create_tensor: loading tensor blk.14.attn_q.weight
create_tensor: loading tensor blk.14.attn_k.weight
create_tensor: loading tensor blk.14.attn_v.weight
create_tensor: loading tensor blk.14.attn_output.weight
create_tensor: loading tensor blk.14.attn_q_norm.weight
create_tensor: loading tensor blk.14.attn_k_norm.weight
create_tensor: loading tensor blk.14.post_attention_norm.weight
create_tensor: loading tensor blk.14.layer_output_scale.weight
create_tensor: loading tensor blk.14.ffn_norm.weight
create_tensor: loading tensor blk.14.ffn_gate.weight
create_tensor: loading tensor blk.14.ffn_up.weight
create_tensor: loading tensor blk.14.ffn_down.weight
create_tensor: loading tensor blk.14.post_ffw_norm.weight
create_tensor: loading tensor blk.15.attn_norm.weight
create_tensor: loading tensor blk.15.attn_q.weight
create_tensor: loading tensor blk.15.attn_k.weight
create_tensor: loading tensor blk.15.attn_v.weight
create_tensor: loading tensor blk.15.attn_output.weight
create_tensor: loading tensor blk.15.attn_q_norm.weight
create_tensor: loading tensor blk.15.attn_k_norm.weight
create_tensor: loading tensor blk.15.post_attention_norm.weight
create_tensor: loading tensor blk.15.layer_output_scale.weight
create_tensor: loading tensor blk.15.ffn_norm.weight
create_tensor: loading tensor blk.15.ffn_gate.weight
create_tensor: loading tensor blk.15.ffn_up.weight
create_tensor: loading tensor blk.15.ffn_down.weight
create_tensor: loading tensor blk.15.post_ffw_norm.weight
create_tensor: loading tensor blk.16.attn_norm.weight
create_tensor: loading tensor blk.16.attn_q.weight
create_tensor: loading tensor blk.16.attn_k.weight
create_tensor: loading tensor blk.16.attn_v.weight
create_tensor: loading tensor blk.16.attn_output.weight
create_tensor: loading tensor blk.16.attn_q_norm.weight
create_tensor: loading tensor blk.16.attn_k_norm.weight
create_tensor: loading tensor blk.16.post_attention_norm.weight
create_tensor: loading tensor blk.16.layer_output_scale.weight
create_tensor: loading tensor blk.16.ffn_norm.weight
create_tensor: loading tensor blk.16.ffn_gate.weight
create_tensor: loading tensor blk.16.ffn_up.weight
create_tensor: loading tensor blk.16.ffn_down.weight
create_tensor: loading tensor blk.16.post_ffw_norm.weight
create_tensor: loading tensor blk.17.attn_norm.weight
create_tensor: loading tensor blk.17.attn_q.weight
create_tensor: loading tensor blk.17.attn_k.weight
create_tensor: loading tensor blk.17.attn_output.weight
create_tensor: loading tensor blk.17.attn_q_norm.weight
create_tensor: loading tensor blk.17.attn_k_norm.weight
create_tensor: loading tensor blk.17.post_attention_norm.weight
create_tensor: loading tensor blk.17.layer_output_scale.weight
create_tensor: loading tensor blk.17.ffn_norm.weight
create_tensor: loading tensor blk.17.ffn_gate.weight
create_tensor: loading tensor blk.17.ffn_up.weight
create_tensor: loading tensor blk.17.ffn_down.weight
create_tensor: loading tensor blk.17.post_ffw_norm.weight
create_tensor: loading tensor blk.18.attn_norm.weight
create_tensor: loading tensor blk.18.attn_q.weight
create_tensor: loading tensor blk.18.attn_k.weight
create_tensor: loading tensor blk.18.attn_v.weight
create_tensor: loading tensor blk.18.attn_output.weight
create_tensor: loading tensor blk.18.attn_q_norm.weight
create_tensor: loading tensor blk.18.attn_k_norm.weight
create_tensor: loading tensor blk.18.post_attention_norm.weight
create_tensor: loading tensor blk.18.layer_output_scale.weight
create_tensor: loading tensor blk.18.ffn_norm.weight
create_tensor: loading tensor blk.18.ffn_gate.weight
create_tensor: loading tensor blk.18.ffn_up.weight
create_tensor: loading tensor blk.18.ffn_down.weight
create_tensor: loading tensor blk.18.post_ffw_norm.weight
create_tensor: loading tensor blk.19.attn_norm.weight
create_tensor: loading tensor blk.19.attn_q.weight
create_tensor: loading tensor blk.19.attn_k.weight
create_tensor: loading tensor blk.19.attn_v.weight
create_tensor: loading tensor blk.19.attn_output.weight
create_tensor: loading tensor blk.19.attn_q_norm.weight
create_tensor: loading tensor blk.19.attn_k_norm.weight
create_tensor: loading tensor blk.19.post_attention_norm.weight
create_tensor: loading tensor blk.19.layer_output_scale.weight
create_tensor: loading tensor blk.19.ffn_norm.weight
create_tensor: loading tensor blk.19.ffn_gate.weight
create_tensor: loading tensor blk.19.ffn_up.weight
create_tensor: loading tensor blk.19.ffn_down.weight
create_tensor: loading tensor blk.19.post_ffw_norm.weight
create_tensor: loading tensor blk.20.attn_norm.weight
create_tensor: loading tensor blk.20.attn_q.weight
create_tensor: loading tensor blk.20.attn_k.weight
create_tensor: loading tensor blk.20.attn_v.weight
create_tensor: loading tensor blk.20.attn_output.weight
create_tensor: loading tensor blk.20.attn_q_norm.weight
create_tensor: loading tensor blk.20.attn_k_norm.weight
create_tensor: loading tensor blk.20.post_attention_norm.weight
create_tensor: loading tensor blk.20.layer_output_scale.weight
create_tensor: loading tensor blk.20.ffn_norm.weight
create_tensor: loading tensor blk.20.ffn_gate.weight
create_tensor: loading tensor blk.20.ffn_up.weight
create_tensor: loading tensor blk.20.ffn_down.weight
create_tensor: loading tensor blk.20.post_ffw_norm.weight
create_tensor: loading tensor blk.21.attn_norm.weight
create_tensor: loading tensor blk.21.attn_q.weight
create_tensor: loading tensor blk.21.attn_k.weight
create_tensor: loading tensor blk.21.attn_v.weight
create_tensor: loading tensor blk.21.attn_output.weight
create_tensor: loading tensor blk.21.attn_q_norm.weight
create_tensor: loading tensor blk.21.attn_k_norm.weight
create_tensor: loading tensor blk.21.post_attention_norm.weight
create_tensor: loading tensor blk.21.layer_output_scale.weight
create_tensor: loading tensor blk.21.ffn_norm.weight
create_tensor: loading tensor blk.21.ffn_gate.weight
create_tensor: loading tensor blk.21.ffn_up.weight
create_tensor: loading tensor blk.21.ffn_down.weight
create_tensor: loading tensor blk.21.post_ffw_norm.weight
create_tensor: loading tensor blk.22.attn_norm.weight
create_tensor: loading tensor blk.22.attn_q.weight
create_tensor: loading tensor blk.22.attn_k.weight
create_tensor: loading tensor blk.22.attn_v.weight
create_tensor: loading tensor blk.22.attn_output.weight
create_tensor: loading tensor blk.22.attn_q_norm.weight
create_tensor: loading tensor blk.22.attn_k_norm.weight
create_tensor: loading tensor blk.22.post_attention_norm.weight
create_tensor: loading tensor blk.22.layer_output_scale.weight
create_tensor: loading tensor blk.22.ffn_norm.weight
create_tensor: loading tensor blk.22.ffn_gate.weight
create_tensor: loading tensor blk.22.ffn_up.weight
create_tensor: loading tensor blk.22.ffn_down.weight
create_tensor: loading tensor blk.22.post_ffw_norm.weight
create_tensor: loading tensor blk.23.attn_norm.weight
create_tensor: loading tensor blk.23.attn_q.weight
create_tensor: loading tensor blk.23.attn_k.weight
create_tensor: loading tensor blk.23.attn_output.weight
create_tensor: loading tensor blk.23.attn_q_norm.weight
create_tensor: loading tensor blk.23.attn_k_norm.weight
create_tensor: loading tensor blk.23.post_attention_norm.weight
create_tensor: loading tensor blk.23.layer_output_scale.weight
create_tensor: loading tensor blk.23.ffn_norm.weight
create_tensor: loading tensor blk.23.ffn_gate.weight
create_tensor: loading tensor blk.23.ffn_up.weight
create_tensor: loading tensor blk.23.ffn_down.weight
create_tensor: loading tensor blk.23.post_ffw_norm.weight
create_tensor: loading tensor blk.24.attn_norm.weight
create_tensor: loading tensor blk.24.attn_q.weight
create_tensor: loading tensor blk.24.attn_k.weight
create_tensor: loading tensor blk.24.attn_v.weight
create_tensor: loading tensor blk.24.attn_output.weight
create_tensor: loading tensor blk.24.attn_q_norm.weight
create_tensor: loading tensor blk.24.attn_k_norm.weight
create_tensor: loading tensor blk.24.post_attention_norm.weight
create_tensor: loading tensor blk.24.layer_output_scale.weight
create_tensor: loading tensor blk.24.ffn_norm.weight
create_tensor: loading tensor blk.24.ffn_gate.weight
create_tensor: loading tensor blk.24.ffn_up.weight
create_tensor: loading tensor blk.24.ffn_down.weight
create_tensor: loading tensor blk.24.post_ffw_norm.weight
create_tensor: loading tensor blk.25.attn_norm.weight
create_tensor: loading tensor blk.25.attn_q.weight
create_tensor: loading tensor blk.25.attn_k.weight
create_tensor: loading tensor blk.25.attn_v.weight
create_tensor: loading tensor blk.25.attn_output.weight
create_tensor: loading tensor blk.25.attn_q_norm.weight
create_tensor: loading tensor blk.25.attn_k_norm.weight
create_tensor: loading tensor blk.25.post_attention_norm.weight
create_tensor: loading tensor blk.25.layer_output_scale.weight
create_tensor: loading tensor blk.25.ffn_norm.weight
create_tensor: loading tensor blk.25.ffn_gate.weight
create_tensor: loading tensor blk.25.ffn_up.weight
create_tensor: loading tensor blk.25.ffn_down.weight
create_tensor: loading tensor blk.25.post_ffw_norm.weight
create_tensor: loading tensor blk.26.attn_norm.weight
create_tensor: loading tensor blk.26.attn_q.weight
create_tensor: loading tensor blk.26.attn_k.weight
create_tensor: loading tensor blk.26.attn_v.weight
create_tensor: loading tensor blk.26.attn_output.weight
create_tensor: loading tensor blk.26.attn_q_norm.weight
create_tensor: loading tensor blk.26.attn_k_norm.weight
create_tensor: loading tensor blk.26.post_attention_norm.weight
create_tensor: loading tensor blk.26.layer_output_scale.weight
create_tensor: loading tensor blk.26.ffn_norm.weight
create_tensor: loading tensor blk.26.ffn_gate.weight
create_tensor: loading tensor blk.26.ffn_up.weight
create_tensor: loading tensor blk.26.ffn_down.weight
create_tensor: loading tensor blk.26.post_ffw_norm.weight
create_tensor: loading tensor blk.27.attn_norm.weight
create_tensor: loading tensor blk.27.attn_q.weight
create_tensor: loading tensor blk.27.attn_k.weight
create_tensor: loading tensor blk.27.attn_v.weight
create_tensor: loading tensor blk.27.attn_output.weight
create_tensor: loading tensor blk.27.attn_q_norm.weight
create_tensor: loading tensor blk.27.attn_k_norm.weight
create_tensor: loading tensor blk.27.post_attention_norm.weight
create_tensor: loading tensor blk.27.layer_output_scale.weight
create_tensor: loading tensor blk.27.ffn_norm.weight
create_tensor: loading tensor blk.27.ffn_gate.weight
create_tensor: loading tensor blk.27.ffn_up.weight
create_tensor: loading tensor blk.27.ffn_down.weight
create_tensor: loading tensor blk.27.post_ffw_norm.weight
create_tensor: loading tensor blk.28.attn_norm.weight
create_tensor: loading tensor blk.28.attn_q.weight
create_tensor: loading tensor blk.28.attn_k.weight
create_tensor: loading tensor blk.28.attn_v.weight
create_tensor: loading tensor blk.28.attn_output.weight
create_tensor: loading tensor blk.28.attn_q_norm.weight
create_tensor: loading tensor blk.28.attn_k_norm.weight
create_tensor: loading tensor blk.28.post_attention_norm.weight
create_tensor: loading tensor blk.28.layer_output_scale.weight
create_tensor: loading tensor blk.28.ffn_norm.weight
create_tensor: loading tensor blk.28.ffn_gate.weight
create_tensor: loading tensor blk.28.ffn_up.weight
create_tensor: loading tensor blk.28.ffn_down.weight
create_tensor: loading tensor blk.28.post_ffw_norm.weight
create_tensor: loading tensor blk.29.attn_norm.weight
create_tensor: loading tensor blk.29.attn_q.weight
create_tensor: loading tensor blk.29.attn_k.weight
create_tensor: loading tensor blk.29.attn_output.weight
create_tensor: loading tensor blk.29.attn_q_norm.weight
create_tensor: loading tensor blk.29.attn_k_norm.weight
create_tensor: loading tensor blk.29.post_attention_norm.weight
create_tensor: loading tensor blk.29.layer_output_scale.weight
create_tensor: loading tensor blk.29.ffn_norm.weight
create_tensor: loading tensor blk.29.ffn_gate.weight
create_tensor: loading tensor blk.29.ffn_up.weight
create_tensor: loading tensor blk.29.ffn_down.weight
create_tensor: loading tensor blk.29.post_ffw_norm.weight
create_tensor: loading tensor blk.30.attn_norm.weight
create_tensor: loading tensor blk.30.attn_q.weight
create_tensor: loading tensor blk.30.attn_k.weight
create_tensor: loading tensor blk.30.attn_v.weight
create_tensor: loading tensor blk.30.attn_output.weight
create_tensor: loading tensor blk.30.attn_q_norm.weight
create_tensor: loading tensor blk.30.attn_k_norm.weight
create_tensor: loading tensor blk.30.post_attention_norm.weight
create_tensor: loading tensor blk.30.layer_output_scale.weight
create_tensor: loading tensor blk.30.ffn_norm.weight
create_tensor: loading tensor blk.30.ffn_gate.weight
create_tensor: loading tensor blk.30.ffn_up.weight
create_tensor: loading tensor blk.30.ffn_down.weight
create_tensor: loading tensor blk.30.post_ffw_norm.weight
create_tensor: loading tensor blk.31.attn_norm.weight
create_tensor: loading tensor blk.31.attn_q.weight
create_tensor: loading tensor blk.31.attn_k.weight
create_tensor: loading tensor blk.31.attn_v.weight
create_tensor: loading tensor blk.31.attn_output.weight
create_tensor: loading tensor blk.31.attn_q_norm.weight
create_tensor: loading tensor blk.31.attn_k_norm.weight
create_tensor: loading tensor blk.31.post_attention_norm.weight
create_tensor: loading tensor blk.31.layer_output_scale.weight
create_tensor: loading tensor blk.31.ffn_norm.weight
create_tensor: loading tensor blk.31.ffn_gate.weight
create_tensor: loading tensor blk.31.ffn_up.weight
create_tensor: loading tensor blk.31.ffn_down.weight
create_tensor: loading tensor blk.31.post_ffw_norm.weight
create_tensor: loading tensor blk.32.attn_norm.weight
create_tensor: loading tensor blk.32.attn_q.weight
create_tensor: loading tensor blk.32.attn_k.weight
create_tensor: loading tensor blk.32.attn_v.weight
create_tensor: loading tensor blk.32.attn_output.weight
create_tensor: loading tensor blk.32.attn_q_norm.weight
create_tensor: loading tensor blk.32.attn_k_norm.weight
create_tensor: loading tensor blk.32.post_attention_norm.weight
create_tensor: loading tensor blk.32.layer_output_scale.weight
create_tensor: loading tensor blk.32.ffn_norm.weight
create_tensor: loading tensor blk.32.ffn_gate.weight
create_tensor: loading tensor blk.32.ffn_up.weight
create_tensor: loading tensor blk.32.ffn_down.weight
create_tensor: loading tensor blk.32.post_ffw_norm.weight
create_tensor: loading tensor blk.33.attn_norm.weight
create_tensor: loading tensor blk.33.attn_q.weight
create_tensor: loading tensor blk.33.attn_k.weight
create_tensor: loading tensor blk.33.attn_v.weight
create_tensor: loading tensor blk.33.attn_output.weight
create_tensor: loading tensor blk.33.attn_q_norm.weight
create_tensor: loading tensor blk.33.attn_k_norm.weight
create_tensor: loading tensor blk.33.post_attention_norm.weight
create_tensor: loading tensor blk.33.layer_output_scale.weight
create_tensor: loading tensor blk.33.ffn_norm.weight
create_tensor: loading tensor blk.33.ffn_gate.weight
create_tensor: loading tensor blk.33.ffn_up.weight
create_tensor: loading tensor blk.33.ffn_down.weight
create_tensor: loading tensor blk.33.post_ffw_norm.weight
create_tensor: loading tensor blk.34.attn_norm.weight
create_tensor: loading tensor blk.34.attn_q.weight
create_tensor: loading tensor blk.34.attn_k.weight
create_tensor: loading tensor blk.34.attn_v.weight
create_tensor: loading tensor blk.34.attn_output.weight
create_tensor: loading tensor blk.34.attn_q_norm.weight
create_tensor: loading tensor blk.34.attn_k_norm.weight
create_tensor: loading tensor blk.34.post_attention_norm.weight
create_tensor: loading tensor blk.34.layer_output_scale.weight
create_tensor: loading tensor blk.34.ffn_norm.weight
create_tensor: loading tensor blk.34.ffn_gate.weight
create_tensor: loading tensor blk.34.ffn_up.weight
create_tensor: loading tensor blk.34.ffn_down.weight
create_tensor: loading tensor blk.34.post_ffw_norm.weight
create_tensor: loading tensor blk.35.attn_norm.weight
create_tensor: loading tensor blk.35.attn_q.weight
create_tensor: loading tensor blk.35.attn_k.weight
create_tensor: loading tensor blk.35.attn_output.weight
create_tensor: loading tensor blk.35.attn_q_norm.weight
create_tensor: loading tensor blk.35.attn_k_norm.weight
create_tensor: loading tensor blk.35.post_attention_norm.weight
create_tensor: loading tensor blk.35.layer_output_scale.weight
create_tensor: loading tensor blk.35.ffn_norm.weight
create_tensor: loading tensor blk.35.ffn_gate.weight
create_tensor: loading tensor blk.35.ffn_up.weight
create_tensor: loading tensor blk.35.ffn_down.weight
create_tensor: loading tensor blk.35.post_ffw_norm.weight
create_tensor: loading tensor blk.36.attn_norm.weight
create_tensor: loading tensor blk.36.attn_q.weight
create_tensor: loading tensor blk.36.attn_k.weight
create_tensor: loading tensor blk.36.attn_v.weight
create_tensor: loading tensor blk.36.attn_output.weight
create_tensor: loading tensor blk.36.attn_q_norm.weight
create_tensor: loading tensor blk.36.attn_k_norm.weight
create_tensor: loading tensor blk.36.post_attention_norm.weight
create_tensor: loading tensor blk.36.layer_output_scale.weight
create_tensor: loading tensor blk.36.ffn_norm.weight
create_tensor: loading tensor blk.36.ffn_gate.weight
create_tensor: loading tensor blk.36.ffn_up.weight
create_tensor: loading tensor blk.36.ffn_down.weight
create_tensor: loading tensor blk.36.post_ffw_norm.weight
create_tensor: loading tensor blk.37.attn_norm.weight
create_tensor: loading tensor blk.37.attn_q.weight
create_tensor: loading tensor blk.37.attn_k.weight
create_tensor: loading tensor blk.37.attn_v.weight
create_tensor: loading tensor blk.37.attn_output.weight
create_tensor: loading tensor blk.37.attn_q_norm.weight
create_tensor: loading tensor blk.37.attn_k_norm.weight
create_tensor: loading tensor blk.37.post_attention_norm.weight
create_tensor: loading tensor blk.37.layer_output_scale.weight
create_tensor: loading tensor blk.37.ffn_norm.weight
create_tensor: loading tensor blk.37.ffn_gate.weight
create_tensor: loading tensor blk.37.ffn_up.weight
create_tensor: loading tensor blk.37.ffn_down.weight
create_tensor: loading tensor blk.37.post_ffw_norm.weight
create_tensor: loading tensor blk.38.attn_norm.weight
create_tensor: loading tensor blk.38.attn_q.weight
create_tensor: loading tensor blk.38.attn_k.weight
create_tensor: loading tensor blk.38.attn_v.weight
create_tensor: loading tensor blk.38.attn_output.weight
create_tensor: loading tensor blk.38.attn_q_norm.weight
create_tensor: loading tensor blk.38.attn_k_norm.weight
create_tensor: loading tensor blk.38.post_attention_norm.weight
create_tensor: loading tensor blk.38.layer_output_scale.weight
create_tensor: loading tensor blk.38.ffn_norm.weight
create_tensor: loading tensor blk.38.ffn_gate.weight
create_tensor: loading tensor blk.38.ffn_up.weight
create_tensor: loading tensor blk.38.ffn_down.weight
create_tensor: loading tensor blk.38.post_ffw_norm.weight
create_tensor: loading tensor blk.39.attn_norm.weight
create_tensor: loading tensor blk.39.attn_q.weight
create_tensor: loading tensor blk.39.attn_k.weight
create_tensor: loading tensor blk.39.attn_v.weight
create_tensor: loading tensor blk.39.attn_output.weight
create_tensor: loading tensor blk.39.attn_q_norm.weight
create_tensor: loading tensor blk.39.attn_k_norm.weight
create_tensor: loading tensor blk.39.post_attention_norm.weight
create_tensor: loading tensor blk.39.layer_output_scale.weight
create_tensor: loading tensor blk.39.ffn_norm.weight
create_tensor: loading tensor blk.39.ffn_gate.weight
create_tensor: loading tensor blk.39.ffn_up.weight
create_tensor: loading tensor blk.39.ffn_down.weight
create_tensor: loading tensor blk.39.post_ffw_norm.weight
create_tensor: loading tensor blk.40.attn_norm.weight
create_tensor: loading tensor blk.40.attn_q.weight
create_tensor: loading tensor blk.40.attn_k.weight
create_tensor: loading tensor blk.40.attn_v.weight
create_tensor: loading tensor blk.40.attn_output.weight
create_tensor: loading tensor blk.40.attn_q_norm.weight
create_tensor: loading tensor blk.40.attn_k_norm.weight
create_tensor: loading tensor blk.40.post_attention_norm.weight
create_tensor: loading tensor blk.40.layer_output_scale.weight
create_tensor: loading tensor blk.40.ffn_norm.weight
create_tensor: loading tensor blk.40.ffn_gate.weight
create_tensor: loading tensor blk.40.ffn_up.weight
create_tensor: loading tensor blk.40.ffn_down.weight
create_tensor: loading tensor blk.40.post_ffw_norm.weight
create_tensor: loading tensor blk.41.attn_norm.weight
create_tensor: loading tensor blk.41.attn_q.weight
create_tensor: loading tensor blk.41.attn_k.weight
create_tensor: loading tensor blk.41.attn_output.weight
create_tensor: loading tensor blk.41.attn_q_norm.weight
create_tensor: loading tensor blk.41.attn_k_norm.weight
create_tensor: loading tensor blk.41.post_attention_norm.weight
create_tensor: loading tensor blk.41.layer_output_scale.weight
create_tensor: loading tensor blk.41.ffn_norm.weight
create_tensor: loading tensor blk.41.ffn_gate.weight
create_tensor: loading tensor blk.41.ffn_up.weight
create_tensor: loading tensor blk.41.ffn_down.weight
create_tensor: loading tensor blk.41.post_ffw_norm.weight
create_tensor: loading tensor blk.42.attn_norm.weight
create_tensor: loading tensor blk.42.attn_q.weight
create_tensor: loading tensor blk.42.attn_k.weight
create_tensor: loading tensor blk.42.attn_v.weight
create_tensor: loading tensor blk.42.attn_output.weight
create_tensor: loading tensor blk.42.attn_q_norm.weight
create_tensor: loading tensor blk.42.attn_k_norm.weight
create_tensor: loading tensor blk.42.post_attention_norm.weight
create_tensor: loading tensor blk.42.layer_output_scale.weight
create_tensor: loading tensor blk.42.ffn_norm.weight
create_tensor: loading tensor blk.42.ffn_gate.weight
create_tensor: loading tensor blk.42.ffn_up.weight
create_tensor: loading tensor blk.42.ffn_down.weight
create_tensor: loading tensor blk.42.post_ffw_norm.weight
create_tensor: loading tensor blk.43.attn_norm.weight
create_tensor: loading tensor blk.43.attn_q.weight
create_tensor: loading tensor blk.43.attn_k.weight
create_tensor: loading tensor blk.43.attn_v.weight
create_tensor: loading tensor blk.43.attn_output.weight
create_tensor: loading tensor blk.43.attn_q_norm.weight
create_tensor: loading tensor blk.43.attn_k_norm.weight
create_tensor: loading tensor blk.43.post_attention_norm.weight
create_tensor: loading tensor blk.43.layer_output_scale.weight
create_tensor: loading tensor blk.43.ffn_norm.weight
create_tensor: loading tensor blk.43.ffn_gate.weight
create_tensor: loading tensor blk.43.ffn_up.weight
create_tensor: loading tensor blk.43.ffn_down.weight
create_tensor: loading tensor blk.43.post_ffw_norm.weight
create_tensor: loading tensor blk.44.attn_norm.weight
create_tensor: loading tensor blk.44.attn_q.weight
create_tensor: loading tensor blk.44.attn_k.weight
create_tensor: loading tensor blk.44.attn_v.weight
create_tensor: loading tensor blk.44.attn_output.weight
create_tensor: loading tensor blk.44.attn_q_norm.weight
create_tensor: loading tensor blk.44.attn_k_norm.weight
create_tensor: loading tensor blk.44.post_attention_norm.weight
create_tensor: loading tensor blk.44.layer_output_scale.weight
create_tensor: loading tensor blk.44.ffn_norm.weight
create_tensor: loading tensor blk.44.ffn_gate.weight
create_tensor: loading tensor blk.44.ffn_up.weight
create_tensor: loading tensor blk.44.ffn_down.weight
create_tensor: loading tensor blk.44.post_ffw_norm.weight
create_tensor: loading tensor blk.45.attn_norm.weight
create_tensor: loading tensor blk.45.attn_q.weight
create_tensor: loading tensor blk.45.attn_k.weight
create_tensor: loading tensor blk.45.attn_v.weight
create_tensor: loading tensor blk.45.attn_output.weight
create_tensor: loading tensor blk.45.attn_q_norm.weight
create_tensor: loading tensor blk.45.attn_k_norm.weight
create_tensor: loading tensor blk.45.post_attention_norm.weight
create_tensor: loading tensor blk.45.layer_output_scale.weight
create_tensor: loading tensor blk.45.ffn_norm.weight
create_tensor: loading tensor blk.45.ffn_gate.weight
create_tensor: loading tensor blk.45.ffn_up.weight
create_tensor: loading tensor blk.45.ffn_down.weight
create_tensor: loading tensor blk.45.post_ffw_norm.weight
create_tensor: loading tensor blk.46.attn_norm.weight
create_tensor: loading tensor blk.46.attn_q.weight
create_tensor: loading tensor blk.46.attn_k.weight
create_tensor: loading tensor blk.46.attn_v.weight
create_tensor: loading tensor blk.46.attn_output.weight
create_tensor: loading tensor blk.46.attn_q_norm.weight
create_tensor: loading tensor blk.46.attn_k_norm.weight
create_tensor: loading tensor blk.46.post_attention_norm.weight
create_tensor: loading tensor blk.46.layer_output_scale.weight
create_tensor: loading tensor blk.46.ffn_norm.weight
create_tensor: loading tensor blk.46.ffn_gate.weight
create_tensor: loading tensor blk.46.ffn_up.weight
create_tensor: loading tensor blk.46.ffn_down.weight
create_tensor: loading tensor blk.46.post_ffw_norm.weight
create_tensor: loading tensor blk.47.attn_norm.weight
create_tensor: loading tensor blk.47.attn_q.weight
create_tensor: loading tensor blk.47.attn_k.weight
create_tensor: loading tensor blk.47.attn_output.weight
create_tensor: loading tensor blk.47.attn_q_norm.weight
create_tensor: loading tensor blk.47.attn_k_norm.weight
create_tensor: loading tensor blk.47.post_attention_norm.weight
create_tensor: loading tensor blk.47.layer_output_scale.weight
create_tensor: loading tensor blk.47.ffn_norm.weight
create_tensor: loading tensor blk.47.ffn_gate.weight
create_tensor: loading tensor blk.47.ffn_up.weight
create_tensor: loading tensor blk.47.ffn_down.weight
create_tensor: loading tensor blk.47.post_ffw_norm.weight
done_getting_tensors: tensor 'token_embd.weight' (q4_K) (and 674 others) cannot be used with preferred buffer type CPU_REPACK, using CPU instead
load_tensors:          CPU model buffer size =  7316.85 MiB
load_all_data: no device found for buffer type CPU for async uploads
........................................................................................
load_tensors: releasing mmaps (not needed after tensor copy)
llama_model_loader: direct I/O is enabled, disabling mmap
llama_model_loader: loaded meta data with 52 key-value pairs and 49 tensors from c:\llama.cpp\models\Gemma-4\12B\gemma-4-12B-it-Q4_K_M-MTP-assistant.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = gemma4_assistant
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                     general.sampling.top_k i32              = 64
llama_model_loader: - kv   3:                     general.sampling.top_p f32              = 0.950000
llama_model_loader: - kv   4:                      general.sampling.temp f32              = 1.000000
llama_model_loader: - kv   5:                               general.name str              = Google Gemma 4 12B It Assistant
llama_model_loader: - kv   6:                           general.finetune str              = 12b-it-assistant
llama_model_loader: - kv   7:                           general.basename str              = google-gemma-4
llama_model_loader: - kv   8:                         general.size_label str              = 423M
llama_model_loader: - kv   9:                            general.license str              = apache-2.0
llama_model_loader: - kv  10:                       general.license.link str              = https://ai.google.dev/gemma/docs/gemm...
llama_model_loader: - kv  11:                               general.tags arr[str,1]       = ["any-to-any"]
llama_model_loader: - kv  12:               gemma4_assistant.block_count u32              = 4
llama_model_loader: - kv  13:            gemma4_assistant.context_length u32              = 262144
llama_model_loader: - kv  14:          gemma4_assistant.embedding_length u32              = 1024
llama_model_loader: - kv  15:       gemma4_assistant.feed_forward_length u32              = 8192
llama_model_loader: - kv  16:      gemma4_assistant.attention.head_count u32              = 16
llama_model_loader: - kv  17:   gemma4_assistant.attention.head_count_kv arr[i32,4]       = [8, 8, 8, 1]
llama_model_loader: - kv  18:            gemma4_assistant.rope.freq_base f32              = 1000000.000000
llama_model_loader: - kv  19:        gemma4_assistant.rope.freq_base_swa f32              = 10000.000000
llama_model_loader: - kv  20: gemma4_assistant.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  21:      gemma4_assistant.attention.key_length u32              = 512
llama_model_loader: - kv  22:    gemma4_assistant.attention.value_length u32              = 512
llama_model_loader: - kv  23:  gemma4_assistant.attention.sliding_window u32              = 1024
llama_model_loader: - kv  24: gemma4_assistant.attention.shared_kv_layers u32              = 4
llama_model_loader: - kv  25: gemma4_assistant.embedding_length_per_layer_input u32              = 0
llama_model_loader: - kv  26: gemma4_assistant.attention.sliding_window_pattern arr[bool,4]      = [true, true, true, false]
llama_model_loader: - kv  27:  gemma4_assistant.attention.key_length_swa u32              = 256
llama_model_loader: - kv  28: gemma4_assistant.attention.value_length_swa u32              = 256
llama_model_loader: - kv  29:      gemma4_assistant.rope.dimension_count u32              = 512
llama_model_loader: - kv  30:  gemma4_assistant.rope.dimension_count_swa u32              = 256
llama_model_loader: - kv  31:               gemma4_assistant.n_centroids u32              = 2048
llama_model_loader: - kv  32:            gemma4_assistant.centroid_top_k u32              = 32
llama_model_loader: - kv  33:           gemma4_assistant.n_embd_backbone u32              = 3840
llama_model_loader: - kv  34:    gemma4_assistant.use_ordered_embeddings bool             = false
llama_model_loader: - kv  35:          gemma4_assistant.attention.k_eq_v bool             = true
llama_model_loader: - kv  36:      gemma4_assistant.requires_target_arch str              = gemma4
llama_model_loader: - kv  37:                       tokenizer.ggml.model str              = gemma4
llama_model_loader: - kv  38:                      tokenizer.ggml.tokens arr[str,262144]  = ["<pad>", "<eos>", "<bos>", "<unk>", ...
llama_model_loader: - kv  39:                      tokenizer.ggml.scores arr[f32,262144]  = [-1000.000000, -1000.000000, -1000.00...
llama_model_loader: - kv  40:                  tokenizer.ggml.token_type arr[i32,262144]  = [3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  41:                      tokenizer.ggml.merges arr[str,514906]  = ["\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n \n", ...
llama_model_loader: - kv  42:                tokenizer.ggml.bos_token_id u32              = 2
llama_model_loader: - kv  43:                tokenizer.ggml.eos_token_id u32              = 1
llama_model_loader: - kv  44:            tokenizer.ggml.unknown_token_id u32              = 3
llama_model_loader: - kv  45:            tokenizer.ggml.padding_token_id u32              = 0
llama_model_loader: - kv  46:               tokenizer.ggml.mask_token_id u32              = 4
llama_model_loader: - kv  47:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  48:               tokenizer.ggml.add_sep_token bool             = false
llama_model_loader: - kv  49:            tokenizer.ggml.add_space_prefix bool             = false
llama_model_loader: - kv  50:               general.quantization_version u32              = 2
llama_model_loader: - kv  51:                          general.file_type u32              = 15
llama_model_loader: - type  f32:   26 tensors
llama_model_loader: - type q4_K:   20 tensors
llama_model_loader: - type q6_K:    3 tensors
print_info: file format = GGUF V3 (latest)
print_info: file type   = Q4_K - Medium
print_info: file size   = 297.03 MiB (5.89 BPW)
init_tokenizer: initializing tokenizer for type 2
load: 0 unused tokens
load: control token:      0 '<pad>' is not marked as EOG
load: control token:      3 '<unk>' is not marked as EOG
load: control token:      2 '<bos>' is not marked as EOG
load: control token:      4 '<mask>' is not marked as EOG
load: control token:     47 '<tool|>' is not marked as EOG
load: control token:     98 '<|think|>' is not marked as EOG
load: control token:     46 '<|tool>' is not marked as EOG
load: control-looking token:     50 '<|tool_response>' was not control-type; this is probably a bug in the model. its type will be overridden
load: control token:    105 '<|turn>' is not marked as EOG
load: control-looking token:    212 '</s>' was not control-type; this is probably a bug in the model. its type will be overridden
load: control token: 258880 '<|image|>' is not marked as EOG
load: control token: 258882 '<image|>' is not marked as EOG
load: control token: 256000 '<|audio>' is not marked as EOG
load: control token: 258883 '<audio|>' is not marked as EOG
load: control token: 255999 '<|image>' is not marked as EOG
load: control token: 258881 '<|audio|>' is not marked as EOG
load: printing all EOG tokens:
load:   - 1 ('<eos>')
load:   - 50 ('<|tool_response>')
load:   - 106 ('<turn|>')
load:   - 212 ('</s>')
load: special_eog_ids contains '<|tool_response>', removing '</s>' token from EOG list
load: special tokens cache size = 23
load: token to piece cache size = 1.9445 MB
print_info: arch                  = gemma4_assistant
print_info: vocab_only            = 0
print_info: no_alloc              = 0
print_info: n_ctx_train           = 262144
print_info: n_embd                = 1024
print_info: n_embd_inp            = 1024
print_info: n_layer               = 4
print_info: n_head                = 16
print_info: n_head_kv             = [8, 8, 8, 1]
print_info: n_rot                 = 512
print_info: n_swa                 = 1024
print_info: is_swa_any            = 1
print_info: n_embd_head_k         = 512
print_info: n_embd_head_v         = 512
print_info: n_gqa                 = [2, 2, 2, 16]
print_info: n_embd_k_gqa          = [2048, 2048, 2048, 512]
print_info: n_embd_v_gqa          = [2048, 2048, 2048, 512]
print_info: f_norm_eps            = 0.0e+00
print_info: f_norm_rms_eps        = 1.0e-06
print_info: f_clamp_kqv           = 0.0e+00
print_info: f_max_alibi_bias      = 0.0e+00
print_info: f_logit_scale         = 0.0e+00
print_info: f_attn_scale          = 1.0e+00
print_info: f_attn_value_scale    = 0.0000
print_info: n_ff                  = 8192
print_info: n_expert              = 0
print_info: n_expert_used         = 0
print_info: n_expert_groups       = 0
print_info: n_group_used          = 0
print_info: causal attn           = 1
print_info: pooling type          = -1
print_info: rope type             = 2
print_info: rope scaling          = linear
print_info: freq_base_train       = 1000000.0
print_info: freq_scale_train      = 1
print_info: freq_base_swa         = 10000.0
print_info: freq_scale_swa        = 1
print_info: n_embd_head_k_swa     = 256
print_info: n_embd_head_v_swa     = 256
print_info: n_rot_swa             = 256
print_info: n_ctx_orig_yarn       = 262144
print_info: rope_yarn_log_mul     = 0.0000
print_info: rope_finetuned        = unknown
print_info: model type            = ?B
print_info: model params          = 422.86 M
print_info: general.name          = Google Gemma 4 12B It Assistant
print_info: vocab type            = BPE
print_info: n_vocab               = 262144
print_info: n_merges              = 514906
print_info: BOS token             = 2 '<bos>'
print_info: EOS token             = 1 '<eos>'
print_info: UNK token             = 3 '<unk>'
print_info: PAD token             = 0 '<pad>'
print_info: MASK token            = 4 '<mask>'
print_info: LF token              = 107 '
'
print_info: EOG token             = 1 '<eos>'
print_info: EOG token             = 50 '<|tool_response>'
print_info: EOG token             = 106 '<turn|>'
print_info: max token length      = 93
load_tensors: loading model tensors, this can take a while... (mmap = false, direct_io = true)
load_tensors: layer   0 assigned to device CPU, is_swa = 1
load_tensors: layer   1 assigned to device CPU, is_swa = 1
load_tensors: layer   2 assigned to device CPU, is_swa = 1
load_tensors: layer   3 assigned to device CPU, is_swa = 0
load_tensors: layer   4 assigned to device CPU, is_swa = 0
create_tensor: loading tensor token_embd.weight
create_tensor: loading tensor mtp.pre_projection.weight
create_tensor: loading tensor mtp.post_projection.weight
create_tensor: loading tensor output_norm.weight
create_tensor: loading tensor blk.0.attn_norm.weight
create_tensor: loading tensor blk.0.attn_q.weight
create_tensor: loading tensor blk.0.attn_output.weight
create_tensor: loading tensor blk.0.attn_q_norm.weight
create_tensor: loading tensor blk.0.post_attention_norm.weight
create_tensor: loading tensor blk.0.layer_output_scale.weight
create_tensor: loading tensor blk.0.ffn_norm.weight
create_tensor: loading tensor blk.0.ffn_gate.weight
create_tensor: loading tensor blk.0.ffn_up.weight
create_tensor: loading tensor blk.0.ffn_down.weight
create_tensor: loading tensor blk.0.post_ffw_norm.weight
create_tensor: loading tensor blk.1.attn_norm.weight
create_tensor: loading tensor blk.1.attn_q.weight
create_tensor: loading tensor blk.1.attn_output.weight
create_tensor: loading tensor blk.1.attn_q_norm.weight
create_tensor: loading tensor blk.1.post_attention_norm.weight
create_tensor: loading tensor blk.1.layer_output_scale.weight
create_tensor: loading tensor blk.1.ffn_norm.weight
create_tensor: loading tensor blk.1.ffn_gate.weight
create_tensor: loading tensor blk.1.ffn_up.weight
create_tensor: loading tensor blk.1.ffn_down.weight
create_tensor: loading tensor blk.1.post_ffw_norm.weight
create_tensor: loading tensor blk.2.attn_norm.weight
create_tensor: loading tensor blk.2.attn_q.weight
create_tensor: loading tensor blk.2.attn_output.weight
create_tensor: loading tensor blk.2.attn_q_norm.weight
create_tensor: loading tensor blk.2.post_attention_norm.weight
create_tensor: loading tensor blk.2.layer_output_scale.weight
create_tensor: loading tensor blk.2.ffn_norm.weight
create_tensor: loading tensor blk.2.ffn_gate.weight
create_tensor: loading tensor blk.2.ffn_up.weight
create_tensor: loading tensor blk.2.ffn_down.weight
create_tensor: loading tensor blk.2.post_ffw_norm.weight
create_tensor: loading tensor blk.3.attn_norm.weight
create_tensor: loading tensor blk.3.attn_q.weight
create_tensor: loading tensor blk.3.attn_output.weight
create_tensor: loading tensor blk.3.attn_q_norm.weight
create_tensor: loading tensor blk.3.post_attention_norm.weight
create_tensor: loading tensor blk.3.layer_output_scale.weight
create_tensor: loading tensor rope_freqs.weight
create_tensor: loading tensor blk.3.ffn_norm.weight
create_tensor: loading tensor blk.3.ffn_gate.weight
create_tensor: loading tensor blk.3.ffn_up.weight
create_tensor: loading tensor blk.3.ffn_down.weight
create_tensor: loading tensor blk.3.post_ffw_norm.weight
done_getting_tensors: tensor 'token_embd.weight' (q6_K) (and 48 others) cannot be used with preferred buffer type CPU_REPACK, using CPU instead
load_tensors:          CPU model buffer size =   297.03 MiB
load_all_data: no device found for buffer type CPU for async uploads
.......................
load_tensors: releasing mmaps (not needed after tensor copy)
[main]: loaded MTP assistant head 'c:\llama.cpp\models\Gemma-4\12B\gemma-4-12B-it-Q4_K_M-MTP-assistant.gguf' (n_embd_backbone=3840)
llama_context: constructing llama_context
llama_context: n_seq_max     = 1
llama_context: n_ctx         = 2048
llama_context: n_ctx_seq     = 2048
llama_context: n_batch       = 512
llama_context: n_ubatch      = 512
llama_context: causal_attn   = 1
llama_context: flash_attn    = auto
llama_context: kv_unified    = false
llama_context: freq_base     = 1000000.0
llama_context: freq_scale    = 1
llama_context: n_rs_seq      = 0
llama_context: n_ctx_seq (2048) < n_ctx_train (262144) -- the full capacity of the model will not be utilized
set_abort_callback: call
llama_context:        CPU  output buffer size =     1.00 MiB
llama_kv_cache_iswa: using full-size SWA cache (ref: https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055)
llama_kv_cache_iswa: creating non-SWA KV cache, size = 2048 cells
llama_kv_cache: layer   0: filtered
llama_kv_cache: layer   1: filtered
llama_kv_cache: layer   2: filtered
llama_kv_cache: layer   3: filtered
llama_kv_cache: layer   4: filtered
llama_kv_cache: layer   5: dev = CPU
llama_kv_cache: layer   6: filtered
llama_kv_cache: layer   7: filtered
llama_kv_cache: layer   8: filtered
llama_kv_cache: layer   9: filtered
llama_kv_cache: layer  10: filtered
llama_kv_cache: layer  11: dev = CPU
llama_kv_cache: layer  12: filtered
llama_kv_cache: layer  13: filtered
llama_kv_cache: layer  14: filtered
llama_kv_cache: layer  15: filtered
llama_kv_cache: layer  16: filtered
llama_kv_cache: layer  17: dev = CPU
llama_kv_cache: layer  18: filtered
llama_kv_cache: layer  19: filtered
llama_kv_cache: layer  20: filtered
llama_kv_cache: layer  21: filtered
llama_kv_cache: layer  22: filtered
llama_kv_cache: layer  23: dev = CPU
llama_kv_cache: layer  24: filtered
llama_kv_cache: layer  25: filtered
llama_kv_cache: layer  26: filtered
llama_kv_cache: layer  27: filtered
llama_kv_cache: layer  28: filtered
llama_kv_cache: layer  29: dev = CPU
llama_kv_cache: layer  30: filtered
llama_kv_cache: layer  31: filtered
llama_kv_cache: layer  32: filtered
llama_kv_cache: layer  33: filtered
llama_kv_cache: layer  34: filtered
llama_kv_cache: layer  35: dev = CPU
llama_kv_cache: layer  36: filtered
llama_kv_cache: layer  37: filtered
llama_kv_cache: layer  38: filtered
llama_kv_cache: layer  39: filtered
llama_kv_cache: layer  40: filtered
llama_kv_cache: layer  41: dev = CPU
llama_kv_cache: layer  42: filtered
llama_kv_cache: layer  43: filtered
llama_kv_cache: layer  44: filtered
llama_kv_cache: layer  45: filtered
llama_kv_cache: layer  46: filtered
llama_kv_cache: layer  47: dev = CPU
llama_kv_cache: reusing layers:
llama_kv_cache: - layer   0: no reuse
llama_kv_cache: - layer   1: no reuse
llama_kv_cache: - layer   2: no reuse
llama_kv_cache: - layer   3: no reuse
llama_kv_cache: - layer   4: no reuse
llama_kv_cache: - layer   5: no reuse
llama_kv_cache: - layer   6: no reuse
llama_kv_cache: - layer   7: no reuse
llama_kv_cache: - layer   8: no reuse
llama_kv_cache: - layer   9: no reuse
llama_kv_cache: - layer  10: no reuse
llama_kv_cache: - layer  11: no reuse
llama_kv_cache: - layer  12: no reuse
llama_kv_cache: - layer  13: no reuse
llama_kv_cache: - layer  14: no reuse
llama_kv_cache: - layer  15: no reuse
llama_kv_cache: - layer  16: no reuse
llama_kv_cache: - layer  17: no reuse
llama_kv_cache: - layer  18: no reuse
llama_kv_cache: - layer  19: no reuse
llama_kv_cache: - layer  20: no reuse
llama_kv_cache: - layer  21: no reuse
llama_kv_cache: - layer  22: no reuse
llama_kv_cache: - layer  23: no reuse
llama_kv_cache: - layer  24: no reuse
llama_kv_cache: - layer  25: no reuse
llama_kv_cache: - layer  26: no reuse
llama_kv_cache: - layer  27: no reuse
llama_kv_cache: - layer  28: no reuse
llama_kv_cache: - layer  29: no reuse
llama_kv_cache: - layer  30: no reuse
llama_kv_cache: - layer  31: no reuse
llama_kv_cache: - layer  32: no reuse
llama_kv_cache: - layer  33: no reuse
llama_kv_cache: - layer  34: no reuse
llama_kv_cache: - layer  35: no reuse
llama_kv_cache: - layer  36: no reuse
llama_kv_cache: - layer  37: no reuse
llama_kv_cache: - layer  38: no reuse
llama_kv_cache: - layer  39: no reuse
llama_kv_cache: - layer  40: no reuse
llama_kv_cache: - layer  41: no reuse
llama_kv_cache: - layer  42: no reuse
llama_kv_cache: - layer  43: no reuse
llama_kv_cache: - layer  44: no reuse
llama_kv_cache: - layer  45: no reuse
llama_kv_cache: - layer  46: no reuse
llama_kv_cache: - layer  47: no reuse
llama_kv_cache:        CPU KV buffer size =    32.00 MiB
llama_kv_cache: size =   32.00 MiB (  2048 cells,   8 layers,  1/1 seqs), K (f16):   16.00 MiB, V (f16):   16.00 MiB
llama_kv_cache: attn_rot_k = 0, n_embd_head_k_all = 512
llama_kv_cache: attn_rot_v = 0, n_embd_head_k_all = 512
llama_kv_cache_iswa: creating     SWA KV cache, size = 2048 cells
llama_kv_cache: layer   0: dev = CPU
llama_kv_cache: layer   1: dev = CPU
llama_kv_cache: layer   2: dev = CPU
llama_kv_cache: layer   3: dev = CPU
llama_kv_cache: layer   4: dev = CPU
llama_kv_cache: layer   5: filtered
llama_kv_cache: layer   6: dev = CPU
llama_kv_cache: layer   7: dev = CPU
llama_kv_cache: layer   8: dev = CPU
llama_kv_cache: layer   9: dev = CPU
llama_kv_cache: layer  10: dev = CPU
llama_kv_cache: layer  11: filtered
llama_kv_cache: layer  12: dev = CPU
llama_kv_cache: layer  13: dev = CPU
llama_kv_cache: layer  14: dev = CPU
llama_kv_cache: layer  15: dev = CPU
llama_kv_cache: layer  16: dev = CPU
llama_kv_cache: layer  17: filtered
llama_kv_cache: layer  18: dev = CPU
llama_kv_cache: layer  19: dev = CPU
llama_kv_cache: layer  20: dev = CPU
llama_kv_cache: layer  21: dev = CPU
llama_kv_cache: layer  22: dev = CPU
llama_kv_cache: layer  23: filtered
llama_kv_cache: layer  24: dev = CPU
llama_kv_cache: layer  25: dev = CPU
llama_kv_cache: layer  26: dev = CPU
llama_kv_cache: layer  27: dev = CPU
llama_kv_cache: layer  28: dev = CPU
llama_kv_cache: layer  29: filtered
llama_kv_cache: layer  30: dev = CPU
llama_kv_cache: layer  31: dev = CPU
llama_kv_cache: layer  32: dev = CPU
llama_kv_cache: layer  33: dev = CPU
llama_kv_cache: layer  34: dev = CPU
llama_kv_cache: layer  35: filtered
llama_kv_cache: layer  36: dev = CPU
llama_kv_cache: layer  37: dev = CPU
llama_kv_cache: layer  38: dev = CPU
llama_kv_cache: layer  39: dev = CPU
llama_kv_cache: layer  40: dev = CPU
llama_kv_cache: layer  41: filtered
llama_kv_cache: layer  42: dev = CPU
llama_kv_cache: layer  43: dev = CPU
llama_kv_cache: layer  44: dev = CPU
llama_kv_cache: layer  45: dev = CPU
llama_kv_cache: layer  46: dev = CPU
llama_kv_cache: layer  47: filtered
llama_kv_cache: reusing layers:
llama_kv_cache: - layer   0: no reuse
llama_kv_cache: - layer   1: no reuse
llama_kv_cache: - layer   2: no reuse
llama_kv_cache: - layer   3: no reuse
llama_kv_cache: - layer   4: no reuse
llama_kv_cache: - layer   5: no reuse
llama_kv_cache: - layer   6: no reuse
llama_kv_cache: - layer   7: no reuse
llama_kv_cache: - layer   8: no reuse
llama_kv_cache: - layer   9: no reuse
llama_kv_cache: - layer  10: no reuse
llama_kv_cache: - layer  11: no reuse
llama_kv_cache: - layer  12: no reuse
llama_kv_cache: - layer  13: no reuse
llama_kv_cache: - layer  14: no reuse
llama_kv_cache: - layer  15: no reuse
llama_kv_cache: - layer  16: no reuse
llama_kv_cache: - layer  17: no reuse
llama_kv_cache: - layer  18: no reuse
llama_kv_cache: - layer  19: no reuse
llama_kv_cache: - layer  20: no reuse
llama_kv_cache: - layer  21: no reuse
llama_kv_cache: - layer  22: no reuse
llama_kv_cache: - layer  23: no reuse
llama_kv_cache: - layer  24: no reuse
llama_kv_cache: - layer  25: no reuse
llama_kv_cache: - layer  26: no reuse
llama_kv_cache: - layer  27: no reuse
llama_kv_cache: - layer  28: no reuse
llama_kv_cache: - layer  29: no reuse
llama_kv_cache: - layer  30: no reuse
llama_kv_cache: - layer  31: no reuse
llama_kv_cache: - layer  32: no reuse
llama_kv_cache: - layer  33: no reuse
llama_kv_cache: - layer  34: no reuse
llama_kv_cache: - layer  35: no reuse
llama_kv_cache: - layer  36: no reuse
llama_kv_cache: - layer  37: no reuse
llama_kv_cache: - layer  38: no reuse
llama_kv_cache: - layer  39: no reuse
llama_kv_cache: - layer  40: no reuse
llama_kv_cache: - layer  41: no reuse
llama_kv_cache: - layer  42: no reuse
llama_kv_cache: - layer  43: no reuse
llama_kv_cache: - layer  44: no reuse
llama_kv_cache: - layer  45: no reuse
llama_kv_cache: - layer  46: no reuse
llama_kv_cache: - layer  47: no reuse
llama_kv_cache:        CPU KV buffer size =   640.00 MiB
llama_kv_cache: size =  640.00 MiB (  2048 cells,  40 layers,  1/1 seqs), K (f16):  320.00 MiB, V (f16):  320.00 MiB
llama_kv_cache: attn_rot_k = 0, n_embd_head_k_all = 256
llama_kv_cache: attn_rot_v = 0, n_embd_head_k_all = 256
llama_context: enumerating backends
llama_context: backend_ptrs.size() = 1
sched_reserve: reserving ...
sched_reserve: max_nodes = 5344
sched_reserve: reserving full memory module
sched_reserve: worst-case: n_tokens = 512, n_seqs = 1, n_outputs = 1
graph_reserve: reserving a graph for ubatch with n_tokens =    1, n_seqs =  1, n_outputs =    1
sched_reserve: Flash Attention was auto, set to enabled
sched_reserve: resolving fused Gated Delta Net support:
graph_reserve: reserving a graph for ubatch with n_tokens =    1, n_seqs =  1, n_outputs =    1
sched_reserve: fused Gated Delta Net (autoregressive) enabled
graph_reserve: reserving a graph for ubatch with n_tokens =   16, n_seqs =  1, n_outputs =   16
sched_reserve: fused Gated Delta Net (chunked) enabled
graph_reserve: reserving a graph for ubatch with n_tokens =  512, n_seqs =  1, n_outputs =  512
graph_reserve: reserving a graph for ubatch with n_tokens =    1, n_seqs =  1, n_outputs =    1
graph_reserve: reserving a graph for ubatch with n_tokens =  512, n_seqs =  1, n_outputs =  512
sched_reserve:        CPU compute buffer size =   527.00 MiB
sched_reserve: graph nodes  = 1972
sched_reserve: graph splits = 1
sched_reserve: reserve took 4.12 ms, sched copies = 1
set_embeddings: value = 1
[main]: MTP speculative decoding active, draft_block_size=3 (drafts 2 tokens/round)

[main]: n_ctx = 2048, n_threads = 8, gpu_layers = 999
[main]: flash-attn = auto (runtime-selected)
common_speculative_impl_draft_gemma4_mtp: adding speculative implementation 'draft-mtp' (gemma4 assistant)
common_speculative_impl_draft_gemma4_mtp: - draft_block_size=3, n_max=2, n_min=0, n_steps=2, n_embd_bb=3840
[main]: system_info: CPU : SSE3 = 1 | SSSE3 = 1 | AVX = 1 | AVX2 = 1 | F16C = 1 | FMA = 1 | AVX512 = 1 | LLAMAFILE = 1 | OPENMP = 1 | REPACK = 1 |

> Running with custom prompt => [1/18]: [This is Izzy's bedroom]
set_embeddings: value = 1
  prefill: 19174.6ms (555 tokens, 28.9 t/s)
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
 {
    "answer": "c. That is correct Mara. It is your sister's room",
    "justification": "Mara finally revealed the truth. Option (c) confirms the truth."
}
  [all token ids (45): 642 107 140 236775 14433 1083 623 236755 236761 2981 563 4338 63338 236761 1030 563 822 12198 236789 236751 2978 827 107 140 236775 4017 2540 1083 623 201883 6861 9937 506 9043 236761 19993 568 236755 236768 44898 506 9043 1781 107 236783]
  [output hex (158 bytes):]
    0000: 20 7b 0a 20 20 20 20 22 61 6e 73 77 65 72 22 3a  | {.    "answer":|
    0010: 20 22 63 2e 20 54 68 61 74 20 69 73 20 63 6f 72  | "c. That is cor|
    0020: 72 65 63 74 20 4d 61 72 61 2e 20 49 74 20 69 73  |rect Mara. It is|
    0030: 20 79 6f 75 72 20 73 69 73 74 65 72 27 73 20 72  | your sister's r|
    0040: 6f 6f 6d 22 2c 0a 20 20 20 20 22 6a 75 73 74 69  |oom",.    "justi|
    0050: 66 69 63 61 74 69 6f 6e 22 3a 20 22 4d 61 72 61  |fication": "Mara|
    0060: 20 66 69 6e 61 6c 6c 79 20 72 65 76 65 61 6c 65  | finally reveale|
    0070: 64 20 74 68 65 20 74 72 75 74 68 2e 20 4f 70 74  |d the truth. Opt|
    0080: 69 6f 6e 20 28 63 29 20 63 6f 6e 66 69 72 6d 73  |ion (c) confirms|
    0090: 20 74 68 65 20 74 72 75 74 68 2e 22 0a 7d        | the truth.".}|
  decode: 3748.5ms (45 tokens, 12.00 t/s)
  decode (no-TTFT): 3748.0ms (45 tokens, 12.01 t/s) [TTFT 0.5ms]

> Running with custom prompt => [2/18]: [This is bedroom for Izzy]
set_embeddings: value = 1
  prefill: 19615.6ms (554 tokens, 28.2 t/s)
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
set_embeddings: value = 1
 {
    "answer": "c. That is correct Mara. It is your sister's room",
    "justification": "Mara is willingly to answer it's her sister Izzy room. Option (c) is the correct answer."
}
  [all token ids (54): 642 107 140 236775 14433 1083 623 236755 236761 2981 563 4338 63338 236761 1030 563 822 12198 236789 236751 2978 827 107 140 236775 4017 2540 1083 623 201883 563 95826 531 3890 625 236789 236751 1116 12198 29689 2946 2978 236761 19993 568 236755 236768 563 506 4338 3890 1781 107 236783]
  [output hex (183 bytes):]
    0000: 20 7b 0a 20 20 20 20 22 61 6e 73 77 65 72 22 3a  | {.    "answer":|
    0010: 20 22 63 2e 20 54 68 61 74 20 69 73 20 63 6f 72  | "c. That is cor|
    0020: 72 65 63 74 20 4d 61 72 61 2e 20 49 74 20 69 73  |rect Mara. It is|
    0030: 20 79 6f 75 72 20 73 69 73 74 65 72 27 73 20 72  | your sister's r|
    0040: 6f 6f 6d 22 2c 0a 20 20 20 20 22 6a 75 73 74 69  |oom",.    "justi|
    0050: 66 69 63 61 74 69 6f 6e 22 3a 20 22 4d 61 72 61  |fication": "Mara|
    0060: 20 69 73 20 77 69 6c 6c 69 6e 67 6c 79 20 74 6f  | is willingly to|
    0070: 20 61 6e 73 77 65 72 20 69 74 27 73 20 68 65 72  | answer it's her|
    0080: 20 73 69 73 74 65 72 20 49 7a 7a 79 20 72 6f 6f  | sister Izzy roo|
    0090: 6d 2e 20 4f 70 74 69 6f 6e 20 28 63 29 20 69 73  |m. Option (c) is|
    00a0: 20 74 68 65 20 63 6f 72 72 65 63 74 20 61 6e 73  | the correct ans|
    00b0: 77 65 72 2e 22 0a 7d                             |wer.".}|
  decode: 4913.5ms (54 tokens, 10.99 t/s)
  decode (no-TTFT): 4912.9ms (54 tokens, 10.99 t/s) [TTFT 0.5ms]

> Running with custom prompt => [3/18]: [This is the bedroom for my sister Izzy]
set_embeddings: value = 1

============================================================================

C:\llama.cpp\b612_052026_mtp\build>bin\RelWithDebInfo\minslm-cli.exe C:\llama.cpp\models\Gemma-4\12B\gemma-4-12b-it-qat-q4_0.gguf 8 ..\examples\llm-infer\prompts\custom_prompts_gemma4.txt --mtp-head c:\llama.cpp\models\Gemma-4\12B\gemma-4-12B-it-qat-assistant-MTP-Q8_0.gguf  --spec-type mtp --draft-block-size 3 v2
Warning: unknown argument '--spec-type' ignored
Warning: unknown argument 'mtp' ignored
[main]: loaded 18 prompt(s) from '..\examples\llm-infer\prompts\custom_prompts_gemma4.txt'
llama_model_loader: direct I/O is enabled, disabling mmap
llama_model_loader: loaded meta data with 44 key-value pairs and 667 tensors from C:\llama.cpp\models\Gemma-4\12B\gemma-4-12b-it-qat-q4_0.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = gemma4
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                     general.sampling.top_k i32              = 64
llama_model_loader: - kv   3:                     general.sampling.top_p f32              = 0.950000
llama_model_loader: - kv   4:                      general.sampling.temp f32              = 1.000000
llama_model_loader: - kv   5:                               general.name str              = 12B_qat_it_dequant_safetensors
llama_model_loader: - kv   6:                           general.finetune str              = 12B_qat_it_dequant_safetensors
llama_model_loader: - kv   7:                         general.size_label str              = 12B
llama_model_loader: - kv   8:                         gemma4.block_count u32              = 48
llama_model_loader: - kv   9:                      gemma4.context_length u32              = 262144
llama_model_loader: - kv  10:                    gemma4.embedding_length u32              = 3840
llama_model_loader: - kv  11:                 gemma4.feed_forward_length u32              = 15360
llama_model_loader: - kv  12:                gemma4.attention.head_count u32              = 16
llama_model_loader: - kv  13:             gemma4.attention.head_count_kv arr[i32,48]      = [8, 8, 8, 8, 8, 1, 8, 8, 8, 8, 8, 1, ...
llama_model_loader: - kv  14:                      gemma4.rope.freq_base f32              = 1000000.000000
llama_model_loader: - kv  15:                  gemma4.rope.freq_base_swa f32              = 10000.000000
llama_model_loader: - kv  16:    gemma4.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  17:                gemma4.attention.key_length u32              = 512
llama_model_loader: - kv  18:              gemma4.attention.value_length u32              = 512
llama_model_loader: - kv  19:             gemma4.final_logit_softcapping f32              = 30.000000
llama_model_loader: - kv  20:            gemma4.attention.sliding_window u32              = 1024
llama_model_loader: - kv  21:          gemma4.attention.shared_kv_layers u32              = 0
llama_model_loader: - kv  22:    gemma4.embedding_length_per_layer_input u32              = 0
llama_model_loader: - kv  23:    gemma4.attention.sliding_window_pattern arr[bool,48]     = [true, true, true, true, true, false,...
llama_model_loader: - kv  24:            gemma4.attention.key_length_swa u32              = 256
llama_model_loader: - kv  25:          gemma4.attention.value_length_swa u32              = 256
llama_model_loader: - kv  26:                gemma4.rope.dimension_count u32              = 512
llama_model_loader: - kv  27:            gemma4.rope.dimension_count_swa u32              = 256
llama_model_loader: - kv  28:             tokenizer.ggml.suppress_tokens arr[i32,2]       = [258883, 258882]
llama_model_loader: - kv  29:                       tokenizer.ggml.model str              = gemma4
llama_model_loader: - kv  30:                      tokenizer.ggml.tokens arr[str,262144]  = ["<pad>", "<eos>", "<bos>", "<unk>", ...
llama_model_loader: - kv  31:                      tokenizer.ggml.scores arr[f32,262144]  = [-1000.000000, -1000.000000, -1000.00...
llama_model_loader: - kv  32:                  tokenizer.ggml.token_type arr[i32,262144]  = [3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  33:                      tokenizer.ggml.merges arr[str,514906]  = ["\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n \n", ...
llama_model_loader: - kv  34:                tokenizer.ggml.bos_token_id u32              = 2
llama_model_loader: - kv  35:                tokenizer.ggml.eos_token_id u32              = 1
llama_model_loader: - kv  36:            tokenizer.ggml.unknown_token_id u32              = 3
llama_model_loader: - kv  37:            tokenizer.ggml.padding_token_id u32              = 0
llama_model_loader: - kv  38:               tokenizer.ggml.mask_token_id u32              = 4
llama_model_loader: - kv  39:                    tokenizer.chat_template str              = {%- macro format_parameters(propertie...
llama_model_loader: - kv  40:            tokenizer.ggml.add_space_prefix bool             = false
llama_model_loader: - kv  41:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  42:               general.quantization_version u32              = 2
llama_model_loader: - kv  43:                          general.file_type u32              = 2
llama_model_loader: - type  f32:  338 tensors
llama_model_loader: - type q4_0:  328 tensors
llama_model_loader: - type q6_K:    1 tensors
print_info: file format = GGUF V3 (latest)
print_info: file type   = Q4_0
print_info: file size   = 6.48 GiB (4.68 BPW)
init_tokenizer: initializing tokenizer for type 2
load: 0 unused tokens
load: control token:      0 '<pad>' is not marked as EOG
load: control token:      3 '<unk>' is not marked as EOG
load: control token:      2 '<bos>' is not marked as EOG
load: control token:      4 '<mask>' is not marked as EOG
load: control token:     47 '<tool|>' is not marked as EOG
load: control token:     98 '<|think|>' is not marked as EOG
load: control token:     46 '<|tool>' is not marked as EOG
load: control-looking token:     50 '<|tool_response>' was not control-type; this is probably a bug in the model. its type will be overridden
load: control token:    105 '<|turn>' is not marked as EOG
load: control-looking token:    212 '</s>' was not control-type; this is probably a bug in the model. its type will be overridden
load: control token: 258880 '<|image|>' is not marked as EOG
load: control token: 258882 '<image|>' is not marked as EOG
load: control token: 256000 '<|audio>' is not marked as EOG
load: control token: 258883 '<audio|>' is not marked as EOG
load: control token: 258884 '<|video|>' is not marked as EOG
load: control token: 255999 '<|image>' is not marked as EOG
load: control token: 258881 '<|audio|>' is not marked as EOG
load: printing all EOG tokens:
load:   - 1 ('<eos>')
load:   - 50 ('<|tool_response>')
load:   - 106 ('<turn|>')
load:   - 212 ('</s>')
load: special_eog_ids contains '<|tool_response>', removing '</s>' token from EOG list
load: special tokens cache size = 24
load: token to piece cache size = 1.9445 MB
print_info: arch                  = gemma4
print_info: vocab_only            = 0
print_info: no_alloc              = 0
print_info: n_ctx_train           = 262144
print_info: n_embd                = 3840
print_info: n_embd_inp            = 3840
print_info: n_layer               = 48
print_info: n_head                = 16
print_info: n_head_kv             = [8, 8, 8, 8, 8, 1, 8, 8, 8, 8, 8, 1, 8, 8, 8, 8, 8, 1, 8, 8, 8, 8, 8, 1, 8, 8, 8, 8, 8, 1, 8, 8, 8, 8, 8, 1, 8, 8, 8, 8, 8, 1, 8, 8, 8, 8, 8, 1]
print_info: n_rot                 = 512
print_info: n_swa                 = 1024
print_info: is_swa_any            = 1
print_info: n_embd_head_k         = 512
print_info: n_embd_head_v         = 512
print_info: n_gqa                 = [2, 2, 2, 2, 2, 16, 2, 2, 2, 2, 2, 16, 2, 2, 2, 2, 2, 16, 2, 2, 2, 2, 2, 16, 2, 2, 2, 2, 2, 16, 2, 2, 2, 2, 2, 16, 2, 2, 2, 2, 2, 16, 2, 2, 2, 2, 2, 16]
print_info: n_embd_k_gqa          = [2048, 2048, 2048, 2048, 2048, 512, 2048, 2048, 2048, 2048, 2048, 512, 2048, 2048, 2048, 2048, 2048, 512, 2048, 2048, 2048, 2048, 2048, 512, 2048, 2048, 2048, 2048, 2048, 512, 2048, 2048, 2048, 2048, 2048, 512, 2048, 2048, 2048, 2048, 2048, 512, 2048, 2048, 2048, 2048, 2048, 512]
print_info: n_embd_v_gqa          = [2048, 2048, 2048, 2048, 2048, 512, 2048, 2048, 2048, 2048, 2048, 512, 2048, 2048, 2048, 2048, 2048, 512, 2048, 2048, 2048, 2048, 2048, 512, 2048, 2048, 2048, 2048, 2048, 512, 2048, 2048, 2048, 2048, 2048, 512, 2048, 2048, 2048, 2048, 2048, 512, 2048, 2048, 2048, 2048, 2048, 512]
print_info: f_norm_eps            = 0.0e+00
print_info: f_norm_rms_eps        = 1.0e-06
print_info: f_clamp_kqv           = 0.0e+00
print_info: f_max_alibi_bias      = 0.0e+00
print_info: f_logit_scale         = 0.0e+00
print_info: f_attn_scale          = 1.0e+00
print_info: f_attn_value_scale    = 0.0000
print_info: n_ff                  = 15360
print_info: n_expert              = 0
print_info: n_expert_used         = 0
print_info: n_expert_groups       = 0
print_info: n_group_used          = 0
print_info: causal attn           = 1
print_info: pooling type          = -1
print_info: rope type             = 2
print_info: rope scaling          = linear
print_info: freq_base_train       = 1000000.0
print_info: freq_scale_train      = 1
print_info: freq_base_swa         = 10000.0
print_info: freq_scale_swa        = 1
print_info: n_embd_head_k_swa     = 256
print_info: n_embd_head_v_swa     = 256
print_info: n_rot_swa             = 256
print_info: n_ctx_orig_yarn       = 262144
print_info: rope_yarn_log_mul     = 0.0000
print_info: rope_finetuned        = unknown
print_info: model type            = ?B
print_info: model params          = 11.91 B
print_info: general.name          = 12B_qat_it_dequant_safetensors
print_info: vocab type            = BPE
print_info: n_vocab               = 262144
print_info: n_merges              = 514906
print_info: BOS token             = 2 '<bos>'
print_info: EOS token             = 1 '<eos>'
print_info: UNK token             = 3 '<unk>'
print_info: PAD token             = 0 '<pad>'
print_info: MASK token            = 4 '<mask>'
print_info: LF token              = 107 '
'
print_info: EOG token             = 1 '<eos>'
print_info: EOG token             = 50 '<|tool_response>'
print_info: EOG token             = 106 '<turn|>'
print_info: max token length      = 93
load_tensors: loading model tensors, this can take a while... (mmap = false, direct_io = true)
load_tensors: layer   0 assigned to device CPU, is_swa = 1
load_tensors: layer   1 assigned to device CPU, is_swa = 1
load_tensors: layer   2 assigned to device CPU, is_swa = 1
load_tensors: layer   3 assigned to device CPU, is_swa = 1
load_tensors: layer   4 assigned to device CPU, is_swa = 1
load_tensors: layer   5 assigned to device CPU, is_swa = 0
load_tensors: layer   6 assigned to device CPU, is_swa = 1
load_tensors: layer   7 assigned to device CPU, is_swa = 1
load_tensors: layer   8 assigned to device CPU, is_swa = 1
load_tensors: layer   9 assigned to device CPU, is_swa = 1
load_tensors: layer  10 assigned to device CPU, is_swa = 1
load_tensors: layer  11 assigned to device CPU, is_swa = 0
load_tensors: layer  12 assigned to device CPU, is_swa = 1
load_tensors: layer  13 assigned to device CPU, is_swa = 1
load_tensors: layer  14 assigned to device CPU, is_swa = 1
load_tensors: layer  15 assigned to device CPU, is_swa = 1
load_tensors: layer  16 assigned to device CPU, is_swa = 1
load_tensors: layer  17 assigned to device CPU, is_swa = 0
load_tensors: layer  18 assigned to device CPU, is_swa = 1
load_tensors: layer  19 assigned to device CPU, is_swa = 1
load_tensors: layer  20 assigned to device CPU, is_swa = 1
load_tensors: layer  21 assigned to device CPU, is_swa = 1
load_tensors: layer  22 assigned to device CPU, is_swa = 1
load_tensors: layer  23 assigned to device CPU, is_swa = 0
load_tensors: layer  24 assigned to device CPU, is_swa = 1
load_tensors: layer  25 assigned to device CPU, is_swa = 1
load_tensors: layer  26 assigned to device CPU, is_swa = 1
load_tensors: layer  27 assigned to device CPU, is_swa = 1
load_tensors: layer  28 assigned to device CPU, is_swa = 1
load_tensors: layer  29 assigned to device CPU, is_swa = 0
load_tensors: layer  30 assigned to device CPU, is_swa = 1
load_tensors: layer  31 assigned to device CPU, is_swa = 1
load_tensors: layer  32 assigned to device CPU, is_swa = 1
load_tensors: layer  33 assigned to device CPU, is_swa = 1
load_tensors: layer  34 assigned to device CPU, is_swa = 1
load_tensors: layer  35 assigned to device CPU, is_swa = 0
load_tensors: layer  36 assigned to device CPU, is_swa = 1
load_tensors: layer  37 assigned to device CPU, is_swa = 1
load_tensors: layer  38 assigned to device CPU, is_swa = 1
load_tensors: layer  39 assigned to device CPU, is_swa = 1
load_tensors: layer  40 assigned to device CPU, is_swa = 1
load_tensors: layer  41 assigned to device CPU, is_swa = 0
load_tensors: layer  42 assigned to device CPU, is_swa = 1
load_tensors: layer  43 assigned to device CPU, is_swa = 1
load_tensors: layer  44 assigned to device CPU, is_swa = 1
load_tensors: layer  45 assigned to device CPU, is_swa = 1
load_tensors: layer  46 assigned to device CPU, is_swa = 1
load_tensors: layer  47 assigned to device CPU, is_swa = 0
load_tensors: layer  48 assigned to device CPU, is_swa = 0
create_tensor: loading tensor token_embd.weight
create_tensor: loading tensor token_embd.weight
create_tensor: loading tensor output_norm.weight
create_tensor: loading tensor blk.0.attn_norm.weight
create_tensor: loading tensor blk.0.attn_q.weight
create_tensor: loading tensor blk.0.attn_k.weight
create_tensor: loading tensor blk.0.attn_v.weight
create_tensor: loading tensor blk.0.attn_output.weight
create_tensor: loading tensor blk.0.attn_q_norm.weight
create_tensor: loading tensor blk.0.attn_k_norm.weight
create_tensor: loading tensor blk.0.post_attention_norm.weight
create_tensor: loading tensor blk.0.layer_output_scale.weight
create_tensor: loading tensor blk.0.ffn_norm.weight
create_tensor: loading tensor blk.0.ffn_gate.weight
create_tensor: loading tensor blk.0.ffn_up.weight
create_tensor: loading tensor blk.0.ffn_down.weight
create_tensor: loading tensor blk.0.post_ffw_norm.weight
create_tensor: loading tensor blk.1.attn_norm.weight
create_tensor: loading tensor blk.1.attn_q.weight
create_tensor: loading tensor blk.1.attn_k.weight
create_tensor: loading tensor blk.1.attn_v.weight
create_tensor: loading tensor blk.1.attn_output.weight
create_tensor: loading tensor blk.1.attn_q_norm.weight
create_tensor: loading tensor blk.1.attn_k_norm.weight
create_tensor: loading tensor blk.1.post_attention_norm.weight
create_tensor: loading tensor blk.1.layer_output_scale.weight
create_tensor: loading tensor blk.1.ffn_norm.weight
create_tensor: loading tensor blk.1.ffn_gate.weight
create_tensor: loading tensor blk.1.ffn_up.weight
create_tensor: loading tensor blk.1.ffn_down.weight
create_tensor: loading tensor blk.1.post_ffw_norm.weight
create_tensor: loading tensor blk.2.attn_norm.weight
create_tensor: loading tensor blk.2.attn_q.weight
create_tensor: loading tensor blk.2.attn_k.weight
create_tensor: loading tensor blk.2.attn_v.weight
create_tensor: loading tensor blk.2.attn_output.weight
create_tensor: loading tensor blk.2.attn_q_norm.weight
create_tensor: loading tensor blk.2.attn_k_norm.weight
create_tensor: loading tensor blk.2.post_attention_norm.weight
create_tensor: loading tensor blk.2.layer_output_scale.weight
create_tensor: loading tensor blk.2.ffn_norm.weight
create_tensor: loading tensor blk.2.ffn_gate.weight
create_tensor: loading tensor blk.2.ffn_up.weight
create_tensor: loading tensor blk.2.ffn_down.weight
create_tensor: loading tensor blk.2.post_ffw_norm.weight
create_tensor: loading tensor blk.3.attn_norm.weight
create_tensor: loading tensor blk.3.attn_q.weight
create_tensor: loading tensor blk.3.attn_k.weight
create_tensor: loading tensor blk.3.attn_v.weight
create_tensor: loading tensor blk.3.attn_output.weight
create_tensor: loading tensor blk.3.attn_q_norm.weight
create_tensor: loading tensor blk.3.attn_k_norm.weight
create_tensor: loading tensor blk.3.post_attention_norm.weight
create_tensor: loading tensor blk.3.layer_output_scale.weight
create_tensor: loading tensor blk.3.ffn_norm.weight
create_tensor: loading tensor blk.3.ffn_gate.weight
create_tensor: loading tensor blk.3.ffn_up.weight
create_tensor: loading tensor blk.3.ffn_down.weight
create_tensor: loading tensor blk.3.post_ffw_norm.weight
create_tensor: loading tensor blk.4.attn_norm.weight
create_tensor: loading tensor blk.4.attn_q.weight
create_tensor: loading tensor blk.4.attn_k.weight
create_tensor: loading tensor blk.4.attn_v.weight
create_tensor: loading tensor blk.4.attn_output.weight
create_tensor: loading tensor blk.4.attn_q_norm.weight
create_tensor: loading tensor blk.4.attn_k_norm.weight
create_tensor: loading tensor blk.4.post_attention_norm.weight
create_tensor: loading tensor blk.4.layer_output_scale.weight
create_tensor: loading tensor blk.4.ffn_norm.weight
create_tensor: loading tensor blk.4.ffn_gate.weight
create_tensor: loading tensor blk.4.ffn_up.weight
create_tensor: loading tensor blk.4.ffn_down.weight
create_tensor: loading tensor blk.4.post_ffw_norm.weight
create_tensor: loading tensor blk.5.attn_norm.weight
create_tensor: loading tensor blk.5.attn_q.weight
create_tensor: loading tensor blk.5.attn_k.weight
create_tensor: loading tensor blk.5.attn_output.weight
create_tensor: loading tensor blk.5.attn_q_norm.weight
create_tensor: loading tensor blk.5.attn_k_norm.weight
create_tensor: loading tensor blk.5.post_attention_norm.weight
create_tensor: loading tensor blk.5.layer_output_scale.weight
create_tensor: loading tensor rope_freqs.weight
create_tensor: loading tensor blk.5.ffn_norm.weight
create_tensor: loading tensor blk.5.ffn_gate.weight
create_tensor: loading tensor blk.5.ffn_up.weight
create_tensor: loading tensor blk.5.ffn_down.weight
create_tensor: loading tensor blk.5.post_ffw_norm.weight
create_tensor: loading tensor blk.6.attn_norm.weight
create_tensor: loading tensor blk.6.attn_q.weight
create_tensor: loading tensor blk.6.attn_k.weight
create_tensor: loading tensor blk.6.attn_v.weight
create_tensor: loading tensor blk.6.attn_output.weight
create_tensor: loading tensor blk.6.attn_q_norm.weight
create_tensor: loading tensor blk.6.attn_k_norm.weight
create_tensor: loading tensor blk.6.post_attention_norm.weight
create_tensor: loading tensor blk.6.layer_output_scale.weight
create_tensor: loading tensor blk.6.ffn_norm.weight
create_tensor: loading tensor blk.6.ffn_gate.weight
create_tensor: loading tensor blk.6.ffn_up.weight
create_tensor: loading tensor blk.6.ffn_down.weight
create_tensor: loading tensor blk.6.post_ffw_norm.weight
create_tensor: loading tensor blk.7.attn_norm.weight
create_tensor: loading tensor blk.7.attn_q.weight
create_tensor: loading tensor blk.7.attn_k.weight
create_tensor: loading tensor blk.7.attn_v.weight
create_tensor: loading tensor blk.7.attn_output.weight
create_tensor: loading tensor blk.7.attn_q_norm.weight
create_tensor: loading tensor blk.7.attn_k_norm.weight
create_tensor: loading tensor blk.7.post_attention_norm.weight
create_tensor: loading tensor blk.7.layer_output_scale.weight
create_tensor: loading tensor blk.7.ffn_norm.weight
create_tensor: loading tensor blk.7.ffn_gate.weight
create_tensor: loading tensor blk.7.ffn_up.weight
create_tensor: loading tensor blk.7.ffn_down.weight
create_tensor: loading tensor blk.7.post_ffw_norm.weight
create_tensor: loading tensor blk.8.attn_norm.weight
create_tensor: loading tensor blk.8.attn_q.weight
create_tensor: loading tensor blk.8.attn_k.weight
create_tensor: loading tensor blk.8.attn_v.weight
create_tensor: loading tensor blk.8.attn_output.weight
create_tensor: loading tensor blk.8.attn_q_norm.weight
create_tensor: loading tensor blk.8.attn_k_norm.weight
create_tensor: loading tensor blk.8.post_attention_norm.weight
create_tensor: loading tensor blk.8.layer_output_scale.weight
create_tensor: loading tensor blk.8.ffn_norm.weight
create_tensor: loading tensor blk.8.ffn_gate.weight
create_tensor: loading tensor blk.8.ffn_up.weight
create_tensor: loading tensor blk.8.ffn_down.weight
create_tensor: loading tensor blk.8.post_ffw_norm.weight
create_tensor: loading tensor blk.9.attn_norm.weight
create_tensor: loading tensor blk.9.attn_q.weight
create_tensor: loading tensor blk.9.attn_k.weight
create_tensor: loading tensor blk.9.attn_v.weight
create_tensor: loading tensor blk.9.attn_output.weight
create_tensor: loading tensor blk.9.attn_q_norm.weight
create_tensor: loading tensor blk.9.attn_k_norm.weight
create_tensor: loading tensor blk.9.post_attention_norm.weight
create_tensor: loading tensor blk.9.layer_output_scale.weight
create_tensor: loading tensor blk.9.ffn_norm.weight
create_tensor: loading tensor blk.9.ffn_gate.weight
create_tensor: loading tensor blk.9.ffn_up.weight
create_tensor: loading tensor blk.9.ffn_down.weight
create_tensor: loading tensor blk.9.post_ffw_norm.weight
create_tensor: loading tensor blk.10.attn_norm.weight
create_tensor: loading tensor blk.10.attn_q.weight
create_tensor: loading tensor blk.10.attn_k.weight
create_tensor: loading tensor blk.10.attn_v.weight
create_tensor: loading tensor blk.10.attn_output.weight
create_tensor: loading tensor blk.10.attn_q_norm.weight
create_tensor: loading tensor blk.10.attn_k_norm.weight
create_tensor: loading tensor blk.10.post_attention_norm.weight
create_tensor: loading tensor blk.10.layer_output_scale.weight
create_tensor: loading tensor blk.10.ffn_norm.weight
create_tensor: loading tensor blk.10.ffn_gate.weight
create_tensor: loading tensor blk.10.ffn_up.weight
create_tensor: loading tensor blk.10.ffn_down.weight
create_tensor: loading tensor blk.10.post_ffw_norm.weight
create_tensor: loading tensor blk.11.attn_norm.weight
create_tensor: loading tensor blk.11.attn_q.weight
create_tensor: loading tensor blk.11.attn_k.weight
create_tensor: loading tensor blk.11.attn_output.weight
create_tensor: loading tensor blk.11.attn_q_norm.weight
create_tensor: loading tensor blk.11.attn_k_norm.weight
create_tensor: loading tensor blk.11.post_attention_norm.weight
create_tensor: loading tensor blk.11.layer_output_scale.weight
create_tensor: loading tensor blk.11.ffn_norm.weight
create_tensor: loading tensor blk.11.ffn_gate.weight
create_tensor: loading tensor blk.11.ffn_up.weight
create_tensor: loading tensor blk.11.ffn_down.weight
create_tensor: loading tensor blk.11.post_ffw_norm.weight
create_tensor: loading tensor blk.12.attn_norm.weight
create_tensor: loading tensor blk.12.attn_q.weight
create_tensor: loading tensor blk.12.attn_k.weight
create_tensor: loading tensor blk.12.attn_v.weight
create_tensor: loading tensor blk.12.attn_output.weight
create_tensor: loading tensor blk.12.attn_q_norm.weight
create_tensor: loading tensor blk.12.attn_k_norm.weight
create_tensor: loading tensor blk.12.post_attention_norm.weight
create_tensor: loading tensor blk.12.layer_output_scale.weight
create_tensor: loading tensor blk.12.ffn_norm.weight
create_tensor: loading tensor blk.12.ffn_gate.weight
create_tensor: loading tensor blk.12.ffn_up.weight
create_tensor: loading tensor blk.12.ffn_down.weight
create_tensor: loading tensor blk.12.post_ffw_norm.weight
create_tensor: loading tensor blk.13.attn_norm.weight
create_tensor: loading tensor blk.13.attn_q.weight
create_tensor: loading tensor blk.13.attn_k.weight
create_tensor: loading tensor blk.13.attn_v.weight
create_tensor: loading tensor blk.13.attn_output.weight
create_tensor: loading tensor blk.13.attn_q_norm.weight
create_tensor: loading tensor blk.13.attn_k_norm.weight
create_tensor: loading tensor blk.13.post_attention_norm.weight
create_tensor: loading tensor blk.13.layer_output_scale.weight
create_tensor: loading tensor blk.13.ffn_norm.weight
create_tensor: loading tensor blk.13.ffn_gate.weight
create_tensor: loading tensor blk.13.ffn_up.weight
create_tensor: loading tensor blk.13.ffn_down.weight
create_tensor: loading tensor blk.13.post_ffw_norm.weight
create_tensor: loading tensor blk.14.attn_norm.weight
create_tensor: loading tensor blk.14.attn_q.weight
create_tensor: loading tensor blk.14.attn_k.weight
create_tensor: loading tensor blk.14.attn_v.weight
create_tensor: loading tensor blk.14.attn_output.weight
create_tensor: loading tensor blk.14.attn_q_norm.weight
create_tensor: loading tensor blk.14.attn_k_norm.weight
create_tensor: loading tensor blk.14.post_attention_norm.weight
create_tensor: loading tensor blk.14.layer_output_scale.weight
create_tensor: loading tensor blk.14.ffn_norm.weight
create_tensor: loading tensor blk.14.ffn_gate.weight
create_tensor: loading tensor blk.14.ffn_up.weight
create_tensor: loading tensor blk.14.ffn_down.weight
create_tensor: loading tensor blk.14.post_ffw_norm.weight
create_tensor: loading tensor blk.15.attn_norm.weight
create_tensor: loading tensor blk.15.attn_q.weight
create_tensor: loading tensor blk.15.attn_k.weight
create_tensor: loading tensor blk.15.attn_v.weight
create_tensor: loading tensor blk.15.attn_output.weight
create_tensor: loading tensor blk.15.attn_q_norm.weight
create_tensor: loading tensor blk.15.attn_k_norm.weight
create_tensor: loading tensor blk.15.post_attention_norm.weight
create_tensor: loading tensor blk.15.layer_output_scale.weight
create_tensor: loading tensor blk.15.ffn_norm.weight
create_tensor: loading tensor blk.15.ffn_gate.weight
create_tensor: loading tensor blk.15.ffn_up.weight
create_tensor: loading tensor blk.15.ffn_down.weight
create_tensor: loading tensor blk.15.post_ffw_norm.weight
create_tensor: loading tensor blk.16.attn_norm.weight
create_tensor: loading tensor blk.16.attn_q.weight
create_tensor: loading tensor blk.16.attn_k.weight
create_tensor: loading tensor blk.16.attn_v.weight
create_tensor: loading tensor blk.16.attn_output.weight
create_tensor: loading tensor blk.16.attn_q_norm.weight
create_tensor: loading tensor blk.16.attn_k_norm.weight
create_tensor: loading tensor blk.16.post_attention_norm.weight
create_tensor: loading tensor blk.16.layer_output_scale.weight
create_tensor: loading tensor blk.16.ffn_norm.weight
create_tensor: loading tensor blk.16.ffn_gate.weight
create_tensor: loading tensor blk.16.ffn_up.weight
create_tensor: loading tensor blk.16.ffn_down.weight
create_tensor: loading tensor blk.16.post_ffw_norm.weight
create_tensor: loading tensor blk.17.attn_norm.weight
create_tensor: loading tensor blk.17.attn_q.weight
create_tensor: loading tensor blk.17.attn_k.weight
create_tensor: loading tensor blk.17.attn_output.weight
create_tensor: loading tensor blk.17.attn_q_norm.weight
create_tensor: loading tensor blk.17.attn_k_norm.weight
create_tensor: loading tensor blk.17.post_attention_norm.weight
create_tensor: loading tensor blk.17.layer_output_scale.weight
create_tensor: loading tensor blk.17.ffn_norm.weight
create_tensor: loading tensor blk.17.ffn_gate.weight
create_tensor: loading tensor blk.17.ffn_up.weight
create_tensor: loading tensor blk.17.ffn_down.weight
create_tensor: loading tensor blk.17.post_ffw_norm.weight
create_tensor: loading tensor blk.18.attn_norm.weight
create_tensor: loading tensor blk.18.attn_q.weight
create_tensor: loading tensor blk.18.attn_k.weight
create_tensor: loading tensor blk.18.attn_v.weight
create_tensor: loading tensor blk.18.attn_output.weight
create_tensor: loading tensor blk.18.attn_q_norm.weight
create_tensor: loading tensor blk.18.attn_k_norm.weight
create_tensor: loading tensor blk.18.post_attention_norm.weight
create_tensor: loading tensor blk.18.layer_output_scale.weight
create_tensor: loading tensor blk.18.ffn_norm.weight
create_tensor: loading tensor blk.18.ffn_gate.weight
create_tensor: loading tensor blk.18.ffn_up.weight
create_tensor: loading tensor blk.18.ffn_down.weight
create_tensor: loading tensor blk.18.post_ffw_norm.weight
create_tensor: loading tensor blk.19.attn_norm.weight
create_tensor: loading tensor blk.19.attn_q.weight
create_tensor: loading tensor blk.19.attn_k.weight
create_tensor: loading tensor blk.19.attn_v.weight
create_tensor: loading tensor blk.19.attn_output.weight
create_tensor: loading tensor blk.19.attn_q_norm.weight
create_tensor: loading tensor blk.19.attn_k_norm.weight
create_tensor: loading tensor blk.19.post_attention_norm.weight
create_tensor: loading tensor blk.19.layer_output_scale.weight
create_tensor: loading tensor blk.19.ffn_norm.weight
create_tensor: loading tensor blk.19.ffn_gate.weight
create_tensor: loading tensor blk.19.ffn_up.weight
create_tensor: loading tensor blk.19.ffn_down.weight
create_tensor: loading tensor blk.19.post_ffw_norm.weight
create_tensor: loading tensor blk.20.attn_norm.weight
create_tensor: loading tensor blk.20.attn_q.weight
create_tensor: loading tensor blk.20.attn_k.weight
create_tensor: loading tensor blk.20.attn_v.weight
create_tensor: loading tensor blk.20.attn_output.weight
create_tensor: loading tensor blk.20.attn_q_norm.weight
create_tensor: loading tensor blk.20.attn_k_norm.weight
create_tensor: loading tensor blk.20.post_attention_norm.weight
create_tensor: loading tensor blk.20.layer_output_scale.weight
create_tensor: loading tensor blk.20.ffn_norm.weight
create_tensor: loading tensor blk.20.ffn_gate.weight
create_tensor: loading tensor blk.20.ffn_up.weight
create_tensor: loading tensor blk.20.ffn_down.weight
create_tensor: loading tensor blk.20.post_ffw_norm.weight
create_tensor: loading tensor blk.21.attn_norm.weight
create_tensor: loading tensor blk.21.attn_q.weight
create_tensor: loading tensor blk.21.attn_k.weight
create_tensor: loading tensor blk.21.attn_v.weight
create_tensor: loading tensor blk.21.attn_output.weight
create_tensor: loading tensor blk.21.attn_q_norm.weight
create_tensor: loading tensor blk.21.attn_k_norm.weight
create_tensor: loading tensor blk.21.post_attention_norm.weight
create_tensor: loading tensor blk.21.layer_output_scale.weight
create_tensor: loading tensor blk.21.ffn_norm.weight
create_tensor: loading tensor blk.21.ffn_gate.weight
create_tensor: loading tensor blk.21.ffn_up.weight
create_tensor: loading tensor blk.21.ffn_down.weight
create_tensor: loading tensor blk.21.post_ffw_norm.weight
create_tensor: loading tensor blk.22.attn_norm.weight
create_tensor: loading tensor blk.22.attn_q.weight
create_tensor: loading tensor blk.22.attn_k.weight
create_tensor: loading tensor blk.22.attn_v.weight
create_tensor: loading tensor blk.22.attn_output.weight
create_tensor: loading tensor blk.22.attn_q_norm.weight
create_tensor: loading tensor blk.22.attn_k_norm.weight
create_tensor: loading tensor blk.22.post_attention_norm.weight
create_tensor: loading tensor blk.22.layer_output_scale.weight
create_tensor: loading tensor blk.22.ffn_norm.weight
create_tensor: loading tensor blk.22.ffn_gate.weight
create_tensor: loading tensor blk.22.ffn_up.weight
create_tensor: loading tensor blk.22.ffn_down.weight
create_tensor: loading tensor blk.22.post_ffw_norm.weight
create_tensor: loading tensor blk.23.attn_norm.weight
create_tensor: loading tensor blk.23.attn_q.weight
create_tensor: loading tensor blk.23.attn_k.weight
create_tensor: loading tensor blk.23.attn_output.weight
create_tensor: loading tensor blk.23.attn_q_norm.weight
create_tensor: loading tensor blk.23.attn_k_norm.weight
create_tensor: loading tensor blk.23.post_attention_norm.weight
create_tensor: loading tensor blk.23.layer_output_scale.weight
create_tensor: loading tensor blk.23.ffn_norm.weight
create_tensor: loading tensor blk.23.ffn_gate.weight
create_tensor: loading tensor blk.23.ffn_up.weight
create_tensor: loading tensor blk.23.ffn_down.weight
create_tensor: loading tensor blk.23.post_ffw_norm.weight
create_tensor: loading tensor blk.24.attn_norm.weight
create_tensor: loading tensor blk.24.attn_q.weight
create_tensor: loading tensor blk.24.attn_k.weight
create_tensor: loading tensor blk.24.attn_v.weight
create_tensor: loading tensor blk.24.attn_output.weight
create_tensor: loading tensor blk.24.attn_q_norm.weight
create_tensor: loading tensor blk.24.attn_k_norm.weight
create_tensor: loading tensor blk.24.post_attention_norm.weight
create_tensor: loading tensor blk.24.layer_output_scale.weight
create_tensor: loading tensor blk.24.ffn_norm.weight
create_tensor: loading tensor blk.24.ffn_gate.weight
create_tensor: loading tensor blk.24.ffn_up.weight
create_tensor: loading tensor blk.24.ffn_down.weight
create_tensor: loading tensor blk.24.post_ffw_norm.weight
create_tensor: loading tensor blk.25.attn_norm.weight
create_tensor: loading tensor blk.25.attn_q.weight
create_tensor: loading tensor blk.25.attn_k.weight
create_tensor: loading tensor blk.25.attn_v.weight
create_tensor: loading tensor blk.25.attn_output.weight
create_tensor: loading tensor blk.25.attn_q_norm.weight
create_tensor: loading tensor blk.25.attn_k_norm.weight
create_tensor: loading tensor blk.25.post_attention_norm.weight
create_tensor: loading tensor blk.25.layer_output_scale.weight
create_tensor: loading tensor blk.25.ffn_norm.weight
create_tensor: loading tensor blk.25.ffn_gate.weight
create_tensor: loading tensor blk.25.ffn_up.weight
create_tensor: loading tensor blk.25.ffn_down.weight
create_tensor: loading tensor blk.25.post_ffw_norm.weight
create_tensor: loading tensor blk.26.attn_norm.weight
create_tensor: loading tensor blk.26.attn_q.weight
create_tensor: loading tensor blk.26.attn_k.weight
create_tensor: loading tensor blk.26.attn_v.weight
create_tensor: loading tensor blk.26.attn_output.weight
create_tensor: loading tensor blk.26.attn_q_norm.weight
create_tensor: loading tensor blk.26.attn_k_norm.weight
create_tensor: loading tensor blk.26.post_attention_norm.weight
create_tensor: loading tensor blk.26.layer_output_scale.weight
create_tensor: loading tensor blk.26.ffn_norm.weight
create_tensor: loading tensor blk.26.ffn_gate.weight
create_tensor: loading tensor blk.26.ffn_up.weight
create_tensor: loading tensor blk.26.ffn_down.weight
create_tensor: loading tensor blk.26.post_ffw_norm.weight
create_tensor: loading tensor blk.27.attn_norm.weight
create_tensor: loading tensor blk.27.attn_q.weight
create_tensor: loading tensor blk.27.attn_k.weight
create_tensor: loading tensor blk.27.attn_v.weight
create_tensor: loading tensor blk.27.attn_output.weight
create_tensor: loading tensor blk.27.attn_q_norm.weight
create_tensor: loading tensor blk.27.attn_k_norm.weight
create_tensor: loading tensor blk.27.post_attention_norm.weight
create_tensor: loading tensor blk.27.layer_output_scale.weight
create_tensor: loading tensor blk.27.ffn_norm.weight
create_tensor: loading tensor blk.27.ffn_gate.weight
create_tensor: loading tensor blk.27.ffn_up.weight
create_tensor: loading tensor blk.27.ffn_down.weight
create_tensor: loading tensor blk.27.post_ffw_norm.weight
create_tensor: loading tensor blk.28.attn_norm.weight
create_tensor: loading tensor blk.28.attn_q.weight
create_tensor: loading tensor blk.28.attn_k.weight
create_tensor: loading tensor blk.28.attn_v.weight
create_tensor: loading tensor blk.28.attn_output.weight
create_tensor: loading tensor blk.28.attn_q_norm.weight
create_tensor: loading tensor blk.28.attn_k_norm.weight
create_tensor: loading tensor blk.28.post_attention_norm.weight
create_tensor: loading tensor blk.28.layer_output_scale.weight
create_tensor: loading tensor blk.28.ffn_norm.weight
create_tensor: loading tensor blk.28.ffn_gate.weight
create_tensor: loading tensor blk.28.ffn_up.weight
create_tensor: loading tensor blk.28.ffn_down.weight
create_tensor: loading tensor blk.28.post_ffw_norm.weight
create_tensor: loading tensor blk.29.attn_norm.weight
create_tensor: loading tensor blk.29.attn_q.weight
create_tensor: loading tensor blk.29.attn_k.weight
create_tensor: loading tensor blk.29.attn_output.weight
create_tensor: loading tensor blk.29.attn_q_norm.weight
create_tensor: loading tensor blk.29.attn_k_norm.weight
create_tensor: loading tensor blk.29.post_attention_norm.weight
create_tensor: loading tensor blk.29.layer_output_scale.weight
create_tensor: loading tensor blk.29.ffn_norm.weight
create_tensor: loading tensor blk.29.ffn_gate.weight
create_tensor: loading tensor blk.29.ffn_up.weight
create_tensor: loading tensor blk.29.ffn_down.weight
create_tensor: loading tensor blk.29.post_ffw_norm.weight
create_tensor: loading tensor blk.30.attn_norm.weight
create_tensor: loading tensor blk.30.attn_q.weight
create_tensor: loading tensor blk.30.attn_k.weight
create_tensor: loading tensor blk.30.attn_v.weight
create_tensor: loading tensor blk.30.attn_output.weight
create_tensor: loading tensor blk.30.attn_q_norm.weight
create_tensor: loading tensor blk.30.attn_k_norm.weight
create_tensor: loading tensor blk.30.post_attention_norm.weight
create_tensor: loading tensor blk.30.layer_output_scale.weight
create_tensor: loading tensor blk.30.ffn_norm.weight
create_tensor: loading tensor blk.30.ffn_gate.weight
create_tensor: loading tensor blk.30.ffn_up.weight
create_tensor: loading tensor blk.30.ffn_down.weight
create_tensor: loading tensor blk.30.post_ffw_norm.weight
create_tensor: loading tensor blk.31.attn_norm.weight
create_tensor: loading tensor blk.31.attn_q.weight
create_tensor: loading tensor blk.31.attn_k.weight
create_tensor: loading tensor blk.31.attn_v.weight
create_tensor: loading tensor blk.31.attn_output.weight
create_tensor: loading tensor blk.31.attn_q_norm.weight
create_tensor: loading tensor blk.31.attn_k_norm.weight
create_tensor: loading tensor blk.31.post_attention_norm.weight
create_tensor: loading tensor blk.31.layer_output_scale.weight
create_tensor: loading tensor blk.31.ffn_norm.weight
create_tensor: loading tensor blk.31.ffn_gate.weight
create_tensor: loading tensor blk.31.ffn_up.weight
create_tensor: loading tensor blk.31.ffn_down.weight
create_tensor: loading tensor blk.31.post_ffw_norm.weight
create_tensor: loading tensor blk.32.attn_norm.weight
create_tensor: loading tensor blk.32.attn_q.weight
create_tensor: loading tensor blk.32.attn_k.weight
create_tensor: loading tensor blk.32.attn_v.weight
create_tensor: loading tensor blk.32.attn_output.weight
create_tensor: loading tensor blk.32.attn_q_norm.weight
create_tensor: loading tensor blk.32.attn_k_norm.weight
create_tensor: loading tensor blk.32.post_attention_norm.weight
create_tensor: loading tensor blk.32.layer_output_scale.weight
create_tensor: loading tensor blk.32.ffn_norm.weight
create_tensor: loading tensor blk.32.ffn_gate.weight
create_tensor: loading tensor blk.32.ffn_up.weight
create_tensor: loading tensor blk.32.ffn_down.weight
create_tensor: loading tensor blk.32.post_ffw_norm.weight
create_tensor: loading tensor blk.33.attn_norm.weight
create_tensor: loading tensor blk.33.attn_q.weight
create_tensor: loading tensor blk.33.attn_k.weight
create_tensor: loading tensor blk.33.attn_v.weight
create_tensor: loading tensor blk.33.attn_output.weight
create_tensor: loading tensor blk.33.attn_q_norm.weight
create_tensor: loading tensor blk.33.attn_k_norm.weight
create_tensor: loading tensor blk.33.post_attention_norm.weight
create_tensor: loading tensor blk.33.layer_output_scale.weight
create_tensor: loading tensor blk.33.ffn_norm.weight
create_tensor: loading tensor blk.33.ffn_gate.weight
create_tensor: loading tensor blk.33.ffn_up.weight
create_tensor: loading tensor blk.33.ffn_down.weight
create_tensor: loading tensor blk.33.post_ffw_norm.weight
create_tensor: loading tensor blk.34.attn_norm.weight
create_tensor: loading tensor blk.34.attn_q.weight
create_tensor: loading tensor blk.34.attn_k.weight
create_tensor: loading tensor blk.34.attn_v.weight
create_tensor: loading tensor blk.34.attn_output.weight
create_tensor: loading tensor blk.34.attn_q_norm.weight
create_tensor: loading tensor blk.34.attn_k_norm.weight
create_tensor: loading tensor blk.34.post_attention_norm.weight
create_tensor: loading tensor blk.34.layer_output_scale.weight
create_tensor: loading tensor blk.34.ffn_norm.weight
create_tensor: loading tensor blk.34.ffn_gate.weight
create_tensor: loading tensor blk.34.ffn_up.weight
create_tensor: loading tensor blk.34.ffn_down.weight
create_tensor: loading tensor blk.34.post_ffw_norm.weight
create_tensor: loading tensor blk.35.attn_norm.weight
create_tensor: loading tensor blk.35.attn_q.weight
create_tensor: loading tensor blk.35.attn_k.weight
create_tensor: loading tensor blk.35.attn_output.weight
create_tensor: loading tensor blk.35.attn_q_norm.weight
create_tensor: loading tensor blk.35.attn_k_norm.weight
create_tensor: loading tensor blk.35.post_attention_norm.weight
create_tensor: loading tensor blk.35.layer_output_scale.weight
create_tensor: loading tensor blk.35.ffn_norm.weight
create_tensor: loading tensor blk.35.ffn_gate.weight
create_tensor: loading tensor blk.35.ffn_up.weight
create_tensor: loading tensor blk.35.ffn_down.weight
create_tensor: loading tensor blk.35.post_ffw_norm.weight
create_tensor: loading tensor blk.36.attn_norm.weight
create_tensor: loading tensor blk.36.attn_q.weight
create_tensor: loading tensor blk.36.attn_k.weight
create_tensor: loading tensor blk.36.attn_v.weight
create_tensor: loading tensor blk.36.attn_output.weight
create_tensor: loading tensor blk.36.attn_q_norm.weight
create_tensor: loading tensor blk.36.attn_k_norm.weight
create_tensor: loading tensor blk.36.post_attention_norm.weight
create_tensor: loading tensor blk.36.layer_output_scale.weight
create_tensor: loading tensor blk.36.ffn_norm.weight
create_tensor: loading tensor blk.36.ffn_gate.weight
create_tensor: loading tensor blk.36.ffn_up.weight
create_tensor: loading tensor blk.36.ffn_down.weight
create_tensor: loading tensor blk.36.post_ffw_norm.weight
create_tensor: loading tensor blk.37.attn_norm.weight
create_tensor: loading tensor blk.37.attn_q.weight
create_tensor: loading tensor blk.37.attn_k.weight
create_tensor: loading tensor blk.37.attn_v.weight
create_tensor: loading tensor blk.37.attn_output.weight
create_tensor: loading tensor blk.37.attn_q_norm.weight
create_tensor: loading tensor blk.37.attn_k_norm.weight
create_tensor: loading tensor blk.37.post_attention_norm.weight
create_tensor: loading tensor blk.37.layer_output_scale.weight
create_tensor: loading tensor blk.37.ffn_norm.weight
create_tensor: loading tensor blk.37.ffn_gate.weight
create_tensor: loading tensor blk.37.ffn_up.weight
create_tensor: loading tensor blk.37.ffn_down.weight
create_tensor: loading tensor blk.37.post_ffw_norm.weight
create_tensor: loading tensor blk.38.attn_norm.weight
create_tensor: loading tensor blk.38.attn_q.weight
create_tensor: loading tensor blk.38.attn_k.weight
create_tensor: loading tensor blk.38.attn_v.weight
create_tensor: loading tensor blk.38.attn_output.weight
create_tensor: loading tensor blk.38.attn_q_norm.weight
create_tensor: loading tensor blk.38.attn_k_norm.weight
create_tensor: loading tensor blk.38.post_attention_norm.weight
create_tensor: loading tensor blk.38.layer_output_scale.weight
create_tensor: loading tensor blk.38.ffn_norm.weight
create_tensor: loading tensor blk.38.ffn_gate.weight
create_tensor: loading tensor blk.38.ffn_up.weight
create_tensor: loading tensor blk.38.ffn_down.weight
create_tensor: loading tensor blk.38.post_ffw_norm.weight
create_tensor: loading tensor blk.39.attn_norm.weight
create_tensor: loading tensor blk.39.attn_q.weight
create_tensor: loading tensor blk.39.attn_k.weight
create_tensor: loading tensor blk.39.attn_v.weight
create_tensor: loading tensor blk.39.attn_output.weight
create_tensor: loading tensor blk.39.attn_q_norm.weight
create_tensor: loading tensor blk.39.attn_k_norm.weight
create_tensor: loading tensor blk.39.post_attention_norm.weight
create_tensor: loading tensor blk.39.layer_output_scale.weight
create_tensor: loading tensor blk.39.ffn_norm.weight
create_tensor: loading tensor blk.39.ffn_gate.weight
create_tensor: loading tensor blk.39.ffn_up.weight
create_tensor: loading tensor blk.39.ffn_down.weight
create_tensor: loading tensor blk.39.post_ffw_norm.weight
create_tensor: loading tensor blk.40.attn_norm.weight
create_tensor: loading tensor blk.40.attn_q.weight
create_tensor: loading tensor blk.40.attn_k.weight
create_tensor: loading tensor blk.40.attn_v.weight
create_tensor: loading tensor blk.40.attn_output.weight
create_tensor: loading tensor blk.40.attn_q_norm.weight
create_tensor: loading tensor blk.40.attn_k_norm.weight
create_tensor: loading tensor blk.40.post_attention_norm.weight
create_tensor: loading tensor blk.40.layer_output_scale.weight
create_tensor: loading tensor blk.40.ffn_norm.weight
create_tensor: loading tensor blk.40.ffn_gate.weight
create_tensor: loading tensor blk.40.ffn_up.weight
create_tensor: loading tensor blk.40.ffn_down.weight
create_tensor: loading tensor blk.40.post_ffw_norm.weight
create_tensor: loading tensor blk.41.attn_norm.weight
create_tensor: loading tensor blk.41.attn_q.weight
create_tensor: loading tensor blk.41.attn_k.weight
create_tensor: loading tensor blk.41.attn_output.weight
create_tensor: loading tensor blk.41.attn_q_norm.weight
create_tensor: loading tensor blk.41.attn_k_norm.weight
create_tensor: loading tensor blk.41.post_attention_norm.weight
create_tensor: loading tensor blk.41.layer_output_scale.weight
create_tensor: loading tensor blk.41.ffn_norm.weight
create_tensor: loading tensor blk.41.ffn_gate.weight
create_tensor: loading tensor blk.41.ffn_up.weight
create_tensor: loading tensor blk.41.ffn_down.weight
create_tensor: loading tensor blk.41.post_ffw_norm.weight
create_tensor: loading tensor blk.42.attn_norm.weight
create_tensor: loading tensor blk.42.attn_q.weight
create_tensor: loading tensor blk.42.attn_k.weight
create_tensor: loading tensor blk.42.attn_v.weight
create_tensor: loading tensor blk.42.attn_output.weight
create_tensor: loading tensor blk.42.attn_q_norm.weight
create_tensor: loading tensor blk.42.attn_k_norm.weight
create_tensor: loading tensor blk.42.post_attention_norm.weight
create_tensor: loading tensor blk.42.layer_output_scale.weight
create_tensor: loading tensor blk.42.ffn_norm.weight
create_tensor: loading tensor blk.42.ffn_gate.weight
create_tensor: loading tensor blk.42.ffn_up.weight
create_tensor: loading tensor blk.42.ffn_down.weight
create_tensor: loading tensor blk.42.post_ffw_norm.weight
create_tensor: loading tensor blk.43.attn_norm.weight
create_tensor: loading tensor blk.43.attn_q.weight
create_tensor: loading tensor blk.43.attn_k.weight
create_tensor: loading tensor blk.43.attn_v.weight
create_tensor: loading tensor blk.43.attn_output.weight
create_tensor: loading tensor blk.43.attn_q_norm.weight
create_tensor: loading tensor blk.43.attn_k_norm.weight
create_tensor: loading tensor blk.43.post_attention_norm.weight
create_tensor: loading tensor blk.43.layer_output_scale.weight
create_tensor: loading tensor blk.43.ffn_norm.weight
create_tensor: loading tensor blk.43.ffn_gate.weight
create_tensor: loading tensor blk.43.ffn_up.weight
create_tensor: loading tensor blk.43.ffn_down.weight
create_tensor: loading tensor blk.43.post_ffw_norm.weight
create_tensor: loading tensor blk.44.attn_norm.weight
create_tensor: loading tensor blk.44.attn_q.weight
create_tensor: loading tensor blk.44.attn_k.weight
create_tensor: loading tensor blk.44.attn_v.weight
create_tensor: loading tensor blk.44.attn_output.weight
create_tensor: loading tensor blk.44.attn_q_norm.weight
create_tensor: loading tensor blk.44.attn_k_norm.weight
create_tensor: loading tensor blk.44.post_attention_norm.weight
create_tensor: loading tensor blk.44.layer_output_scale.weight
create_tensor: loading tensor blk.44.ffn_norm.weight
create_tensor: loading tensor blk.44.ffn_gate.weight
create_tensor: loading tensor blk.44.ffn_up.weight
create_tensor: loading tensor blk.44.ffn_down.weight
create_tensor: loading tensor blk.44.post_ffw_norm.weight
create_tensor: loading tensor blk.45.attn_norm.weight
create_tensor: loading tensor blk.45.attn_q.weight
create_tensor: loading tensor blk.45.attn_k.weight
create_tensor: loading tensor blk.45.attn_v.weight
create_tensor: loading tensor blk.45.attn_output.weight
create_tensor: loading tensor blk.45.attn_q_norm.weight
create_tensor: loading tensor blk.45.attn_k_norm.weight
create_tensor: loading tensor blk.45.post_attention_norm.weight
create_tensor: loading tensor blk.45.layer_output_scale.weight
create_tensor: loading tensor blk.45.ffn_norm.weight
create_tensor: loading tensor blk.45.ffn_gate.weight
create_tensor: loading tensor blk.45.ffn_up.weight
create_tensor: loading tensor blk.45.ffn_down.weight
create_tensor: loading tensor blk.45.post_ffw_norm.weight
create_tensor: loading tensor blk.46.attn_norm.weight
create_tensor: loading tensor blk.46.attn_q.weight
create_tensor: loading tensor blk.46.attn_k.weight
create_tensor: loading tensor blk.46.attn_v.weight
create_tensor: loading tensor blk.46.attn_output.weight
create_tensor: loading tensor blk.46.attn_q_norm.weight
create_tensor: loading tensor blk.46.attn_k_norm.weight
create_tensor: loading tensor blk.46.post_attention_norm.weight
create_tensor: loading tensor blk.46.layer_output_scale.weight
create_tensor: loading tensor blk.46.ffn_norm.weight
create_tensor: loading tensor blk.46.ffn_gate.weight
create_tensor: loading tensor blk.46.ffn_up.weight
create_tensor: loading tensor blk.46.ffn_down.weight
create_tensor: loading tensor blk.46.post_ffw_norm.weight
create_tensor: loading tensor blk.47.attn_norm.weight
create_tensor: loading tensor blk.47.attn_q.weight
create_tensor: loading tensor blk.47.attn_k.weight
create_tensor: loading tensor blk.47.attn_output.weight
create_tensor: loading tensor blk.47.attn_q_norm.weight
create_tensor: loading tensor blk.47.attn_k_norm.weight
create_tensor: loading tensor blk.47.post_attention_norm.weight
create_tensor: loading tensor blk.47.layer_output_scale.weight
create_tensor: loading tensor blk.47.ffn_norm.weight
create_tensor: loading tensor blk.47.ffn_gate.weight
create_tensor: loading tensor blk.47.ffn_up.weight
create_tensor: loading tensor blk.47.ffn_down.weight
create_tensor: loading tensor blk.47.post_ffw_norm.weight
done_getting_tensors: tensor 'token_embd.weight' (q6_K) (and 674 others) cannot be used with preferred buffer type CPU_REPACK, using CPU instead
load_tensors:          CPU model buffer size =  7425.13 MiB
load_all_data: no device found for buffer type CPU for async uploads
.................................................................................
load_tensors: releasing mmaps (not needed after tensor copy)
llama_model_loader: direct I/O is enabled, disabling mmap
llama_model_loader: loaded meta data with 52 key-value pairs and 49 tensors from c:\llama.cpp\models\Gemma-4\12B\gemma-4-12B-it-qat-assistant-MTP-Q8_0.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = gemma4-assistant
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                     general.sampling.top_k i32              = 64
llama_model_loader: - kv   3:                     general.sampling.top_p f32              = 0.950000
llama_model_loader: - kv   4:                      general.sampling.temp f32              = 1.000000
llama_model_loader: - kv   5:                               general.name str              = Gemma 4 12B It Qat Q4_0 Unquantized A...
llama_model_loader: - kv   6:                           general.finetune str              = 12b-it-qat-unquantized-assistant
llama_model_loader: - kv   7:                           general.basename str              = gemma-4
llama_model_loader: - kv   8:                         general.size_label str              = 423M
llama_model_loader: - kv   9:                            general.license str              = apache-2.0
llama_model_loader: - kv  10:                       general.license.link str              = https://ai.google.dev/gemma/docs/gemm...
llama_model_loader: - kv  11:                   general.base_model.count u32              = 1
llama_model_loader: - kv  12:                  general.base_model.0.name str              = Gemma 4 12B It Assistant
llama_model_loader: - kv  13:          general.base_model.0.organization str              = Google
llama_model_loader: - kv  14:              general.base_model.0.repo_url str              = https://huggingface.co/google/gemma-4...
llama_model_loader: - kv  15:                               general.tags arr[str,1]       = ["any-to-any"]
llama_model_loader: - kv  16:               gemma4-assistant.block_count u32              = 4
llama_model_loader: - kv  17:            gemma4-assistant.context_length u32              = 262144
llama_model_loader: - kv  18:          gemma4-assistant.embedding_length u32              = 1024
llama_model_loader: - kv  19:       gemma4-assistant.feed_forward_length u32              = 8192
llama_model_loader: - kv  20:      gemma4-assistant.attention.head_count u32              = 16
llama_model_loader: - kv  21:   gemma4-assistant.attention.head_count_kv arr[i32,4]       = [8, 8, 8, 1]
llama_model_loader: - kv  22:            gemma4-assistant.rope.freq_base f32              = 1000000.000000
llama_model_loader: - kv  23:        gemma4-assistant.rope.freq_base_swa f32              = 10000.000000
llama_model_loader: - kv  24: gemma4-assistant.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  25:      gemma4-assistant.attention.key_length u32              = 512
llama_model_loader: - kv  26:    gemma4-assistant.attention.value_length u32              = 512
llama_model_loader: - kv  27:                          general.file_type u32              = 7
llama_model_loader: - kv  28:  gemma4-assistant.attention.sliding_window u32              = 1024
llama_model_loader: - kv  29: gemma4-assistant.attention.shared_kv_layers u32              = 4
llama_model_loader: - kv  30: gemma4-assistant.embedding_length_per_layer_input u32              = 0
llama_model_loader: - kv  31: gemma4-assistant.attention.sliding_window_pattern arr[bool,4]      = [true, true, true, false]
llama_model_loader: - kv  32:  gemma4-assistant.attention.key_length_swa u32              = 256
llama_model_loader: - kv  33: gemma4-assistant.attention.value_length_swa u32              = 256
llama_model_loader: - kv  34:      gemma4-assistant.rope.dimension_count u32              = 512
llama_model_loader: - kv  35:  gemma4-assistant.rope.dimension_count_swa u32              = 256
llama_model_loader: - kv  36:      gemma4-assistant.embedding_length_out u32              = 3840
llama_model_loader: - kv  37:      gemma4-assistant.nextn_predict_layers u32              = 4
llama_model_loader: - kv  38:               general.quantization_version u32              = 2
llama_model_loader: - kv  39:                       tokenizer.ggml.model str              = gemma4
llama_model_loader: - kv  40:                      tokenizer.ggml.tokens arr[str,262144]  = ["<pad>", "<eos>", "<bos>", "<unk>", ...
llama_model_loader: - kv  41:                      tokenizer.ggml.scores arr[f32,262144]  = [-1000.000000, -1000.000000, -1000.00...
llama_model_loader: - kv  42:                  tokenizer.ggml.token_type arr[i32,262144]  = [3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  43:                      tokenizer.ggml.merges arr[str,514906]  = ["\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n \n", ...
llama_model_loader: - kv  44:                tokenizer.ggml.bos_token_id u32              = 2
llama_model_loader: - kv  45:                tokenizer.ggml.eos_token_id u32              = 1
llama_model_loader: - kv  46:            tokenizer.ggml.unknown_token_id u32              = 3
llama_model_loader: - kv  47:            tokenizer.ggml.padding_token_id u32              = 0
llama_model_loader: - kv  48:               tokenizer.ggml.mask_token_id u32              = 4
llama_model_loader: - kv  49:                    tokenizer.chat_template str              = {%- macro format_parameters(propertie...
llama_model_loader: - kv  50:            tokenizer.ggml.add_space_prefix bool             = false
llama_model_loader: - kv  51:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - type  f32:   26 tensors
llama_model_loader: - type q8_0:   23 tensors
print_info: file format = GGUF V3 (latest)
print_info: file type   = Q8_0
print_info: file size   = 428.53 MiB (8.50 BPW)
llama_model_load: error loading model: unknown model architecture: 'gemma4-assistant'
llama_model_load_from_file_impl: failed to load model
llama_model_load_mtp_from_file: failed to load assistant from 'c:\llama.cpp\models\Gemma-4\12B\gemma-4-12B-it-qat-assistant-MTP-Q8_0.gguf'
Warning: failed to load MTP head 'c:\llama.cpp\models\Gemma-4\12B\gemma-4-12B-it-qat-assistant-MTP-Q8_0.gguf' (rc=-3); continuing without MTP
llama_context: constructing llama_context
llama_context: n_seq_max     = 1
llama_context: n_ctx         = 2048
llama_context: n_ctx_seq     = 2048
llama_context: n_batch       = 512
llama_context: n_ubatch      = 512
llama_context: causal_attn   = 1
llama_context: flash_attn    = auto
llama_context: kv_unified    = false
llama_context: freq_base     = 1000000.0
llama_context: freq_scale    = 1
llama_context: n_rs_seq      = 0
llama_context: n_ctx_seq (2048) < n_ctx_train (262144) -- the full capacity of the model will not be utilized
set_abort_callback: call
llama_context:        CPU  output buffer size =     1.00 MiB
llama_kv_cache_iswa: using full-size SWA cache (ref: https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055)
llama_kv_cache_iswa: creating non-SWA KV cache, size = 2048 cells
llama_kv_cache: layer   0: filtered
llama_kv_cache: layer   1: filtered
llama_kv_cache: layer   2: filtered
llama_kv_cache: layer   3: filtered
llama_kv_cache: layer   4: filtered
llama_kv_cache: layer   5: dev = CPU
llama_kv_cache: layer   6: filtered
llama_kv_cache: layer   7: filtered
llama_kv_cache: layer   8: filtered
llama_kv_cache: layer   9: filtered
llama_kv_cache: layer  10: filtered
llama_kv_cache: layer  11: dev = CPU
llama_kv_cache: layer  12: filtered
llama_kv_cache: layer  13: filtered
llama_kv_cache: layer  14: filtered
llama_kv_cache: layer  15: filtered
llama_kv_cache: layer  16: filtered
llama_kv_cache: layer  17: dev = CPU
llama_kv_cache: layer  18: filtered
llama_kv_cache: layer  19: filtered
llama_kv_cache: layer  20: filtered
llama_kv_cache: layer  21: filtered
llama_kv_cache: layer  22: filtered
llama_kv_cache: layer  23: dev = CPU
llama_kv_cache: layer  24: filtered
llama_kv_cache: layer  25: filtered
llama_kv_cache: layer  26: filtered
llama_kv_cache: layer  27: filtered
llama_kv_cache: layer  28: filtered
llama_kv_cache: layer  29: dev = CPU
llama_kv_cache: layer  30: filtered
llama_kv_cache: layer  31: filtered
llama_kv_cache: layer  32: filtered
llama_kv_cache: layer  33: filtered
llama_kv_cache: layer  34: filtered
llama_kv_cache: layer  35: dev = CPU
llama_kv_cache: layer  36: filtered
llama_kv_cache: layer  37: filtered
llama_kv_cache: layer  38: filtered
llama_kv_cache: layer  39: filtered
llama_kv_cache: layer  40: filtered
llama_kv_cache: layer  41: dev = CPU
llama_kv_cache: layer  42: filtered
llama_kv_cache: layer  43: filtered
llama_kv_cache: layer  44: filtered
llama_kv_cache: layer  45: filtered
llama_kv_cache: layer  46: filtered
llama_kv_cache: layer  47: dev = CPU
llama_kv_cache: reusing layers:
llama_kv_cache: - layer   0: no reuse
llama_kv_cache: - layer   1: no reuse
llama_kv_cache: - layer   2: no reuse
llama_kv_cache: - layer   3: no reuse
llama_kv_cache: - layer   4: no reuse
llama_kv_cache: - layer   5: no reuse
llama_kv_cache: - layer   6: no reuse
llama_kv_cache: - layer   7: no reuse
llama_kv_cache: - layer   8: no reuse
llama_kv_cache: - layer   9: no reuse
llama_kv_cache: - layer  10: no reuse
llama_kv_cache: - layer  11: no reuse
llama_kv_cache: - layer  12: no reuse
llama_kv_cache: - layer  13: no reuse
llama_kv_cache: - layer  14: no reuse
llama_kv_cache: - layer  15: no reuse
llama_kv_cache: - layer  16: no reuse
llama_kv_cache: - layer  17: no reuse
llama_kv_cache: - layer  18: no reuse
llama_kv_cache: - layer  19: no reuse
llama_kv_cache: - layer  20: no reuse
llama_kv_cache: - layer  21: no reuse
llama_kv_cache: - layer  22: no reuse
llama_kv_cache: - layer  23: no reuse
llama_kv_cache: - layer  24: no reuse
llama_kv_cache: - layer  25: no reuse
llama_kv_cache: - layer  26: no reuse
llama_kv_cache: - layer  27: no reuse
llama_kv_cache: - layer  28: no reuse
llama_kv_cache: - layer  29: no reuse
llama_kv_cache: - layer  30: no reuse
llama_kv_cache: - layer  31: no reuse
llama_kv_cache: - layer  32: no reuse
llama_kv_cache: - layer  33: no reuse
llama_kv_cache: - layer  34: no reuse
llama_kv_cache: - layer  35: no reuse
llama_kv_cache: - layer  36: no reuse
llama_kv_cache: - layer  37: no reuse
llama_kv_cache: - layer  38: no reuse
llama_kv_cache: - layer  39: no reuse
llama_kv_cache: - layer  40: no reuse
llama_kv_cache: - layer  41: no reuse
llama_kv_cache: - layer  42: no reuse
llama_kv_cache: - layer  43: no reuse
llama_kv_cache: - layer  44: no reuse
llama_kv_cache: - layer  45: no reuse
llama_kv_cache: - layer  46: no reuse
llama_kv_cache: - layer  47: no reuse
llama_kv_cache:        CPU KV buffer size =    32.00 MiB
llama_kv_cache: size =   32.00 MiB (  2048 cells,   8 layers,  1/1 seqs), K (f16):   16.00 MiB, V (f16):   16.00 MiB
llama_kv_cache: attn_rot_k = 0, n_embd_head_k_all = 512
llama_kv_cache: attn_rot_v = 0, n_embd_head_k_all = 512
llama_kv_cache_iswa: creating     SWA KV cache, size = 2048 cells
llama_kv_cache: layer   0: dev = CPU
llama_kv_cache: layer   1: dev = CPU
llama_kv_cache: layer   2: dev = CPU
llama_kv_cache: layer   3: dev = CPU
llama_kv_cache: layer   4: dev = CPU
llama_kv_cache: layer   5: filtered
llama_kv_cache: layer   6: dev = CPU
llama_kv_cache: layer   7: dev = CPU
llama_kv_cache: layer   8: dev = CPU
llama_kv_cache: layer   9: dev = CPU
llama_kv_cache: layer  10: dev = CPU
llama_kv_cache: layer  11: filtered
llama_kv_cache: layer  12: dev = CPU
llama_kv_cache: layer  13: dev = CPU
llama_kv_cache: layer  14: dev = CPU
llama_kv_cache: layer  15: dev = CPU
llama_kv_cache: layer  16: dev = CPU
llama_kv_cache: layer  17: filtered
llama_kv_cache: layer  18: dev = CPU
llama_kv_cache: layer  19: dev = CPU
llama_kv_cache: layer  20: dev = CPU
llama_kv_cache: layer  21: dev = CPU
llama_kv_cache: layer  22: dev = CPU
llama_kv_cache: layer  23: filtered
llama_kv_cache: layer  24: dev = CPU
llama_kv_cache: layer  25: dev = CPU
llama_kv_cache: layer  26: dev = CPU
llama_kv_cache: layer  27: dev = CPU
llama_kv_cache: layer  28: dev = CPU
llama_kv_cache: layer  29: filtered
llama_kv_cache: layer  30: dev = CPU
llama_kv_cache: layer  31: dev = CPU
llama_kv_cache: layer  32: dev = CPU
llama_kv_cache: layer  33: dev = CPU
llama_kv_cache: layer  34: dev = CPU
llama_kv_cache: layer  35: filtered
llama_kv_cache: layer  36: dev = CPU
llama_kv_cache: layer  37: dev = CPU
llama_kv_cache: layer  38: dev = CPU
llama_kv_cache: layer  39: dev = CPU
llama_kv_cache: layer  40: dev = CPU
llama_kv_cache: layer  41: filtered
llama_kv_cache: layer  42: dev = CPU
llama_kv_cache: layer  43: dev = CPU
llama_kv_cache: layer  44: dev = CPU
llama_kv_cache: layer  45: dev = CPU
llama_kv_cache: layer  46: dev = CPU
llama_kv_cache: layer  47: filtered
llama_kv_cache: reusing layers:
llama_kv_cache: - layer   0: no reuse
llama_kv_cache: - layer   1: no reuse
llama_kv_cache: - layer   2: no reuse
llama_kv_cache: - layer   3: no reuse
llama_kv_cache: - layer   4: no reuse
llama_kv_cache: - layer   5: no reuse
llama_kv_cache: - layer   6: no reuse
llama_kv_cache: - layer   7: no reuse
llama_kv_cache: - layer   8: no reuse
llama_kv_cache: - layer   9: no reuse
llama_kv_cache: - layer  10: no reuse
llama_kv_cache: - layer  11: no reuse
llama_kv_cache: - layer  12: no reuse
llama_kv_cache: - layer  13: no reuse
llama_kv_cache: - layer  14: no reuse
llama_kv_cache: - layer  15: no reuse
llama_kv_cache: - layer  16: no reuse
llama_kv_cache: - layer  17: no reuse
llama_kv_cache: - layer  18: no reuse
llama_kv_cache: - layer  19: no reuse
llama_kv_cache: - layer  20: no reuse
llama_kv_cache: - layer  21: no reuse
llama_kv_cache: - layer  22: no reuse
llama_kv_cache: - layer  23: no reuse
llama_kv_cache: - layer  24: no reuse
llama_kv_cache: - layer  25: no reuse
llama_kv_cache: - layer  26: no reuse
llama_kv_cache: - layer  27: no reuse
llama_kv_cache: - layer  28: no reuse
llama_kv_cache: - layer  29: no reuse
llama_kv_cache: - layer  30: no reuse
llama_kv_cache: - layer  31: no reuse
llama_kv_cache: - layer  32: no reuse
llama_kv_cache: - layer  33: no reuse
llama_kv_cache: - layer  34: no reuse
llama_kv_cache: - layer  35: no reuse
llama_kv_cache: - layer  36: no reuse
llama_kv_cache: - layer  37: no reuse
llama_kv_cache: - layer  38: no reuse
llama_kv_cache: - layer  39: no reuse
llama_kv_cache: - layer  40: no reuse
llama_kv_cache: - layer  41: no reuse
llama_kv_cache: - layer  42: no reuse
llama_kv_cache: - layer  43: no reuse
llama_kv_cache: - layer  44: no reuse
llama_kv_cache: - layer  45: no reuse
llama_kv_cache: - layer  46: no reuse
llama_kv_cache: - layer  47: no reuse
llama_kv_cache:        CPU KV buffer size =   640.00 MiB
llama_kv_cache: size =  640.00 MiB (  2048 cells,  40 layers,  1/1 seqs), K (f16):  320.00 MiB, V (f16):  320.00 MiB
llama_kv_cache: attn_rot_k = 0, n_embd_head_k_all = 256
llama_kv_cache: attn_rot_v = 0, n_embd_head_k_all = 256
llama_context: enumerating backends
llama_context: backend_ptrs.size() = 1
sched_reserve: reserving ...
sched_reserve: max_nodes = 5344
sched_reserve: reserving full memory module
sched_reserve: worst-case: n_tokens = 512, n_seqs = 1, n_outputs = 1
graph_reserve: reserving a graph for ubatch with n_tokens =    1, n_seqs =  1, n_outputs =    1
sched_reserve: Flash Attention was auto, set to enabled
sched_reserve: resolving fused Gated Delta Net support:
graph_reserve: reserving a graph for ubatch with n_tokens =    1, n_seqs =  1, n_outputs =    1
sched_reserve: fused Gated Delta Net (autoregressive) enabled
graph_reserve: reserving a graph for ubatch with n_tokens =   16, n_seqs =  1, n_outputs =   16
sched_reserve: fused Gated Delta Net (chunked) enabled
graph_reserve: reserving a graph for ubatch with n_tokens =  512, n_seqs =  1, n_outputs =  512
graph_reserve: reserving a graph for ubatch with n_tokens =    1, n_seqs =  1, n_outputs =    1
graph_reserve: reserving a graph for ubatch with n_tokens =  512, n_seqs =  1, n_outputs =  512
sched_reserve:        CPU compute buffer size =   527.00 MiB
sched_reserve: graph nodes  = 1972
sched_reserve: graph splits = 1
sched_reserve: reserve took 4.44 ms, sched copies = 1

[main]: n_ctx = 2048, n_threads = 8, gpu_layers = 999
[main]: flash-attn = auto (runtime-selected)
[main]: system_info: CPU : SSE3 = 1 | SSSE3 = 1 | AVX = 1 | AVX2 = 1 | F16C = 1 | FMA = 1 | AVX512 = 1 | LLAMAFILE = 1 | OPENMP = 1 | REPACK = 1 |

> Running with custom prompt => [1/18]: [This is Izzy's bedroom]
  prefill: 16677.2ms (555 tokens, 33.3 t/s)
 "<|channel>thought
<channel|>```json
{
    "answer": "c. That is correct Mara. It is your sister's room",
    justification": "Mara has correctly identified the location as Izzy's bedroom, so I must affirm her answer."
}
  [output diag: bytes=221 nuls=0 other_ctrl=0 empty_pieces=1/58]
  [empty piece token ids (first 1): 258883]
  [all token ids (59): 623 100 45518 107 101 2717 3723 107 236782 107 140 236775 14433 1083 623 236755 236761 2981 563 4338 63338 236761 1030 563 822 12198 236789 236751 2978 827 107 140 258883 4017 2540 1083 623 201883 815 15195 8385 506 4563 618 29689 2946 236789 236751 15238 236764 834 564 1921 29691 1116 3890 1781 107 236783]
  [output hex (221 bytes):]
    0000: 20 22 3c 7c 63 68 61 6e 6e 65 6c 3e 74 68 6f 75  | "<|channel>thou|
    0010: 67 68 74 0a 3c 63 68 61 6e 6e 65 6c 7c 3e 60 60  |ght.<channel|>``|
    0020: 60 6a 73 6f 6e 0a 7b 0a 20 20 20 20 22 61 6e 73  |`json.{.    "ans|
    0030: 77 65 72 22 3a 20 22 63 2e 20 54 68 61 74 20 69  |wer": "c. That i|
    0040: 73 20 63 6f 72 72 65 63 74 20 4d 61 72 61 2e 20  |s correct Mara. |
    0050: 49 74 20 69 73 20 79 6f 75 72 20 73 69 73 74 65  |It is your siste|
    0060: 72 27 73 20 72 6f 6f 6d 22 2c 0a 20 20 20 20 6a  |r's room",.    j|
    0070: 75 73 74 69 66 69 63 61 74 69 6f 6e 22 3a 20 22  |ustification": "|
    0080: 4d 61 72 61 20 68 61 73 20 63 6f 72 72 65 63 74  |Mara has correct|
    0090: 6c 79 20 69 64 65 6e 74 69 66 69 65 64 20 74 68  |ly identified th|
    00a0: 65 20 6c 6f 63 61 74 69 6f 6e 20 61 73 20 49 7a  |e location as Iz|
    00b0: 7a 79 27 73 20 62 65 64 72 6f 6f 6d 2c 20 73 6f  |zy's bedroom, so|
    00c0: 20 49 20 6d 75 73 74 20 61 66 66 69 72 6d 20 68  | I must affirm h|
    00d0: 65 72 20 61 6e 73 77 65 72 2e 22 0a 7d           |er answer.".}|
  decode: 9638.6ms (58 tokens, 6.02 t/s)
  decode (no-TTFT): 9637.6ms (59 tokens, 6.12 t/s) [TTFT 1.0ms]

