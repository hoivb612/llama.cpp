D:\llama.cpp\b612.dc.080625\build>bin\RelWithDebInfo\llama-retrieval.exe -t 8 --chunk-size 1 -c 63 --context-file ..\examples\retrieval\1liners.txt -m d:\llama.cpp\models\BGE\bge-large-en-v1.5-q8_0.gguf -paffin -repack-xbox
ggml_cpu_init: from ggml-cpu-b612.c - calling ggml!ggml_init() to initialize fp16 tables
ggml_cpu_init: from ggml-cpu-b612.c - initializing GELU, SILU and EXP fp32 tables
ggml_cpu_set_tensor_repack_mode: set tensor repacking to 2
processing files: ..\examples\retrieval\1liners.txt
build: 6206 (9b2de29b) with MSVC 19.44.35211.0 for x64
Number of chunks: 640
common_init_from_params: KV cache shifting is not supported for this context, disabling KV cache shifting
common_init_from_params: added [SEP] logit bias = -inf
common_init_from_params: setting dry_penalty_last_n to ctx_size = 63
common_init_from_params: warming up the model with an empty run - please wait ... (--no-warmup to disable)
main: Tokenizing data...
main: Creating Embeddings...
main: llama_n_seq_max for llama_context is 63
main: Querying loop starts...
Total items processed: 640
Query time             =   9.51s (14.86ms per item)
Tokenization time      =   2.80ms( 0.00ms per chunk)
Create Embeddings time =   6.00s ( 9.38ms per chunk)
Errors                 =   0

llama_perf_context_print:        load time =     328.54 ms
llama_perf_context_print: prompt eval time =   15148.77 ms / 23589 tokens (    0.64 ms per token,  1557.16 tokens per second)
llama_perf_context_print:        eval time =       0.00 ms /     1 runs   (    0.00 ms per token,     0.00 tokens per second)
llama_perf_context_print:       total time =   15513.63 ms / 23590 tokens
llama_perf_context_print:    graphs reused =          0


total elapsed time   15.84sec



          Total     Total  Tensor
   Count Time(sec)   %     Time(us) Tensor Op
  210438     0.39   3.25       1.83 GGML_OP_ADD
   42434     0.03   0.29       0.82 GGML_OP_MUL
   42434     0.16   1.33       3.73 GGML_OP_NORM
  166272     9.87  83.20      59.39 GGML_OP_MUL_MAT
   41568     0.24   2.01       5.75 GGML_OP_CONT
    4330     0.01   0.05       1.28 GGML_OP_GET_ROWS
   20784     0.16   1.34       7.65 GGML_OP_SOFT_MAX
   20784     1.01   8.52      48.65 GGML_OP_UNARY

  549044    11.87 100.00

Graph Size  #_Nodes  #_Tensors
  851         866      736966

Total         866      736966
Total OPs Tensors      549044
Total NOP Tensors      187922 (skipped)

vector dot matrix multiply type frequency
   Count     %    Time(ms)       %   init_mat(ms) vec_dot_type
   41568   25.00     718.36     7.28      0.00    GGML_TYPE_f32
  124704   75.00    9155.78    92.72    312.79    GGML_TYPE_q8_0_q8_0_x8

  166272  100.00

Vector Dot Matrix Multiply Src0 Type Frequency

          Total    Total  Tensor
   Count Time(sec)   %   Time(ms) Src0 Type

   41568     0.71   7.24     0.02 ggml_type_f32
  124704     9.14  92.76     0.07 ggml_type_q8_0_x8

  166272     9.86 100.00

total number of mul_mat init conversions 124704
total elapsed init conversion time  0.31sec
average init conversion time  2.51us

total number of mul_mat repack conversions 144
total number of FAILED mul_mat repack conversions 0
total elapsed repack conversion time  0.27sec
average repack conversion time 1876.02us

vector row size count histogram for quant type: f32

  Size   Count    %     Time(ms)    Max(ms) From_Float(ms)
     8      24   0.06       0.05      0.00      0.00
    32     336   0.81       1.60      0.01      0.00
    36     408   0.98       2.09      0.01      0.00
    40     528   1.27       2.95      0.01      0.00
    44     576   1.39       3.74      0.05      0.00
    48     816   1.96       5.95      0.01      0.00
    52    1032   2.48       8.23      0.08      0.00
    56     696   1.67       6.31      0.02      0.00
    60    1176   2.83      11.67      0.05      0.00
    64    1128   2.71      11.12      0.05      0.00
    68    1344   3.23      13.85      0.08      0.00
    72     840   2.02       8.94      0.01      0.00
    76     816   1.96       9.29      0.08      0.00
    80     792   1.91      10.00      0.16      0.00
    84     696   1.67       9.60      0.09      0.00
    88     528   1.27       7.88      0.08      0.00
    92     696   1.67      10.87      0.07      0.00
    96     576   1.39       9.61      0.02      0.00
   100     504   1.21       8.87      0.03      0.00
   104     336   0.81       6.26      0.03      0.00
   108     120   0.29       2.38      0.07      0.00
   112     264   0.64       5.41      0.07      0.00
   116     360   0.87       7.91      0.03      0.00
   120     120   0.29       2.74      0.03      0.00
   124     144   0.35       3.54      0.04      0.00
   128      96   0.23       1.93      0.06      0.00
   132     144   0.35       3.19      0.03      0.00
   136      96   0.23       2.14      0.03      0.00
   140      72   0.17       1.71      0.03      0.00
   144      72   0.17       1.67      0.03      0.00
   148      96   0.23       2.39      0.03      0.00
   152     120   0.29       3.18      0.03      0.00
   156      48   0.12       1.38      0.04      0.00
   160     120   0.29       3.44      0.03      0.00
   164      96   0.23       2.94      0.04      0.00
   168      24   0.06       0.78      0.04      0.00
   172      72   0.17       2.31      0.04      0.00
   176      96   0.23       3.21      0.04      0.00
   180     120   0.29       4.51      0.10      0.00
   184      72   0.17       2.61      0.04      0.00
   188     144   0.35       5.52      0.04      0.00
   192     384   0.92      12.58      0.08      0.00
   196     360   0.87      12.23      0.09      0.00
   200     360   0.87      12.49      0.14      0.00
   204     144   0.35       5.04      0.04      0.00
   208     144   0.35       5.00      0.04      0.00
   212     312   0.75      11.65      0.05      0.00
   216     336   0.81      13.23      0.13      0.00
   220     336   0.81      13.64      0.09      0.00
   224     192   0.46       7.93      0.11      0.00
   228     264   0.64      11.28      0.05      0.00
   232     336   0.81      14.68      0.05      0.00
   236     336   0.81      15.35      0.12      0.00
   240     168   0.40       7.95      0.06      0.00
   244     216   0.52      10.71      0.08      0.00
   248     264   0.64      13.65      0.11      0.00
   252     288   0.69      15.41      0.06      0.00
   256   20784  50.00     321.80      0.26      0.00

         41568 100.00     718.36 (avg row size 182)

  Max entry: ne00 ne01 ne10 ne11  Time(ms)
               64   60   64   60      0.26

vector row size count histogram for quant type: q8_0_x8

  Size   Count    %     Time(ms)    Max(ms) From_Float(ms)
  1088  103920  83.33    6209.37      0.49    200.65
  4352   20784  16.67    2946.41      0.44    112.14

        124704 100.00    9155.78 (avg row size 1632)

  Max entry: ne00 ne01 ne10 ne11  Time(ms)
             1024 4096 1024 4096      0.49


D:\llama.cpp\b612.dc.080625\build>