#ifndef FUSION_CPU_EP_SYNC_H
#define FUSION_CPU_EP_SYNC_H

typedef struct FusionThreadPool FusionThreadPool;
typedef void (*fusion_body_fn)(void* ctx, int task, int nways);

FusionThreadPool* fusion_pool_create(int num_ways);
int fusion_pool_size(const FusionThreadPool* pool);
void fusion_pool_parallel_for(FusionThreadPool* pool, int nways,
                              fusion_body_fn fn, void* ctx);
void fusion_pool_destroy(FusionThreadPool* pool);

int fusion_ccx_pin_enabled(void);
void fusion_pin_worker_to_core(int worker_idx, int nthreads);

#endif
