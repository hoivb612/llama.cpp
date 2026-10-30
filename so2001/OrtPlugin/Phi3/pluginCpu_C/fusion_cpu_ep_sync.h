#ifndef FUSION_CPU_EP_SYNC_H
#define FUSION_CPU_EP_SYNC_H

typedef void (*fusion_body_fn)(void* ctx, int task, int nways);

void
fusion_pool_create(
    void);

int
fusion_pool_size(
    void);

void
fusion_pool_parallel_for(
    int nways,
    fusion_body_fn fn, void* ctx);

#endif
