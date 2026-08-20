#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct ggml_tensor;

// Collects two kinds of MoE statistics from the compute graph.
//
// 1. Expert routing histogram, per layer.
//    The router's top-k selection is materialised as an I32 tensor of shape
//    [n_expert_used, n_tokens] tagged "ffn_moe_topk"; each element is an
//    activated expert index, so counting them gives the histogram.
//
// 2. Per-weight-tensor placement and cost.
//    The expert matmuls (ffn_moe_up / gate / down) take the big weight tensor
//    as src[0]. That tensor's name is exactly what --override-tensor matches
//    on, and its buffer tells which device holds it. Recording name, device,
//    size and accumulated time therefore produces the table needed to decide
//    an -ot layout, which the routing histogram alone cannot: all experts of
//    a layer live in ONE tensor ([n_embd, n_ff, n_expert]), so -ot can never
//    address an individual expert.
//
// Timing caveat: installing any eval callback makes the scheduler evaluate the
// graph in chunks instead of one split at a time, and the measured interval
// covers the chunk ending at the observed node rather than that node alone.
// The overhead is therefore present in every sample; treat the numbers as
// comparable to each other, not as absolute kernel times.
// Rinkimo rezimas. Skirtumas esminis, nes ggml_backend_sched po KIEKVIENO
// gabalo, kuris baigiasi stebimu mazgu, daro ggml_backend_synchronize().
//
//   PLACEMENT  ask fazeje nuskaitomi src[0] metaduomenys (vardas, irenginys,
//              dydis) ir VISADA grazinama false. Gabalo riba nesukuriama,
//              sinchronizacijos nera, GPU duomenys neperkeliami. Kaina ~0.
//              To uztenka -ot sprendimams.
//
//   FULL       papildomai renkama ekspertu histograma ir laikas. Tam butina
//              grazinti true, o tai sukelia sinchronizacija kiekvienam
//              stebimam mazgui. Ismatuota kaina: 18,6 % pralaidumo.
//              Laikui matuoti pigiau GGML_VK_PERF_LOGGER=1 (7,8 %), nes jis
//              naudoja Vulkan query pool ir sinchronizuoja karta per grafa.
enum common_moe_stats_mode {
    COMMON_MOE_STATS_OFF = 0,
    COMMON_MOE_STATS_PLACEMENT,
    COMMON_MOE_STATS_FULL,
};

struct common_moe_stats {
    common_moe_stats();
    ~common_moe_stats();

    common_moe_stats(const common_moe_stats &) = delete;
    common_moe_stats & operator=(const common_moe_stats &) = delete;

    struct layer_counts {
        int                   il = -1;
        std::vector<uint64_t> experts;    // experts[expert_id] = activation count
        uint64_t              total = 0;
    };

    struct tensor_stats {
        std::string name;                 // weight tensor, e.g. blk.19.ffn_up_exps.weight
        std::string device;               // buffer holding it, e.g. Vulkan1
        size_t      bytes    = 0;
        uint64_t    calls    = 0;
        double      time_ms  = 0.0;       // accumulated, includes observation overhead
        int         il       = -1;
    };

    std::vector<layer_counts> snapshot() const;
    std::vector<tensor_stats> tensors()  const;

    // Share of activations captured by the busiest `top_n` experts of a layer.
    // -1.0 if the layer has not been seen.
    double concentration(int il, size_t top_n) const;

    void reset();
    uint64_t observed() const;

    void set_mode(common_moe_stats_mode m);
    common_moe_stats_mode mode() const;

    struct impl;
    std::unique_ptr<impl> pimpl;
};

// ggml_backend_sched_eval_callback; user_data must be a common_moe_stats *.
bool common_moe_stats_cb_eval(struct ggml_tensor * t, bool ask, void * user_data);
