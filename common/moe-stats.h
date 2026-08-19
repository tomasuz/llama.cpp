#pragma once

#include <cstdint>
#include <memory>
#include <vector>

struct ggml_tensor;

// Collects MoE expert-routing statistics from the compute graph.
//
// The router's top-k selection is materialised as an I32 tensor of shape
// [n_expert_used, n_tokens] that llm_graph_context tags as "ffn_moe_topk"; the
// default graph callback then renames it to "ffn_moe_topk-<il>". Every element
// is the index of an expert that was activated for one token, so counting them
// yields a per-layer activation histogram without touching the routing code.
//
// Intended to be installed as a ggml_backend_sched_eval_callback:
//
//     common_moe_stats stats;
//     params.cb_eval           = common_moe_stats_cb_eval;
//     params.cb_eval_user_data = &stats;
//
// The callback only requests data for tensors it actually needs, so graphs
// without MoE layers cost one string comparison per node.
struct common_moe_stats {
    common_moe_stats();
    ~common_moe_stats();

    common_moe_stats(const common_moe_stats &) = delete;
    common_moe_stats & operator=(const common_moe_stats &) = delete;

    struct layer_counts {
        int                   il = -1;
        std::vector<uint64_t> experts;      // experts[expert_id] = activation count
        uint64_t              total = 0;    // sum of experts[], i.e. tokens * n_expert_used
    };

    // Snapshot of everything collected so far, ordered by layer index.
    // Safe to call from another thread while inference is running.
    std::vector<layer_counts> snapshot() const;

    // Fraction of activations that went to the busiest `top_n` experts of a
    // layer. 1.0 means fully concentrated, top_n/n_expert means perfectly even.
    // Returns -1.0 if the layer has not been seen yet.
    double concentration(int il, size_t top_n) const;

    void reset();

    // Number of tensors observed. Useful to tell "no MoE in this model" from
    // "callback was never installed".
    uint64_t observed() const;

    struct impl;
    std::unique_ptr<impl> pimpl;
};

// ggml_backend_sched_eval_callback; user_data must be a common_moe_stats *.
bool common_moe_stats_cb_eval(struct ggml_tensor * t, bool ask, void * user_data);
