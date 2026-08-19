#include "moe-stats.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>

static const char * MOE_TOPK_PREFIX = "ffn_moe_topk";

struct common_moe_stats::impl {
    mutable std::mutex           mtx;
    std::map<int, layer_counts>  layers;     // keyed by layer index, kept ordered
    uint64_t                     observed = 0;
    std::vector<int32_t>         buf;        // reused staging buffer for tensor reads
};

common_moe_stats::common_moe_stats() : pimpl(new impl()) {}
common_moe_stats::~common_moe_stats() = default;

std::vector<common_moe_stats::layer_counts> common_moe_stats::snapshot() const {
    std::lock_guard<std::mutex> lock(pimpl->mtx);

    std::vector<layer_counts> out;
    out.reserve(pimpl->layers.size());
    for (const auto & kv : pimpl->layers) {
        out.push_back(kv.second);
    }
    return out;
}

double common_moe_stats::concentration(int il, size_t top_n) const {
    std::lock_guard<std::mutex> lock(pimpl->mtx);

    const auto it = pimpl->layers.find(il);
    if (it == pimpl->layers.end() || it->second.total == 0 || top_n == 0) {
        return -1.0;
    }

    std::vector<uint64_t> sorted = it->second.experts;
    top_n = std::min(top_n, sorted.size());

    std::partial_sort(sorted.begin(), sorted.begin() + top_n, sorted.end(), std::greater<uint64_t>());

    uint64_t head = 0;
    for (size_t i = 0; i < top_n; ++i) {
        head += sorted[i];
    }

    return (double) head / (double) it->second.total;
}

void common_moe_stats::reset() {
    std::lock_guard<std::mutex> lock(pimpl->mtx);
    pimpl->layers.clear();
    pimpl->observed = 0;
}

uint64_t common_moe_stats::observed() const {
    std::lock_guard<std::mutex> lock(pimpl->mtx);
    return pimpl->observed;
}

// Tensors are named "<tag>-<il>" by the default graph callback
// (ggml_format_name(cur, "%s-%d", name, il)). Returns -1 if the name does not
// carry a layer suffix, which also guards against a future naming change
// silently collapsing every layer into one bucket.
static int moe_layer_from_name(const char * name) {
    const char * dash = strrchr(name, '-');
    if (!dash || dash[1] == '\0') {
        return -1;
    }

    char * end = nullptr;
    const long il = strtol(dash + 1, &end, 10);
    if (end == dash + 1 || *end != '\0' || il < 0) {
        return -1;
    }

    return (int) il;
}

static bool moe_is_topk_tensor(const struct ggml_tensor * t) {
    return strncmp(t->name, MOE_TOPK_PREFIX, strlen(MOE_TOPK_PREFIX)) == 0;
}

bool common_moe_stats_cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * stats = (common_moe_stats *) user_data;
    if (!stats || !t) {
        return false;
    }

    // Phase 1: the scheduler asks whether this tensor's data is wanted.
    // Saying "no" here means no device->host copy is made, so non-MoE graphs
    // pay only for the name comparison.
    if (ask) {
        return moe_is_topk_tensor(t);
    }

    // Phase 2: data is available.
    if (!moe_is_topk_tensor(t) || t->type != GGML_TYPE_I32) {
        return true;
    }

    const int il = moe_layer_from_name(t->name);
    if (il < 0) {
        return true;
    }

    const int64_t n_elem = ggml_nelements(t);
    if (n_elem <= 0) {
        return true;
    }

    auto * pimpl = stats->pimpl.get();
    std::lock_guard<std::mutex> lock(pimpl->mtx);

    pimpl->buf.resize((size_t) n_elem);
    ggml_backend_tensor_get(t, pimpl->buf.data(), 0, ggml_nbytes(t));

    auto & lc = pimpl->layers[il];
    lc.il = il;

    for (int64_t i = 0; i < n_elem; ++i) {
        const int32_t e = pimpl->buf[i];
        if (e < 0) {
            continue;   // padding / unused slot
        }
        if ((size_t) e >= lc.experts.size()) {
            lc.experts.resize((size_t) e + 1, 0);
        }
        lc.experts[(size_t) e]++;
        lc.total++;
    }

    pimpl->observed++;

    return true;
}
