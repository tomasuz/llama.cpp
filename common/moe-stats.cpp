#include "moe-stats.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <mutex>

static const char * MOE_TOPK_PREFIX = "ffn_moe_topk";
static const char * EXPS_MARKER     = "_exps.";

struct common_moe_stats::impl {
    mutable std::mutex          mtx;
    std::map<int, layer_counts> layers;
    std::map<std::string, tensor_stats> tensors;   // keyed by weight tensor name
    uint64_t                    observed = 0;
    common_moe_stats_mode       mode = COMMON_MOE_STATS_PLACEMENT;
    std::vector<int32_t>        buf;

    // Set when an observed node is announced (ask=true) and consumed when the
    // same node comes back computed (ask=false).
    const ggml_tensor * pending = nullptr;
    std::chrono::steady_clock::time_point pending_t0;
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

std::vector<common_moe_stats::tensor_stats> common_moe_stats::tensors() const {
    std::lock_guard<std::mutex> lock(pimpl->mtx);
    std::vector<tensor_stats> out;
    out.reserve(pimpl->tensors.size());
    for (const auto & kv : pimpl->tensors) {
        out.push_back(kv.second);
    }
    std::sort(out.begin(), out.end(), [](const tensor_stats & a, const tensor_stats & b) {
        return a.time_ms > b.time_ms;   // most expensive first
    });
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
    pimpl->tensors.clear();
    pimpl->observed = 0;
    pimpl->pending  = nullptr;
}

void common_moe_stats::set_mode(common_moe_stats_mode m) {
    std::lock_guard<std::mutex> lock(pimpl->mtx);
    pimpl->mode = m;
}

common_moe_stats_mode common_moe_stats::mode() const {
    std::lock_guard<std::mutex> lock(pimpl->mtx);
    return pimpl->mode;
}

uint64_t common_moe_stats::observed() const {
    std::lock_guard<std::mutex> lock(pimpl->mtx);
    return pimpl->observed;
}

// Tensors are named "<tag>-<il>" by the default graph callback. Returns -1 when
// no layer suffix is present, so a naming change skips the sample instead of
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

// Layer index of a weight tensor, e.g. "blk.19.ffn_up_exps.weight" -> 19.
static int moe_layer_from_weight_name(const char * name) {
    if (strncmp(name, "blk.", 4) != 0) {
        return -1;
    }
    char * end = nullptr;
    const long il = strtol(name + 4, &end, 10);
    if (end == name + 4 || *end != '.' || il < 0) {
        return -1;
    }
    return (int) il;
}

static bool moe_is_topk_tensor(const ggml_tensor * t) {
    return strncmp(t->name, MOE_TOPK_PREFIX, strlen(MOE_TOPK_PREFIX)) == 0;
}

// True for the expert matmuls: their src[0] is the big *_exps.weight tensor,
// which is what --override-tensor patterns address.
static const ggml_tensor * moe_expert_weight(const ggml_tensor * t) {
    const ggml_tensor * w = t->src[0];
    if (w && strstr(w->name, EXPS_MARKER) != nullptr) {
        return w;
    }
    return nullptr;
}

static bool moe_is_interesting(const ggml_tensor * t) {
    return moe_is_topk_tensor(t) || moe_expert_weight(t) != nullptr;
}

bool common_moe_stats_cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * stats = (common_moe_stats *) user_data;
    if (!stats || !t) {
        return false;
    }
    auto * pimpl = stats->pimpl.get();

    // Phase 1: the scheduler asks whether this node's data is wanted. Saying no
    // means no device->host copy and no chunk boundary here, so uninteresting
    // nodes cost one name comparison.
    if (ask) {
        const ggml_tensor * w = moe_expert_weight(t);

        // src[0] metaduomenys (vardas, buferis, dydis) priskirti dar modelio
        // ikelimo metu, tad juos galima nuskaityti CIA, negrazinant true.
        // Grazinus false gabalo riba nesukuriama -> jokios sinchronizacijos.
        if (w) {
            std::lock_guard<std::mutex> lock(pimpl->mtx);
            auto & ts = pimpl->tensors[w->name];
            if (ts.calls == 0 && ts.name.empty()) {
                ts.name   = w->name;
                ts.bytes  = ggml_nbytes(w);
                ts.il     = moe_layer_from_weight_name(w->name);
                ts.device = w->buffer ? ggml_backend_buffer_name(w->buffer) : "unknown";
                pimpl->observed++;
            }
            if (pimpl->mode != COMMON_MOE_STATS_FULL) {
                return false;
            }
            pimpl->pending    = t;
            pimpl->pending_t0 = std::chrono::steady_clock::now();
            return true;
        }

        // topk reikalauja REALIU reiksmiu is GPU, tad be sinchronizacijos
        // ju gauti negalima. Renkam tik FULL rezime.
        if (pimpl->mode == COMMON_MOE_STATS_FULL && moe_is_topk_tensor(t)) {
            std::lock_guard<std::mutex> lock(pimpl->mtx);
            pimpl->pending    = t;
            pimpl->pending_t0 = std::chrono::steady_clock::now();
            return true;
        }
        return false;
    }

    // Phase 2: the node has been computed.
    const auto t1 = std::chrono::steady_clock::now();

    std::lock_guard<std::mutex> lock(pimpl->mtx);

    double elapsed_ms = 0.0;
    if (pimpl->pending == t) {
        elapsed_ms = std::chrono::duration<double, std::milli>(t1 - pimpl->pending_t0).count();
    }
    pimpl->pending = nullptr;

    // --- per-weight-tensor placement and cost ---
    if (const ggml_tensor * w = moe_expert_weight(t)) {
        auto & ts = pimpl->tensors[w->name];   // statiniai laukai jau uzpildyti ask fazeje
        ts.calls++;
        ts.time_ms += elapsed_ms;
        return true;
    }

    // --- expert routing histogram ---
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

    // A non-contiguous tensor cannot be walked as a flat array, and its
    // ggml_nbytes() covers the strided extent rather than n_elem*4 bytes.
    // Sizing the buffer from n_elem and then reading ggml_nbytes() into it
    // overflows the heap, so refuse such tensors outright.
    if (!ggml_is_contiguous(t)) {
        return true;
    }
    const size_t nbytes = ggml_nbytes(t);
    if (nbytes != (size_t) n_elem * sizeof(int32_t)) {
        return true;
    }

    pimpl->buf.resize((size_t) n_elem);
    ggml_backend_tensor_get(t, pimpl->buf.data(), 0, nbytes);

    auto & lc = pimpl->layers[il];
    lc.il = il;
    for (int64_t i = 0; i < n_elem; ++i) {
        const int32_t e = pimpl->buf[i];
        if (e < 0) {
            continue;
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
