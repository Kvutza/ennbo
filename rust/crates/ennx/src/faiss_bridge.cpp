#include <faiss/IndexFlat.h>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <string>

static_assert(sizeof(faiss::idx_t) == sizeof(std::int64_t));

struct EnnFaissIndex {
    std::unique_ptr<faiss::IndexFlatL2> index;
};

thread_local std::string enn_faiss_error;

static void set_error(const std::exception& error) {
    enn_faiss_error = error.what();
}

static void set_error(const char* error) {
    enn_faiss_error = error;
}

extern "C" {

void* enn_faiss_new(std::size_t num_dim) noexcept {
    try {
        auto* result = new EnnFaissIndex;
        result->index = std::make_unique<faiss::IndexFlatL2>(num_dim);
        return result;
    } catch (const std::exception& error) {
        set_error(error);
    } catch (...) {
        set_error("unknown Faiss construction error");
    }
    return nullptr;
}

void enn_faiss_free(void* handle) noexcept {
    delete static_cast<EnnFaissIndex*>(handle);
}

int enn_faiss_reset(void* handle) noexcept {
    try {
        if (handle == nullptr) {
            set_error("null Faiss index");
            return 1;
        }
        static_cast<EnnFaissIndex*>(handle)->index->reset();
        return 0;
    } catch (const std::exception& error) {
        set_error(error);
    } catch (...) {
        set_error("unknown Faiss reset error");
    }
    return 1;
}

int enn_faiss_add(void* handle, std::size_t num_rows, const float* rows) noexcept {
    try {
        if (handle == nullptr) {
            set_error("null Faiss index");
            return 1;
        }
        static_cast<EnnFaissIndex*>(handle)->index->add(
            static_cast<faiss::idx_t>(num_rows), rows);
        return 0;
    } catch (const std::exception& error) {
        set_error(error);
    } catch (...) {
        set_error("unknown Faiss add error");
    }
    return 1;
}

std::size_t enn_faiss_len(const void* handle) noexcept {
    if (handle == nullptr) {
        return 0;
    }
    return static_cast<const EnnFaissIndex*>(handle)->index->ntotal;
}

int enn_faiss_search(
    void* handle,
    std::size_t num_queries,
    std::size_t k,
    const float* queries,
    float* distances,
    std::int64_t* labels) noexcept {
    try {
        if (handle == nullptr) {
            set_error("null Faiss index");
            return 1;
        }
        static_cast<EnnFaissIndex*>(handle)->index->search(
            static_cast<faiss::idx_t>(num_queries),
            queries,
            static_cast<faiss::idx_t>(k),
            distances,
            reinterpret_cast<faiss::idx_t*>(labels));
        return 0;
    } catch (const std::exception& error) {
        set_error(error);
    } catch (...) {
        set_error("unknown Faiss search error");
    }
    return 1;
}

const char* enn_faiss_last_error() noexcept {
    return enn_faiss_error.c_str();
}

}
