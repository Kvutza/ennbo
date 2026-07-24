#include <c_api/Index_c.h>
#include <c_api/index_factory_c.h>

#include <array>
#include <cstdlib>

namespace {

void require(bool condition) {
  if (!condition) {
    std::abort();
  }
}

void check_index(const char* description) {
  FaissIndex* index = nullptr;
  require(faiss_index_factory(&index, 2, description, METRIC_L2) == 0);
  require(index != nullptr);

  constexpr std::array<float, 8> rows = {
      0.0F, 0.0F,
      1.0F, 0.0F,
      0.0F, 1.0F,
      1.0F, 1.0F,
  };
  require(faiss_Index_add(index, 4, rows.data()) == 0);

  constexpr std::array<float, 2> query = {0.0F, 0.0F};
  std::array<float, 2> distances = {};
  std::array<idx_t, 2> labels = {};
  require(faiss_Index_search(
              index,
              1,
              query.data(),
              2,
              distances.data(),
              labels.data()) == 0);
  require(labels[0] == 0);
  require(distances[0] == 0.0F);

  faiss_Index_free(index);
}

}  // namespace

int main() {
  check_index("Flat");
  check_index("HNSW32");
  return 0;
}
