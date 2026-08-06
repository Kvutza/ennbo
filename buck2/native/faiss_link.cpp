#include <faiss/IndexFlat.h>

#include <cmath>
#include <cstddef>

int main() {
    faiss::IndexFlatL2 index(2);
    const float rows[] = {0.0F, 0.0F, 2.0F, 0.0F};
    index.add(2, rows);

    const float query[] = {1.75F, 0.0F};
    faiss::idx_t neighbor = -1;
    float distance = 0.0F;
    index.search(1, query, 1, &distance, &neighbor);

    return neighbor == 1 && std::abs(distance - 0.0625F) < 1.0e-6F ? 0 : 1;
}
