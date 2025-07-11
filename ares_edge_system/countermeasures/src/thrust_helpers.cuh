#include <thrust/count.h>

namespace thrust {
    // Add missing thrust::count function
    template <typename Iterator, typename T>
    size_t count(Iterator first, Iterator last, const T& value) {
        size_t result = 0;
        while (first != last) {
            if (*first == value) {
                result++;
            }
            ++first;
        }
        return result;
    }
}
