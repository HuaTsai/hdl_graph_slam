#include <chrono>

#define TIME_IT_BEGIN(name) \
  const auto _##name##Begin = std::chrono::high_resolution_clock::now();

#define TIME_IT_END(name) \
  const auto _##name##End = std::chrono::high_resolution_clock::now(); \
  const auto _##name##Duration = _##name##End - _##name##Begin; \
  fprintf(stderr, #name " duration: %ld us\n", std::chrono::duration_cast<std::chrono::microseconds>(_##name##Duration).count());

