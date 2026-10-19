#ifndef ACTION_QUEUE_INTEGRAL_H
#define ACTION_QUEUE_INTEGRAL_H
#include <cstdint>
namespace ns3 {
// Piecewise-constant byte occupancy, integrated at every queue mutation.
class ActionQueueIntegral {
public:
  void Reset(uint64_t now, uint64_t bytes) {
    active = true; start = last = now; current = bytes; area = 0;
  }
  void Set(uint64_t now, uint64_t bytes) {
    if (!active) return;
    area += double(now - last) * current;
    last = now; current = bytes;
  }
  double Mean(uint64_t now) const {
    return active && now > start
      ? (area + double(now - last) * current) / double(now - start) : 0;
  }
private:
  bool active = false;
  uint64_t start = 0, last = 0, current = 0;
  double area = 0;
};
}
#endif
