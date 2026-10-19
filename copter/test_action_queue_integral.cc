#include "action-queue-integral.h"
#include <cassert>
#include <cmath>
int main() {
    ns3::ActionQueueIntegral q;
    q.Reset(100, 0);
    q.Set(200, 1000);
    q.Set(300, 0);
    assert(std::abs(q.Mean(500) - 250) < 1e-10);
    q.Reset(500, 1000);
    assert(q.Mean(1000) == 1000);
    q.Set(1000, 2000);
    assert(q.Mean(1500) == 1500);
    q.Reset(1500, 2000);
    assert(q.Mean(2000) == 2000);
}
