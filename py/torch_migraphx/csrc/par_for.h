/*
 * Copyright (c) 2022-present, Advanced Micro Devices, Inc. All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#include <algorithm>
#include <cassert>
#include <cmath>
#include <thread>
#include <vector>

namespace torch_migraphx {

struct joinable_thread : std::thread {
  template <class... Xs>
  joinable_thread(Xs &&... xs)
      : std::thread(std::forward<Xs>(xs)...) // NOLINT
  {}

  joinable_thread &operator=(joinable_thread &&other) = default;
  joinable_thread(joinable_thread &&other) = default;

  ~joinable_thread() {
    if (this->joinable())
      this->join();
  }
};

template <class F>
auto thread_invoke(std::size_t i, std::size_t tid, F f) -> decltype(f(i, tid)) {
  f(i, tid);
}

template <class F>
auto thread_invoke(std::size_t i, std::size_t, F f) -> decltype(f(i)) {
  f(i);
}

template <class F>
void par_for_impl(std::size_t n, std::size_t threadsize, F f) {
  if (threadsize <= 1) {
    for (std::size_t i = 0; i < n; i++)
      thread_invoke(i, 0, f);
  } else {
    std::vector<joinable_thread> threads(threadsize);
// Using const here causes gcc 5 to ICE
#if (!defined(__GNUC__) || __GNUC__ != 5)
    const
#endif
        std::size_t grainsize =
            std::ceil(static_cast<double>(n) / threads.size());

    std::size_t work = 0;
    std::size_t tid = 0;
    std::generate(threads.begin(), threads.end(), [=, &work, &tid] {
      auto result = joinable_thread([=] {
        std::size_t start = work;
        std::size_t last = std::min(n, work + grainsize);
        for (std::size_t i = start; i < last; i++) {
          thread_invoke(i, tid, f);
        }
      });
      work += grainsize;
      ++tid;
      return result;
    });
    assert(work >= n);
  }
}

template <class F> void par_for(std::size_t n, std::size_t min_grain, F f) {
  const auto threadsize =
      std::min<std::size_t>(std::thread::hardware_concurrency(),
                            n / std::max<std::size_t>(1, min_grain));
  par_for_impl(n, threadsize, f);
}

template <class F> void par_for(std::size_t n, F f) {
  const int min_grain = 8;
  par_for(n, min_grain, f);
}

} // namespace torch_migraphx
