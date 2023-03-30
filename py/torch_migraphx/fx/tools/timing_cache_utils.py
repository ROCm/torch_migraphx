#####################################################################################
# Copyright (c) 2022-present, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2020-present, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#####################################################################################
import logging
import os

logger = logging.getLogger(__name__)


class TimingCacheManager:
    def __init__(self, timing_cache_prefix: str = "", save_timing_cache=False):
        # Setting timing cache for TRTInterpreter
        tc = os.environ.get("TRT_TIMING_CACHE_PREFIX", "")
        timing_cache_prefix_name = timing_cache_prefix
        if not timing_cache_prefix and tc:
            timing_cache_prefix_name = tc

        self.timing_cache_prefix_name = timing_cache_prefix_name
        self.save_timing_cache = save_timing_cache

    def get_file_full_name(self, name: str):
        return f"{self.timing_cache_prefix_name}_{name}.npy"

    def get_timing_cache_trt(self, timing_cache_file: str) -> bytearray:
        timing_cache_file = self.get_file_full_name(timing_cache_file)
        try:
            with open(timing_cache_file, "rb") as raw_cache:
                cache_data = raw_cache.read()
            return bytearray(cache_data)
        except Exception:
            return None

    def update_timing_cache(self, timing_cache_file: str,
                            serilized_cache: bytearray) -> None:
        if not self.save_timing_cache:
            return
        timing_cache_file = self.get_file_full_name(timing_cache_file)
        with open(timing_cache_file, "wb") as local_cache:
            local_cache.seek(0)
            local_cache.write(serilized_cache)
            local_cache.truncate()