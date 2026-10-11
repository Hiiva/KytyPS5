#pragma once

// Pure selector for the two adjacent per-eye render targets observed in the
// Horizon PS VR2 render cache. It does not access a Vulkan device or guest RAM.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <span>
#include <vector>

namespace Libs::Graphics::VrEyePair {

struct Candidate {
	uint64_t address = 0;
	uint64_t allocation_size = 0;
	uint64_t last_access_tick = 0;
	bool     gpu_modified = false;
	bool     render_target = false;
};

struct Pair {
	size_t left_index = 0;
	size_t right_index = 0;
};

// A true eye pair has equal, 64-KiB-aligned allocations immediately adjacent
// in GPU memory. Prefer the pair most recently used by both eyes, then the
// strongest evidence of GPU rendering. Ties resolve by GPU address, not
// cache insertion order. No approximate matches are permitted.
[[nodiscard]] inline std::optional<Pair> Select(std::span<const Candidate> images) {
	constexpr uint64_t alignment = 0x10000u;
	const auto valid = [](const Candidate& candidate) {
		return candidate.address != 0 &&
		       (candidate.address % alignment) == 0 &&
		       candidate.allocation_size >= alignment &&
		       (candidate.allocation_size % alignment) == 0 &&
		       candidate.address <= std::numeric_limits<uint64_t>::max() -
		                                candidate.allocation_size;
	};
	std::vector<size_t> order(images.size());
	std::iota(order.begin(), order.end(), size_t {0});
	std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
		if (images[a].address != images[b].address) {
			return images[a].address < images[b].address;
		}
		return a < b;
	});

	std::optional<Pair> selected;
	uint64_t best_min_tick = 0;
	uint32_t best_quality = 0;
	uint64_t best_address = 0;
	for (const size_t left_index : order) {
		const Candidate& left = images[left_index];
		if (!valid(left)) {
			continue;
		}
		const uint64_t next_address = left.address + left.allocation_size;
		const auto match = std::lower_bound(
		    order.begin(), order.end(), next_address,
		    [&](size_t index, uint64_t address) {
			    return images[index].address < address;
		    });
		for (auto it = match; it != order.end() &&
		                      images[*it].address == next_address; ++it) {
			const size_t right_index = *it;
			const Candidate& right = images[right_index];
			if (!valid(right) || right.allocation_size != left.allocation_size) {
				continue;
			}
			const uint64_t min_tick = std::min(left.last_access_tick,
			                                   right.last_access_tick);
			const uint32_t quality =
			    static_cast<uint32_t>(left.gpu_modified) +
			    static_cast<uint32_t>(right.gpu_modified) +
			    (left.render_target && right.render_target ? 2u : 0u);
			if (!selected || min_tick > best_min_tick ||
			    (min_tick == best_min_tick && quality > best_quality) ||
			    (min_tick == best_min_tick && quality == best_quality &&
			     left.address < best_address)) {
				selected = Pair {left_index, right_index};
				best_min_tick = min_tick;
				best_quality = quality;
				best_address = left.address;
			}
		}
	}
	return selected;
}

} // namespace Libs::Graphics::VrEyePair
