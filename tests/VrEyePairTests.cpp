// Pure tests: never start the game, emulator, window, or Vulkan.
#include "graphics/host_gpu/renderer/cache/vrEyePair.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <limits>

namespace {
using namespace Libs::Graphics::VrEyePair;

void Verify(bool result, const char* condition, int line) {
	if (!result) {
		std::fprintf(stderr, "VrEyePairTests: line %d failed: %s\n", line, condition);
		std::abort();
	}
}
#define VERIFY(cond) Verify(static_cast<bool>(cond), #cond, __LINE__)

constexpr uint64_t kEyeBytes = 0x1000000u;
constexpr uint64_t kStart = 0x20289f0000ull;

void AdjacentEyes() {
	// Observed Horizon allocation addresses; input is deliberately reversed.
	std::array<Candidate, 2> images {{
	    {kStart + kEyeBytes, kEyeBytes, 50, true, true},
	    {kStart,             kEyeBytes, 60, true, true},
	}};
	const auto pair = Select(images);
	VERIFY(pair.has_value());
	VERIFY(pair->left_index == 1u && pair->right_index == 0u);
}

void PreferRecentPair() {
	std::array<Candidate, 5> images {{
	    {kStart,                    kEyeBytes, 4, true, true},
	    {kStart + kEyeBytes,        kEyeBytes, 6, true, true},
	    {kStart + 5u * kEyeBytes,   kEyeBytes, 20, false, false},
	    {kStart + 6u * kEyeBytes,   kEyeBytes, 21, false, false},
	    {kStart + 12u * kEyeBytes,  kEyeBytes, 100, true, true},
	}};
	const auto pair = Select(images);
	VERIFY(pair.has_value());
	VERIFY(pair->left_index == 2u && pair->right_index == 3u);
	// Once equally recent, prefer a GPU-rendered pair.
	images[0].last_access_tick = 20;
	images[1].last_access_tick = 20;
	const auto rendered = Select(images);
	VERIFY(rendered.has_value());
	VERIFY(rendered->left_index == 0u && rendered->right_index == 1u);
}

void RejectUnpairedSurfaces() {
	const std::array<Candidate, 3> images {{
	    {kStart, kEyeBytes, 100, true, true},
	    {kStart + 2u * kEyeBytes, kEyeBytes, 100, true, true},
	    {kStart + 4u * kEyeBytes, 2u * kEyeBytes, 100, true, true},
	}};
	VERIFY(!Select(images).has_value());
	VERIFY(!Select(std::span<const Candidate> {}).has_value());
}

void RejectCorruptAllocations() {
	const std::array<Candidate, 6> images {{
	    {0u,                 kEyeBytes,  1, true, true},
	    {kStart + 1u,        kEyeBytes,  1, true, true},
	    {kStart,             0u,         1, true, true},
	    {kStart + kEyeBytes, 0x101u,     1, true, true},
	    {0xffffffffffff0000ull, 0x20000u,  1, true, true},
	    {0xffffffffffff0000ull - 0x20000u, 0x20000u, 1, true, true},
	}};
	VERIFY(!Select(images).has_value());
}

void DuplicateViews() {
	const std::array<Candidate, 4> images {{
	    {kStart,             kEyeBytes, 21, true, true},
	    {kStart,             kEyeBytes,  2, false, false},
	    {kStart + kEyeBytes, kEyeBytes, 25, true, true},
	    {kStart + kEyeBytes, kEyeBytes,  5, false, false},
	}};
	const auto pair = Select(images);
	VERIFY(pair.has_value());
	VERIFY(pair->left_index == 0u && pair->right_index == 2u);
}

void DeterministicTie() {
	const std::array<Candidate, 4> images {{
	    {kStart + 11u * kEyeBytes, kEyeBytes, 100, true, true},
	    {kStart + 10u * kEyeBytes, kEyeBytes, 100, true, true},
	    {kStart + 1u * kEyeBytes,  kEyeBytes, 100, true, true},
	    {kStart,                  kEyeBytes, 100, true, true},
	}};
	const auto pair = Select(images);
	VERIFY(pair.has_value());
	VERIFY(pair->left_index == 3u && pair->right_index == 2u);
}
} // namespace

int main() {
	AdjacentEyes();
	PreferRecentPair();
	RejectUnpairedSurfaces();
	RejectCorruptAllocations();
	DuplicateViews();
	DeterministicTie();
	std::puts("VrEyePairTests: passed");
}
