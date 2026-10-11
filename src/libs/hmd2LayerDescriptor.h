#pragma once

// Read-only decoder for the PS VR2 category-16 layer descriptor. A layer
// contains two native 32-byte TSharp surface descriptors after a 0x10 type.
// The decoder uses values, not pointers, so it can be tested without PS5
// libraries, a running title, or any Vulkan device.

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>

namespace Libs::Hmd2LayerDescriptor {

inline constexpr size_t kLayerBytes = 0x44u;
inline constexpr uint32_t kStereoLayerType = 0x10u;

struct Texture {
	uint64_t address = 0;
	uint32_t width = 0;
	uint32_t height = 0;
	uint32_t format = 0;
	uint32_t tile_mode = 0;
	uint32_t image_type = 0;
	bool     compressed = false;
};

struct Layer {
	Texture left;
	Texture right;
};

[[nodiscard]] inline bool DecodeTexture(std::span<const uint8_t> data, Texture* texture) {
	if (texture == nullptr || data.size() < 32u) {
		return false;
	}
	std::array<uint32_t, 8> words {};
	std::memcpy(words.data(), data.data(), 32u);
	const uint64_t base = (static_cast<uint64_t>(words[1] & 0xffu) << 32u) |
	                      static_cast<uint64_t>(words[0]);
	Texture parsed {};
	parsed.address = base << 8u;
	parsed.width = 1u + ((words[1] >> 30u) & 3u) + ((words[2] & 0xfffu) << 2u);
	parsed.height = 1u + ((words[2] >> 14u) & 0x3fffu);
	parsed.format = (words[1] >> 20u) & 0x1ffu;
	parsed.tile_mode = (words[3] >> 20u) & 0x1fu;
	parsed.image_type = (words[3] >> 28u) & 0xfu;
	parsed.compressed = ((words[6] >> 21u) & 1u) != 0u;
	if (parsed.address == 0 || parsed.width < 64u || parsed.width > 16384u ||
	    parsed.height < 64u || parsed.height > 16384u ||
	    parsed.format == 0u || parsed.image_type != 9u) {
		return false;
	}
	*texture = parsed;
	return true;
}

[[nodiscard]] inline bool Decode(std::span<const uint8_t> data, Layer* layer) {
	if (layer == nullptr || data.size() < kLayerBytes) {
		return false;
	}
	uint32_t type = 0;
	std::memcpy(&type, data.data(), sizeof(type));
	if (type != kStereoLayerType) {
		return false;
	}
	Layer parsed {};
	if (!DecodeTexture(data.subspan(4u, 32u), &parsed.left) ||
	    !DecodeTexture(data.subspan(0x24u, 32u), &parsed.right)) {
		return false;
	}
	*layer = parsed;
	return true;
}

} // namespace Libs::Hmd2LayerDescriptor
