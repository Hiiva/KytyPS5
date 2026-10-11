// Pure offline tests for PS VR2 layer/TSharp decoding. No game is launched.
#include "libs/hmd2LayerDescriptor.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {
using namespace Libs::Hmd2LayerDescriptor;

void Verify(bool success, const char* reason, int line) {
	if (!success) {
		std::fprintf(stderr, "Hmd2LayerDescriptorTests: line %d: %s\n", line, reason);
		std::abort();
	}
}
#define VERIFY(condition) Verify(static_cast<bool>(condition), #condition, __LINE__)

[[nodiscard]] std::array<uint8_t, kLayerBytes> Fixture() {
	// The first descriptor is based on the preexisting Horizon reprojection
	// log. It has a 3328x1664, 32-bit, tiled PS5 surface at 0x2072570000.
	constexpr std::array<uint32_t, 17> words {
	    0x10u,
	    0x20725700u, 0xc1600000u, 0x019fc33fu, 0x91b00204u,
	    0u, 0x00700000u, 0u, 0u,
	    0x20735700u, 0xc1600000u, 0x019fc33fu, 0x91b00204u,
	    0u, 0x00700000u, 0u, 0u,
	};
	std::array<uint8_t, kLayerBytes> bytes {};
	std::memcpy(bytes.data(), words.data(), bytes.size());
	return bytes;
}

void NormalLayer() {
	const auto bytes = Fixture();
	Layer result {};
	VERIFY(Decode(bytes, &result));
	VERIFY(result.left.address == 0x2072570000ull);
	VERIFY(result.right.address == 0x2073570000ull);
	VERIFY(result.left.width == 3328u);
	VERIFY(result.left.height == 1664u);
	VERIFY(result.left.format == 22u);
	VERIFY(result.left.tile_mode == 27u);
	VERIFY(result.left.image_type == 9u);
	VERIFY(!result.left.compressed);
	VERIFY(result.right.width == result.left.width);
}

void HighAddressBits() {
	auto bytes = Fixture();
	std::array<uint32_t, 17> words {};
	std::memcpy(words.data(), bytes.data(), bytes.size());
	words[2] |= 0x42u;  // Low 8 bits of TSharp word 1 extend the GPU address.
	std::memcpy(bytes.data(), words.data(), bytes.size());
	Layer result {};
	VERIFY(Decode(bytes, &result));
	VERIFY(result.left.address == 0x422072570000ull);
	VERIFY(result.left.width == 3328u);
}

void NativeReprojectionSource() {
    // 0xe0-byte nested Horizon reprojection block from an earlier, read-only
    // guest-memory dump. The TSharp descriptor starts at byte 8, rather
    // than byte 4 as in a category-16 video-out layer.
    constexpr std::array<uint32_t, 10> words {{
        0x00000001u, 0x00000000u,
        0x20723700u, 0xc1600000u, 0x019fc33fu, 0x91b00204u,
        0x00000000u, 0x00700000u, 0x00000000u, 0x00000000u
    }};
    std::array<uint8_t, 40> raw {};
    std::memcpy(raw.data(), words.data(), raw.size());
    Texture texture {};
    VERIFY(DecodeTexture(std::span(raw).subspan(8u, 32u), &texture));
    VERIFY(texture.address == 0x2072370000ull);
    VERIFY(texture.width == 3328u && texture.height == 1664u);
    VERIFY(texture.format == 22u && texture.tile_mode == 27u);
    VERIFY(texture.image_type == 9u && !texture.compressed);
    texture.address = 0xfeedbeefu;
    VERIFY(!DecodeTexture(std::span(raw).subspan(8u, 31u), &texture));
    VERIFY(texture.address == 0xfeedbeefu);
}
void InvalidLayer() {
	auto bytes = Fixture();
	Layer result {};
	result.left.address = 0xdeadbeefu;
	VERIFY(!Decode(std::span(bytes.data(), bytes.size() - 1u), &result));
	VERIFY(result.left.address == 0xdeadbeefu);
	VERIFY(!Decode(bytes, nullptr));

	const uint32_t invalid_type = 8u;
	std::memcpy(bytes.data(), &invalid_type, 4u);
	VERIFY(!Decode(bytes, &result));
	VERIFY(result.left.address == 0xdeadbeefu);
	bytes = Fixture();
	// Width below the 64-pixel lower bound.
	const uint32_t narrow = 0x019fc000u;
	std::memcpy(bytes.data() + 0x0cu, &narrow, 4u);
	VERIFY(!Decode(bytes, &result));
	VERIFY(result.left.address == 0xdeadbeefu);
}
} // namespace

int main() {
	NormalLayer();
	HighAddressBits();
	NativeReprojectionSource();
	InvalidLayer();
	std::puts("Hmd2LayerDescriptorTests: passed");
}
