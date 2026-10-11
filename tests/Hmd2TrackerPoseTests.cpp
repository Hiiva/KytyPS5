// Offline verification of the VR tracking-result pose layout, with no guest.
#include "libs/hmd2TrackerPose.h"

#include <array>
#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>

namespace {
using namespace Libs::Hmd2TrackerPose;
void Verify(bool ok, const char* expression, int line) {
    if (!ok) {
        std::fprintf(stderr, "Hmd2TrackerPoseTests: line %d failed: %s\n",
                     line, expression);
        std::abort();
    }
}
#define VERIFY(expr) Verify(static_cast<bool>(expr), #expr, __LINE__)


float ReadFloat(const std::array<uint8_t, 0x160>& data, size_t offset) {
	float v = 0.0f;
	std::memcpy(&v, data.data() + offset, sizeof(v));
	return v;
}

uint32_t ReadU32(const std::array<uint8_t, 0x160>& data, size_t offset) {
	uint32_t v = 0;
	std::memcpy(&v, data.data() + offset, sizeof(v));
	return v;
}

void Near(float a, float b) {
	VERIFY(std::abs(a - b) <= 1.0e-4f);
}

void HeadsetPose() {
	std::array<uint8_t, 0x160> bytes {};
	bytes.fill(0xcc);
	const std::array<float, 3> position {1.0f, 2.0f, -3.0f};
	const std::array<float, 4> orientation {0.0f, 0.0f, 0.0f, 1.0f};
	const uint64_t time = 0x1234567812345678ull;
	VERIFY(Write(bytes, position, orientation, time, true));
	VERIFY(ReadU32(bytes, 0x04) == 1u);
	VERIFY(ReadU32(bytes, 0x0c) == 2u);
	VERIFY(ReadU32(bytes, 0x18) == 1u);
	uint64_t decoded_time = 0;
	std::memcpy(&decoded_time, bytes.data() + 0x10, sizeof(decoded_time));
	VERIFY(decoded_time == time);
	Near(ReadFloat(bytes, 0x60), 1.0f);
	Near(ReadFloat(bytes, 0x64), 2.0f);
	Near(ReadFloat(bytes, 0x68), -3.0f);
	Near(ReadFloat(bytes, 0xa0), 1.0f - kInterpupillaryDistance * 0.5f);
	Near(ReadFloat(bytes, 0xe0), 1.0f + kInterpupillaryDistance * 0.5f);
	Near(ReadFloat(bytes, 0x120), 1.0f);
	for (size_t pose : {size_t {0x60}, size_t {0xa0}, size_t {0xe0}, size_t {0x120}}) {
		Near(ReadFloat(bytes, pose + 0x10), 0.0f);
		Near(ReadFloat(bytes, pose + 0x14), 0.0f);
		Near(ReadFloat(bytes, pose + 0x18), 0.0f);
		Near(ReadFloat(bytes, pose + 0x1c), 1.0f);
	}
	// Unknown native fields and result padding must remain untouched.
	VERIFY(bytes[0x00] == 0xcc && bytes[0x1c] == 0xcc);
	VERIFY(bytes[0x150] == 0xcc);
}

void RotatedEyes() {
	std::array<uint8_t, 0x160> bytes {};
	bytes.fill(0xee);
	constexpr float s = 0.70710678118f;
	VERIFY(Write(bytes, {0.0f, 1.6f, 0.0f}, {0.0f, s, 0.0f, s}, 0u, true));
	Near(ReadFloat(bytes, 0xa0), 0.0f);
	Near(ReadFloat(bytes, 0xe0), 0.0f);
	Near(ReadFloat(bytes, 0xa8), kInterpupillaryDistance * 0.5f);
	Near(ReadFloat(bytes, 0xe8), -kInterpupillaryDistance * 0.5f);
	Near(ReadFloat(bytes, 0xa4), 1.6f);
	Near(ReadFloat(bytes, 0xe4), 1.6f);
}

void ControllerPose() {
	std::array<uint8_t, 0x160> bytes {};
	VERIFY(Write(bytes, {0.2f, 1.1f, -0.4f}, {0.0f, 0.0f, 0.0f, 5.0f},
	             123u, false));
	for (size_t pose : {size_t {0x60}, size_t {0xa0}, size_t {0xe0}, size_t {0x120}}) {
		Near(ReadFloat(bytes, pose), 0.2f);
		Near(ReadFloat(bytes, pose + 4), 1.1f);
		Near(ReadFloat(bytes, pose + 8), -0.4f);
		Near(ReadFloat(bytes, pose + 0x1c), 1.0f);
	}
}

void FailClosed() {
	std::array<uint8_t, 0x160> bytes {};
	bytes.fill(0xaa);
	auto truncated = std::span(bytes.data(), kResultBytes - 1u);
	VERIFY(!Write(truncated, {1.0f, 2.0f, 3.0f},
	              {0.0f, 0.0f, 0.0f, 1.0f}, 100u, true));
	VERIFY(bytes[4] == 0xaa);
	const float infinity = std::numeric_limits<float>::infinity();
	VERIFY(!Write(bytes, {infinity, 2.0f, 3.0f},
	              {0.0f, 0.0f, 0.0f, 1.0f}, 100u, true));
	VERIFY(bytes[4] == 0xaa);
	const float nan = std::numeric_limits<float>::quiet_NaN();
	VERIFY(Write(bytes, {1.0f, 2.0f, 3.0f},
	             {nan, 0.0f, 0.0f, 0.0f}, 100u, true));
	Near(ReadFloat(bytes, 0x7c), 1.0f);
}
} // namespace

int main() {
	HeadsetPose();
	RotatedEyes();
	ControllerPose();
	FailClosed();
	std::puts("Hmd2TrackerPoseTests: passed");
}
