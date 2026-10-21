#pragma once

// Layout observed in the guest's sceVrTracker2GetResult consumer.
// This helper constructs the pose portion of a native PS VR2 tracking result.
// It is intentionally independent of the host renderer and guest memory APIs
// so all arithmetic and offsets can be tested without running an emulator.

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>

namespace Libs::Hmd2TrackerPose {

inline constexpr size_t kResultBytes = 0x140u;
inline constexpr float  kInterpupillaryDistance = 0.063f;

struct Pose {
	std::array<float, 3> position;
	std::array<float, 4> quaternion;
};

[[nodiscard]] inline std::array<float, 4> Normalize(std::array<float, 4> q) {
	float magnitude_squared = 0.0f;
	for (const auto component : q) {
		if (!std::isfinite(component)) {
			return {0.0f, 0.0f, 0.0f, 1.0f};
		}
		magnitude_squared += component * component;
	}
	if (!std::isfinite(magnitude_squared) || magnitude_squared < 1.0e-8f) {
		return {0.0f, 0.0f, 0.0f, 1.0f};
	}
	const float inverse_length = 1.0f / std::sqrt(magnitude_squared);
	for (auto& component : q) {
		component *= inverse_length;
	}
	return q;
}

// Rotate the headset-local left/right eye displacement into tracker space.
[[nodiscard]] inline std::array<float, 3> EyePosition(
    std::array<float, 3> position, const std::array<float, 4>& q,
    float local_x) {
	const auto [x, y, z, w] = q;
	position[0] += (1.0f - 2.0f * (y * y + z * z)) * local_x;
	position[1] += 2.0f * (x * y + z * w) * local_x;
	position[2] += 2.0f * (x * z - y * w) * local_x;
	return position;
}

[[nodiscard]] inline bool Write(std::span<uint8_t> out,
                               std::array<float, 3> position,
                               std::array<float, 4> orientation,
                               uint64_t predicted_time, bool headset) {
	if (out.size() < kResultBytes) {
		return false;
	}
	for (const auto coordinate : position) {
		if (!std::isfinite(coordinate)) {
			return false;
		}
	}
	orientation = Normalize(orientation);
	const uint32_t valid = 1u;
	const uint32_t mode  = 2u; // 6DoF (both translation and orientation)
	const uint32_t pose_state = 1u;
	// The guest normalizes both quality fields by 9 and discards controller
	// translation when position confidence is zero, even with a valid pose.
	const uint32_t confidence = 9u;
	std::memcpy(out.data() + 0x04u, &valid, sizeof(valid));
	std::memcpy(out.data() + 0x0cu, &mode, sizeof(mode));
	std::memcpy(out.data() + 0x10u, &predicted_time, sizeof(predicted_time));
	std::memcpy(out.data() + 0x18u, &pose_state, sizeof(pose_state));
	std::memcpy(out.data() + 0x1cu, &confidence, sizeof(confidence));
	std::memcpy(out.data() + 0x20u, &confidence, sizeof(confidence));

	const auto write_pose = [&](size_t offset, const std::array<float, 3>& p) {
		std::memcpy(out.data() + offset, p.data(), 3u * sizeof(float));
		std::memcpy(out.data() + offset + 0x10u, orientation.data(),
		            4u * sizeof(float));
	};
	write_pose(0x60u, position);
	const float half_ipd = kInterpupillaryDistance * 0.5f;
	write_pose(0xa0u, headset ? EyePosition(position, orientation, -half_ipd) : position);
	write_pose(0xe0u, headset ? EyePosition(position, orientation, +half_ipd) : position);
	write_pose(0x120u, position);
	return true;
}

} // namespace Libs::Hmd2TrackerPose
