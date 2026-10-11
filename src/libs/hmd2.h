#ifndef EMULATOR_SRC_LIBS_HMD2_H_
#define EMULATOR_SRC_LIBS_HMD2_H_

#include <atomic>
#include <cstdint>

namespace Libs::Hmd2 {

inline constexpr uint32_t PANEL_WIDTH  = 4000;
inline constexpr uint32_t PANEL_HEIGHT = 2040;
inline constexpr uint64_t OUTPUT_MODE_89_91HZ  = 0x2000c;
inline constexpr uint64_t OUTPUT_MODE_119_88HZ = 0x2000f;
inline constexpr int32_t ERROR_REPROJECTION_NOT_INITIALIZED = -1972240373; // 0x8a72000b

struct ReprojectionState {
	std::atomic<uint64_t> output_mode {0};
	std::atomic<uint32_t> timing_us {3000};
};

struct ReprojectionPreviewSource {
    uint64_t address = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t format = 0;
    uint32_t tile_mode = 0;
};

// Experimental desktop mirror of the reprojection source image, not the
// panel surface. Populated only with KYTY_VR_MIRROR_SOURCE=1.
[[nodiscard]] bool GetReprojectionPreviewSource(ReprojectionPreviewSource* source);
[[nodiscard]] ReprojectionState* GetReprojectionState();

} // namespace Libs::Hmd2

#endif // EMULATOR_SRC_LIBS_HMD2_H_
