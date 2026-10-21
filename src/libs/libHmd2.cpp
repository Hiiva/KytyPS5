#include "common/assert.h"
#include "common/emulatorConfig.h"
#include "graphics/guest_gpu/tile.h"
#include "graphics/host_gpu/hostMemory.h"
#include "libs/agc.h"
#include "libs/errno.h"
#include "libs/hmd2.h"
#include "libs/hmd2LayerDescriptor.h"
#include "libs/hmd2TrackerPose.h"
#include "libs/controller.h"
#include "kernel/pthread.h"
#include "libs/libs.h"
#include "loader/guestInstructionPatcher.h"
#include "loader/symbolDatabase.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <new>

namespace Libs {

LIB_VERSION("Hmd2", 1, "Hmd2", 1, 1);

namespace Hmd2 {

constexpr int32_t ERROR_ALREADY_INITIALIZED = -1972240383; // 0x8a720001
constexpr int32_t ERROR_NOT_INITIALIZED     = -1972240382; // 0x8a720002
constexpr int32_t ERROR_PARAMETER_NULL      = -1972240376; // 0x8a720008
constexpr int32_t ERROR_REPROJECTION_ALREADY_INITIALIZED = -1972240372; // 0x8a72000c
constexpr int32_t ERROR_REPROJECTION_IN_VR_MODE = -1972240356; // 0x8a72001c

struct SceHmd2InitializeParam {
	void*   reserved0;
	uint8_t reserved[8];
};

struct SceHmd2DeviceInformation {
	uint32_t status;
	uint32_t reserve0;
	struct {
		struct {
			uint32_t width;
			uint32_t height;
		} panelResolution;
		struct {
			uint16_t refreshRate90Hz;
			uint16_t refreshRate120Hz;
		} flipToDisplayLatency;
	} deviceInfo;
	uint8_t hmuMount;
	uint8_t lensSeparationDistance;
	uint8_t reserve1[2];
	float   virtualImageDistance;
};

struct SceHmd2FieldOfView {
	float tanOut;
	float tanIn;
	float tanTop;
	float tanBottom;
};

struct SceHmd2ReprojectionInitializeParam {
	void*    pReprojectionBuff;
	void*    pDisplayBuff;
	int32_t  threadPriority;
	int32_t  cpuAffinityMask;
	uint32_t pipeId;
	uint32_t queueId;
	uint32_t type;
	void*    pSeeThroughBuff;
	int32_t  reprojectionTiming;
	uint32_t reserved[5];
};

static_assert(sizeof(SceHmd2InitializeParam) == 16);
static_assert(offsetof(SceHmd2InitializeParam, reserved) == 8);
static_assert(sizeof(SceHmd2DeviceInformation) == 28);
static_assert(offsetof(SceHmd2DeviceInformation, deviceInfo) == 8);
static_assert(offsetof(SceHmd2DeviceInformation, hmuMount) == 20);
static_assert(offsetof(SceHmd2DeviceInformation, virtualImageDistance) == 24);
static_assert(sizeof(SceHmd2FieldOfView) == 16);
static_assert(alignof(SceHmd2FieldOfView) == 4);
static_assert(sizeof(SceHmd2ReprojectionInitializeParam) == 72);
static_assert(alignof(SceHmd2ReprojectionInitializeParam) == 8);
static_assert(offsetof(SceHmd2ReprojectionInitializeParam, threadPriority) == 16);
static_assert(offsetof(SceHmd2ReprojectionInitializeParam, type) == 32);
static_assert(offsetof(SceHmd2ReprojectionInitializeParam, pSeeThroughBuff) == 40);
static_assert(offsetof(SceHmd2ReprojectionInitializeParam, reprojectionTiming) == 48);
static_assert(offsetof(SceHmd2ReprojectionInitializeParam, reserved) == 52);

static std::atomic<bool> g_initialized = false;
static std::mutex g_reprojection_mutex;
static std::mutex g_preview_source_mutex;
static ReprojectionPreviewSource g_preview_source {};

bool GetReprojectionPreviewSource(ReprojectionPreviewSource* source) {
    if (source == nullptr) {
        return false;
    }
    std::scoped_lock lock {g_preview_source_mutex};
    if (g_preview_source.address == 0) {
        return false;
    }
    *source = g_preview_source;
    return true;
}

static std::atomic<ReprojectionState*> g_reprojection_state {nullptr};

ReprojectionState* GetReprojectionState() {
	return g_reprojection_state.load(std::memory_order_acquire);
}

static Graphics::SizeAlign ReprojectionDisplaySize() {
	// The host display target is one uncompressed tiled 10-bit panel surface.
	Graphics::TileSizeAlign size {};
	Graphics::TileGetTextureTotalSize(Graphics::Prospero::BufferFormat::k10_10_10_2UNorm,
	                                  PANEL_WIDTH, PANEL_HEIGHT, 1, 1,
	                                  Graphics::Prospero::TileMode::kRenderTarget, false, size);
	EXIT_IF(size.size == 0 || size.align == 0);
	return {size.size, size.align};
}

static int32_t KYTY_SYSV_ABI Hmd2Initialize(const SceHmd2InitializeParam* param) {
	PRINT_NAME();
	if (param == nullptr) {
		return ERROR_PARAMETER_NULL;
	}
	if (g_initialized.exchange(true)) {
		return ERROR_ALREADY_INITIALIZED;
	}
	LOGF("Hmd2: initialized, virtual headset=%s\n", Config::VrEnabled() ? "enabled" : "disabled");
	return OK;
}

static int32_t KYTY_SYSV_ABI Hmd2GetDeviceInformation(SceHmd2DeviceInformation* info) {
	PRINT_NAME();
	if (!g_initialized.load()) {
		return ERROR_NOT_INITIALIZED;
	}
	if (info == nullptr) {
		return ERROR_PARAMETER_NULL;
	}
	*info = {};
	if (Config::VrEnabled()) {
		// Virtual headset defaults: READY, MOUNT, 63 mm lens separation.
		info->hmuMount               = 1;
		info->lensSeparationDistance = 63;
		// Headset characteristics: 4000x2040 panel and 2 m virtual image distance.
		info->deviceInfo.panelResolution = {PANEL_WIDTH, PANEL_HEIGHT};
		info->virtualImageDistance       = 2.0f;
		// Host presentation has no emulated headset scan-out delay; latency remains zero.
	} else {
		info->status = 2; // SCE_HMD2_DEVICE_STATUS_NOT_DETECTED
	}
	static std::atomic<uint32_t> info_logs {0};
	if (info_logs.fetch_add(1u, std::memory_order_relaxed) < 8u) {
		LOGF("Hmd2: device status=%u panel=%ux%u mounted=%u\n", info->status,
	     info->deviceInfo.panelResolution.width, info->deviceInfo.panelResolution.height,
	     info->hmuMount);
	}
	return OK;
}

static int32_t KYTY_SYSV_ABI Hmd2GetFieldOfViewWithoutHandle(SceHmd2FieldOfView* fov) {
	PRINT_NAME();
	if (!g_initialized.load()) {
		return ERROR_NOT_INITIALIZED;
	}
	if (fov == nullptr) {
		return ERROR_PARAMETER_NULL;
	}
	// Virtual ER15 field of view: 55/42.5/51.2/51.2 degrees.
	// These tangents describe that geometry, not bit-exact firmware output.
	*fov = {1.4281480312f, 0.9163311720f, 1.2437491417f, 1.2437491417f};
	LOGF("Hmd2: FOV tangents out=%f in=%f top=%f bottom=%f\n", fov->tanOut, fov->tanIn,
	     fov->tanTop, fov->tanBottom);
	return OK;
}

static int32_t KYTY_SYSV_ABI Hmd2Open(int32_t user_id, int32_t type, int32_t index,
                                      const void* param) {
    if (!g_initialized.load(std::memory_order_acquire) || !Config::VrEnabled()) {
        return ERROR_NOT_INITIALIZED;
    }
    LOGF("Hmd2: open user=%d type=%d index=%d param=%p\n", user_id, type, index, param);
    const char* requested = std::getenv("KYTY_VR_EXPERIMENTAL_HMD");
    // Keep the known-working virtual HMD handle 0 as the default.
    // Handle 1 enables the guest's headset tracking path for experiments.
    return requested != nullptr && std::strcmp(requested, "1") == 0 ? 1 : 0;
}

static int32_t KYTY_SYSV_ABI Hmd2GetFieldOfView(int32_t handle, SceHmd2FieldOfView* fov) {
    if (handle != 0) {
        LOGF("Hmd2: unexpected FOV handle=%d\n", handle);
    }
    return Hmd2GetFieldOfViewWithoutHandle(fov);
}

static int32_t KYTY_SYSV_ABI Hmd2GetDeviceInformationByHandle(
    int32_t handle, SceHmd2DeviceInformation* info) {
    if (handle != 0) {
        LOGF("Hmd2: unexpected info handle=%d\n", handle);
    }
    return Hmd2GetDeviceInformation(info);
}

static std::atomic<uint64_t> g_last_virtual_frame {0};

static int32_t KYTY_SYSV_ABI Hmd2ReprojectionBeginFrame(uint64_t frame) {
    if (GetReprojectionState() == nullptr) {
        return ERROR_REPROJECTION_NOT_INITIALIZED;
    }
    Loader::DrainHorizonKismetTrace();
    g_last_virtual_frame.store(frame, std::memory_order_relaxed);
    return OK;
}

static int32_t KYTY_SYSV_ABI Hmd2ReprojectionGetPredictedDisplayTime(
    uint64_t frame, uint64_t* time) {
    if (time == nullptr) {
        return ERROR_PARAMETER_NULL;
    }
    // Prediction is in the kernel process-clock domain, in microseconds.
    *time = LibKernel::KernelGetProcessTime() + 11000u;
    static std::atomic<uint32_t> logs {0};
    const auto n = logs.fetch_add(1u, std::memory_order_relaxed);
    if (n < 4u) {
        LOGF("Hmd2: predicted display frame=%" PRIu64 " time=%" PRIu64 "\n", frame, *time);
    }
    return OK;
}

static int32_t KYTY_SYSV_ABI Hmd2ReprojectionSetParam(
    uint32_t index, uint64_t frame, const void* param, const void* trace) {
    if (GetReprojectionState() == nullptr) {
        return ERROR_REPROJECTION_NOT_INITIALIZED;
    }
    if (param == nullptr) {
        return ERROR_PARAMETER_NULL;
    }
    static std::atomic<uint32_t> logs {0};
    if (logs.fetch_add(1u, std::memory_order_relaxed) < 4u) {
        LOGF("Hmd2: SetParam index=%u frame=%" PRIu64 " param=%p trace=%p\n",
             index, frame, param, trace);
    }
    if (Config::VrEnabled()) {
        const char* mirror = std::getenv("KYTY_VR_MIRROR_SOURCE");
        if (mirror != nullptr && mirror[0] == '1' &&
            Graphics::HostMemoryRangeIsReadable(reinterpret_cast<uint64_t>(param), 24u)) {
            uint64_t layers = 0;
            uint32_t count = 0;
            const auto* const bytes = static_cast<const uint8_t*>(param);
            std::memcpy(&layers, bytes, sizeof(layers));
            std::memcpy(&count, bytes + 8, sizeof(count));
            if (count >= 1u && count <= 4u &&
                Graphics::HostMemoryRangeIsReadable(layers, 0xe0u)) {
                uint64_t source_block = 0;
                std::memcpy(&source_block,
                            reinterpret_cast<const uint8_t*>(layers) + 0xd8u,
                            sizeof(source_block));
                if (Graphics::HostMemoryRangeIsReadable(source_block, 40u)) {
                    // The nested +8 payload is the same native TSharp
                    // texture descriptor used by type-0x10 video-out layers.
                    // Use the tested decoder for both paths.
                    Hmd2LayerDescriptor::Texture texture {};
                    const bool decoded = Hmd2LayerDescriptor::DecodeTexture(
                        std::span<const uint8_t>(
                            reinterpret_cast<const uint8_t*>(source_block) + 8u, 32u),
                        &texture);
                    if (std::getenv("KYTY_VR_DUMP_SOURCE") != nullptr) {
                        static std::atomic<uint32_t> dump_count {0};
                        if (dump_count.fetch_add(1u, std::memory_order_relaxed) < 2u) {
                            const auto* nested_words =
                                reinterpret_cast<const uint32_t*>(source_block);
                            if (Graphics::HostMemoryRangeIsReadable(source_block, 0xe0u)) {
                                for (uint32_t k = 0; k < 0xe0u / 4u; k += 4u) {
                                    LOGF("Hmd2: nested-data[%02x] %08x %08x %08x %08x\n",
                                         k * 4u, nested_words[k], nested_words[k + 1],
                                         nested_words[k + 2], nested_words[k + 3]);
                                }
                                const uint32_t follow_offsets[] = {0x68u, 0x98u, 0xa0u};
                                for (uint32_t offset : follow_offsets) {
                                    uint64_t ptr = 0;
                                    std::memcpy(&ptr,
                                                reinterpret_cast<const uint8_t*>(source_block) + offset,
                                                sizeof(ptr));
                                    const bool readable =
                                        Graphics::HostMemoryRangeIsReadable(ptr, 0x80u);
                                    LOGF("Hmd2: nested follow +%02x addr=%016" PRIx64
                                         " readable=%d\n", offset, ptr, readable);
                                    if (readable) {
                                        const auto* words =
                                            reinterpret_cast<const uint32_t*>(ptr);
                                        for (uint32_t k = 0; k < 0x80u / 4u; k += 4u) {
                                            LOGF("Hmd2: follow+%02x[%02x] %08x %08x %08x %08x\n",
                                                 offset, k * 4u, words[k], words[k + 1],
                                                 words[k + 2], words[k + 3]);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    // Observed Horizon source: 3328x1664, R32_FLOAT, tiled 0x1b.
                    // The format may contain a packed four-channel output;
                    // the diagnostic display interprets it as RGBA8.
                    if (decoded && (texture.address & 0xffffu) == 0u &&
                        texture.width <= 8192u && texture.height <= 8192u &&
                        texture.format == 22u && texture.tile_mode == 0x1bu &&
                        !texture.compressed) {
                        std::scoped_lock lock {g_preview_source_mutex};
                        g_preview_source = {texture.address, texture.width,
                                            texture.height, texture.format,
                                            texture.tile_mode};
                        static std::atomic<uint32_t> preview_logs {0};
                        if (preview_logs.fetch_add(1u, std::memory_order_relaxed) < 4u) {
                            LOGF("Hmd2: direct-preview source addr=%016" PRIx64
                                 " extent=%ux%u guestfmt=%u tile=%u\n",
                                 texture.address, texture.width, texture.height,
                                 texture.format, texture.tile_mode);
                        }
                    }
                }
            }
        }
    }
    // The guest submits per-eye texture layers here. Full PS VR2
    // reprojection/compositing has not yet been implemented. Desktop mode
    // can only show surfaces rendered directly into the video-out buffers.
    return OK;
}

static int32_t KYTY_SYSV_ABI Hmd2ReprojectionSetRenderConfig(
    uint64_t half_rate, uint64_t reserved0, uint64_t reserved1) {
    LOGF("Hmd2: SetRenderConfig rate=%" PRIu64 " reserved=%" PRIu64 ",%" PRIu64 "\n",
         half_rate, reserved0, reserved1);
    return OK;
}
// Headset mirroring and optional gaze requests have no physical device backing.
// These preserve the firmware success path without assuming unknown result layouts.
static int32_t KYTY_SYSV_ABI Hmd2ReprojectionSetMirroringOption(const void* option) {
    static std::atomic_bool logged = false;
    if (!logged.exchange(true, std::memory_order_relaxed)) {
        LOGF("Hmd2: mirroring option requested: %p\n", option);
    }
    return OK;
}

static int32_t KYTY_SYSV_ABI Hmd2ReprojectionEnableMirroring(const void* param) {
    LOGF("Hmd2: mirroring enabled, param=%p\n", param);
    return OK;
}

// 0x38-byte native PS VR2 gaze result layout, as observed in PS5VR.
// On a desktop without an eye tracker, use the headset's forward vector.
static int32_t KYTY_SYSV_ABI Hmd2GazeGetResult(const void* param, void* result) {
    if (param == nullptr || result == nullptr) {
        return ERROR_PARAMETER_NULL;
    }
    const auto request_address = reinterpret_cast<uint64_t>(param);
    const auto result_address = reinterpret_cast<uint64_t>(result);
    if (!Graphics::HostMemoryRangeIsReadable(request_address, 16) ||
        !Graphics::HostMemoryRangeIsReadable(result_address, 0x38)) {
        return ERROR_PARAMETER_NULL;
    }
    const auto* const request = static_cast<const uint8_t*>(param);
    auto* const output = static_cast<uint8_t*>(result);
    uint32_t request_size = 0;
    uint32_t result_size = 0;
    int32_t handle = 0;
    std::memcpy(&request_size, request, sizeof(request_size));
    std::memcpy(&handle, request + 4, sizeof(handle));
    std::memcpy(&result_size, output, sizeof(result_size));
    static std::atomic<uint32_t> logs {0};
    if (logs.fetch_add(1u, std::memory_order_relaxed) < 8u) {
        LOGF("Hmd2: gaze result request size=0x%x hmd=%d result size=0x%x\n",
             request_size, handle, result_size);
    }
    if (request_size != 0x10 || result_size < 0x38) {
        return ERROR_PARAMETER_NULL;
    }
    const uint32_t status = 0;
    const uint32_t valid = 1;
    // Horizon initializes this native gaze vector along +Z before the call.
    // Its VR conversion applies the PS5-to-engine sign change afterward.
    const float forward[3] = {0.0f, 0.0f, 1.0f};
    std::memcpy(output + 4, &status, sizeof(status));
    std::memcpy(output + 0x10, &valid, sizeof(valid));
    std::memcpy(output + 0x14, forward, sizeof(forward));
    return OK;
}

static int32_t KYTY_SYSV_ABI Hmd2GazeGetResultForFoveatedRendering(
    const void* request, void* result) {
    // Native UE 4.26 Horizon requests 0x10 bytes and a 0x18-byte result.
    // Its caller consumes the normalized +Z gaze vector at result +0x08.
    // Without PS VR2 eye tracking, supply a stable center-of-view gaze.
    if (request == nullptr || result == nullptr ||
        !Graphics::HostMemoryRangeIsReadable(reinterpret_cast<uint64_t>(request), 0x10u) ||
        !Graphics::HostMemoryRangeIsReadable(reinterpret_cast<uint64_t>(result), 0x18u)) {
        return ERROR_PARAMETER_NULL;
    }
    const auto* input = static_cast<const uint8_t*>(request);
    auto* output = static_cast<uint8_t*>(result);
    uint32_t request_size = 0;
    uint32_t result_size = 0;
    std::memcpy(&request_size, input, sizeof(request_size));
    std::memcpy(&result_size, output, sizeof(result_size));
    if (request_size != 0x10u || result_size < 0x18u) {
        return ERROR_PARAMETER_NULL;
    }
    const uint32_t valid = 1u;
    const float direction[3] = {0.0f, 0.0f, 1.0f};
    std::memcpy(output + 4u, &valid, sizeof(valid));
    std::memcpy(output + 8u, direction, sizeof(direction));
    static std::atomic<uint32_t> logs {0};
    if (logs.fetch_add(1u, std::memory_order_relaxed) < 4u) {
        LOGF("Hmd2: virtual foveated gaze request size=0x%x result size=0x%x\n",
             request_size, result_size);
    }
    return OK;
}
static std::atomic<bool> g_tracker_initialized {false};

static int32_t KYTY_SYSV_ABI VrTracker2QueryMemory(const void* param, uint32_t* output) {
    if (param == nullptr || output == nullptr) {
        return ERROR_PARAMETER_NULL;
    }
    // Query result contains a size and alignment in DWORD slots 1 and 2.
    // The guest will allocate and pass this memory to Initialize.
    output[0] = 0x20u;
    output[1] = 0x100000u;
    output[2] = 0x10000u;
    LOGF("VrTracker2: QueryMemory size=%u align=%u\n", output[1], output[2]);
    return OK;
}

static int32_t KYTY_SYSV_ABI VrTracker2Initialize(const void* param) {
    if (param == nullptr) {
        return ERROR_PARAMETER_NULL;
    }
    g_tracker_initialized.store(true, std::memory_order_release);
    LOGF("VrTracker2: Initialize param=%p\n", param);
    return OK;
}

static int32_t KYTY_SYSV_ABI VrTracker2RegisterDevice(int32_t type, int32_t handle) {
    static std::atomic<uint32_t> logs {0};
    if (logs.fetch_add(1u, std::memory_order_relaxed) < 8u) {
        LOGF("VrTracker2: RegisterDevice type=%d handle=%d\n", type, handle);
    }
    return g_tracker_initialized.load(std::memory_order_acquire) ? OK : ERROR_NOT_INITIALIZED;
}

static int32_t KYTY_SYSV_ABI VrTracker2SetCoordinateSystem(uint64_t coordinate) {
    LOGF("VrTracker2: SetCoordinateSystem value=0x%" PRIx64 "\n", coordinate);
    return OK;
}

static int32_t KYTY_SYSV_ABI VrTracker2ResetLocalCoordinate() {
    return OK;
}

static int32_t KYTY_SYSV_ABI VrTracker2GetResult(const void* param, void* result) {
    if (param == nullptr || result == nullptr) {
        return ERROR_PARAMETER_NULL;
    }
    auto* const in = reinterpret_cast<const uint8_t*>(param);
    auto* const out = reinterpret_cast<uint8_t*>(result);
    const int32_t handle = *reinterpret_cast<const int32_t*>(in + 4);
    const uint32_t param_size = *reinterpret_cast<const uint32_t*>(in);
    static std::array<std::atomic<uint32_t>, 5> result_logs {};
    const auto log_index = handle >= 0 && handle < 4 ? handle : 4;
    const auto n = result_logs[static_cast<size_t>(log_index)].fetch_add(1u, std::memory_order_relaxed);
    if (n < 4u) {
        LOGF("VrTracker2: GetResult handle=%d paramsize=0x%x result=%p first=0x%x caller=%p in=%p\n",
             handle, param_size, result, *reinterpret_cast<const uint32_t*>(out),
             __builtin_return_address(0), param);
    }
    if (n < 4u && Graphics::HostMemoryRangeIsReadable(
            reinterpret_cast<uint64_t>(param), 0x30u)) {
        const auto* q = reinterpret_cast<const uint32_t*>(in);
        LOGF("VrTracker2: request [%08x %08x %08x %08x | %08x %08x %08x %08x | %08x %08x %08x %08x]\n",
             q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11]);
    }
    // TEMP investigation: test the three guest-consumed validity fields
    // separately, without changing default desktop behavior.
    if (const char* quality = std::getenv("KYTY_VR_TRACKING_QUALITY");
        quality != nullptr) {
        const bool all = std::strcmp(quality, "all") == 0 || std::strcmp(quality, "1") == 0;
        uint32_t mode = 0;
        if (all || std::strcmp(quality, "mode2") == 0 ||
            std::strcmp(quality, "mode2flag") == 0) {
            mode = 2;
        } else if (std::strcmp(quality, "mode1") == 0) {
            mode = 1;
        }
        if (mode != 0) {
            std::memcpy(out + 0x0c, &mode, sizeof(mode));
        }
        if (all || std::strcmp(quality, "stamp") == 0) {
            uint64_t timestamp = 0;
            std::memcpy(&timestamp, in + 0x10, sizeof(timestamp));
            std::memcpy(out + 0x10, &timestamp, sizeof(timestamp));
        }
        if (all || std::strcmp(quality, "flag") == 0 ||
            std::strcmp(quality, "mode2flag") == 0) {
            const uint32_t valid_pose = 1;
            std::memcpy(out + 0x18, &valid_pose, sizeof(valid_pose));
        }
    }
    // PS VR2 tracker result: status at +4; head position at +0x60, quaternion
    // at +0x70, and per-eye orientation quaternions at +0xb0 and +0xf0.
    // Only these fields are written, avoiding a blind guest-buffer memset.
    *reinterpret_cast<uint32_t*>(out + 4) = 1u;
    float head_pos[3] = {};
    float quat[4] = {};
    Controller::VirtualHeadGetPose(head_pos, quat);
    if (handle == 2 || handle == 3) {
        Controller::VirtualHandGetPose(handle == 2 ? 0 : 1, head_pos, quat);
    }
    static std::atomic<uint32_t> pose_logs {0};
    if (pose_logs.fetch_add(1u, std::memory_order_relaxed) < 4u) {
        LOGF("Virtual tracking handle=%d position=(%.2f, %.2f, %.2f) quaternion=(%.2f, %.2f, %.2f, %.2f)\n",
             handle, head_pos[0], head_pos[1], head_pos[2],
             quat[0], quat[1], quat[2], quat[3]);
    }
    // Horizon rejects controller poses with missing state or confidence.
    // Initialize their complete result by default, leaving headset tracking
    // opt-in. An explicit 0 retains partial results for diagnostics.
    if (const char* full = std::getenv("KYTY_VR_FULL_TRACKING_RESULT");
        (full != nullptr && std::strcmp(full, "1") == 0) ||
        ((handle == 2 || handle == 3) && (full == nullptr || full[0] == '\0'))) {
        const bool headset = handle == 1;
        const size_t result_bytes = headset ? Hmd2TrackerPose::kHeadsetResultBytes
                                           : Hmd2TrackerPose::kResultBytes;
        if (!Graphics::HostMemoryRangeIsReadable(
                reinterpret_cast<uint64_t>(param), 0x20u) ||
            !Graphics::HostMemoryRangeIsReadable(
                reinterpret_cast<uint64_t>(result), result_bytes)) {
            return ERROR_PARAMETER_NULL;
        }
        uint64_t predicted_time = 0;
        std::memcpy(&predicted_time, in + 0x10u, sizeof(predicted_time));
        const std::array<float, 3> position {head_pos[0], head_pos[1], head_pos[2]};
        const std::array<float, 4> orientation {quat[0], quat[1], quat[2], quat[3]};
        if (!Hmd2TrackerPose::Write(
                std::span<uint8_t>(out, result_bytes), position,
                orientation, predicted_time, headset)) {
            return ERROR_PARAMETER_NULL;
        }
        static std::atomic<uint32_t> full_logs {0};
        if (full_logs.fetch_add(1u, std::memory_order_relaxed) < 4u) {
            LOGF("VrTracker2: complete desktop 6DoF pose handle=%d time=%" PRIu64
                 " ipd=%.4f m\n", handle, predicted_time,
                 Hmd2TrackerPose::kInterpupillaryDistance);
        }
        return OK;
    }
    std::memcpy(out + 0x60, head_pos, sizeof(head_pos));
    std::memcpy(out + 0x70, quat, sizeof(quat));
    std::memcpy(out + 0xb0, quat, sizeof(quat));
    std::memcpy(out + 0xf0, quat, sizeof(quat));
    return OK;
}

static int32_t KYTY_SYSV_ABI VrTracker2Finalize() {
    g_tracker_initialized.store(false, std::memory_order_release);
    return OK;
}
// Horizon calls these after reprojection initialization. The guest saves RAX/RDX
// and uses them as size/alignment for two direct-memory allocations. The firmware
// sizes are unknown; these conservative values are boot probes, not an API contract.
static Graphics::SizeAlign KYTY_SYSV_ABI Hmd2BootQueryBufferA() {
    const Graphics::SizeAlign size {0x100000u, 0x10000u};
    LOGF("Hmd2: provisional buffer A size=%" PRIu64 " align=%zu\n", size.m_size, size.m_align);
    return size;
}

static Graphics::SizeAlign KYTY_SYSV_ABI Hmd2BootQueryBufferB() {
    const Graphics::SizeAlign size {0x100000u, 0x10000u};
    LOGF("Hmd2: provisional buffer B size=%" PRIu64 " align=%zu\n", size.m_size, size.m_align);
    return size;
}
static Graphics::SizeAlign KYTY_SYSV_ABI Hmd2ReprojectionQueryBufferSizeAlign() {
	const Graphics::SizeAlign size {sizeof(ReprojectionState), alignof(ReprojectionState)};
	LOGF("Hmd2: reprojection work size=%" PRIu64 " alignment=%zu\n", size.m_size, size.m_align);
	return size;
}

static Graphics::SizeAlign KYTY_SYSV_ABI Hmd2ReprojectionQueryDisplayBufferSizeAlign() {
	const auto size = ReprojectionDisplaySize();
	LOGF("Hmd2: reprojection display size=%" PRIu64 " alignment=%zu\n", size.m_size,
	     size.m_align);
	return size;
}

static int32_t KYTY_SYSV_ABI
Hmd2ReprojectionInitialize(const SceHmd2ReprojectionInitializeParam* param, void*) {
	std::scoped_lock lock {g_reprojection_mutex};
	if (GetReprojectionState() != nullptr) {
		return ERROR_REPROJECTION_ALREADY_INITIALIZED;
	}
	if (param == nullptr || param->pReprojectionBuff == nullptr || param->pDisplayBuff == nullptr) {
		return ERROR_PARAMETER_NULL;
	}
	EXIT_NOT_IMPLEMENTED(param->pSeeThroughBuff != nullptr);
	const auto display_size = ReprojectionDisplaySize();
	auto* state = new (param->pReprojectionBuff) ReprojectionState;
	if (param->reprojectionTiming != 0) {
		state->timing_us.store(static_cast<uint32_t>(param->reprojectionTiming),
		                      std::memory_order_relaxed);
	}
	g_reprojection_state.store(state, std::memory_order_release);
	LOGF("Hmd2: reprojection initialized work=%p display=%p size=%" PRIu64 " timing=%u\n",
	     static_cast<void*>(state), param->pDisplayBuff, display_size.m_size,
	     state->timing_us.load(std::memory_order_relaxed));
	return OK;
}

static int32_t KYTY_SYSV_ABI Hmd2ReprojectionEnableVrMode(uint64_t output_mode) {
	auto* state = GetReprojectionState();
	if (state == nullptr) {
		return ERROR_REPROJECTION_NOT_INITIALIZED;
	}
	if (output_mode != OUTPUT_MODE_89_91HZ && output_mode != OUTPUT_MODE_119_88HZ) {
		return VideoOut::VIDEO_OUT_ERROR_INVALID_VALUE;
	}
	if (!Config::VrEnabled()) {
		return VideoOut::VIDEO_OUT_ERROR_NO_DEVICE;
	}
	uint64_t expected = 0;
	if (!state->output_mode.compare_exchange_strong(expected, output_mode)) {
		return ERROR_REPROJECTION_IN_VR_MODE;
	}
	LOGF("Hmd2: VR output mode=0x%" PRIx64 "\n", output_mode);
	return OK;
}

static int32_t KYTY_SYSV_ABI Hmd2ReprojectionSetTiming(int32_t timing) {
	auto* state = GetReprojectionState();
	if (state == nullptr) {
		return ERROR_REPROJECTION_NOT_INITIALIZED;
	}
	EXIT_NOT_IMPLEMENTED(timing < 2000 || timing > 7000);
	const auto rounded = static_cast<uint32_t>((timing + 99) / 100 * 100);
	state->timing_us.store(rounded, std::memory_order_release);
	LOGF("Hmd2: reprojection timing=%u us\n", rounded);
	return OK;
}

} // namespace Hmd2

LIB_DEFINE(InitHmd2_1) {
	LIB_FUNC("c812oYs7Vsc", Hmd2::Hmd2Initialize);
	LIB_FUNC("f3kPeoTZnIE", Hmd2::Hmd2Open);
	LIB_FUNC("E+b+c1Idyew", Hmd2::Hmd2GetFieldOfView);
	LIB_FUNC("+seEJVlljr0", Hmd2::Hmd2ReprojectionSetMirroringOption);
	LIB_FUNC("Pb1d+j-bBSc", Hmd2::Hmd2BootQueryBufferA);
	LIB_FUNC("wEO+gMHs9NU", Hmd2::Hmd2BootQueryBufferB);
    LIB_FUNC("4BlE4IPXP0Q", Hmd2::Hmd2GetDeviceInformationByHandle);
    LIB_FUNC("Ocf081WpBpA", Hmd2::Hmd2ReprojectionBeginFrame);
    LIB_FUNC("SVEG+1D7qHA", Hmd2::Hmd2ReprojectionGetPredictedDisplayTime);
    LIB_FUNC("Qx7SkcgAzok", Hmd2::Hmd2ReprojectionEnableMirroring);
    LIB_FUNC("retc+-uRMhk", Hmd2::Hmd2GazeGetResultForFoveatedRendering);
    LIB_FUNC("lAoFUedcfqA", Hmd2::Hmd2GazeGetResult);
    LIB_FUNC("xMo9ENEu2E0", Hmd2::Hmd2ReprojectionSetParam);
    LIB_FUNC("hA9LshbSkzw", Hmd2::Hmd2ReprojectionSetRenderConfig);	LIB_FUNC("bIi4YUfSRys", Hmd2::Hmd2GetDeviceInformation);
	LIB_FUNC("gF8+lvc7GuQ", Hmd2::Hmd2GetFieldOfViewWithoutHandle);
	LIB_FUNC("U-CnbmeyYaA", Hmd2::Hmd2ReprojectionQueryBufferSizeAlign);
	LIB_FUNC("-C2nkoEYOnU", Hmd2::Hmd2ReprojectionQueryDisplayBufferSizeAlign);
	LIB_FUNC("C0rPwER-yxg", Hmd2::Hmd2ReprojectionInitialize);
	LIB_FUNC("VVvFh51o20s", Hmd2::Hmd2ReprojectionEnableVrMode);
	LIB_FUNC("FkQX7rjFomk", Hmd2::Hmd2ReprojectionSetTiming);
}

namespace VrTracker2Module {
LIB_VERSION("VrTracker2", 1, "VrTracker2", 1, 1);
LIB_DEFINE(InitVrTracker2_1) {
	LIB_FUNC("TwqZnaIjWv4", Hmd2::VrTracker2QueryMemory);
	LIB_FUNC("6Jy73SRfG-o", Hmd2::VrTracker2Initialize);
    LIB_FUNC("Dog+g25QYjw", Hmd2::VrTracker2RegisterDevice);
    LIB_FUNC("J4Vh3VVX0iU", Hmd2::VrTracker2GetResult);
    LIB_FUNC("IQ3UD6SZbXo", Hmd2::VrTracker2Finalize);
    LIB_FUNC("UVCMLmS-Eas", Hmd2::VrTracker2SetCoordinateSystem);
    LIB_FUNC("IdI2f+xHIeA", Hmd2::VrTracker2ResetLocalCoordinate);
}
} // namespace VrTracker2Module

} // namespace Libs
