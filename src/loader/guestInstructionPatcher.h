// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#ifndef KYTY_LOADER_GUEST_INSTRUCTION_PATCHER_H_
#define KYTY_LOADER_GUEST_INSTRUCTION_PATCHER_H_

#include "common/common.h"

#include <span>
#include <vector>

namespace Loader {

struct GuestInstructionHostFeatures {
	bool sse4a = false;
	bool rdpid = false;
	bool clwb  = false;
};

GuestInstructionHostFeatures GetGuestInstructionHostFeatures();

struct InstructionPatchCounts {
	uint64_t found   = 0;
	uint64_t native  = 0;
	uint64_t trapped = 0;

	uint64_t Skipped() const { return found - native - trapped; }

	InstructionPatchCounts& operator+=(const InstructionPatchCounts& other) {
		found += other.found;
		native += other.native;
		trapped += other.trapped;
		return *this;
	}
};

struct GuestInstructionPatchResult {
	uint64_t               function_count                           = 0;
	uint64_t               instruction_count                        = 0;
	uint64_t               red_zone_function_count                  = 0;
	uint64_t               memory_instruction_count                 = 0;
	uint64_t               short_memory_instruction_count           = 0;
	uint64_t               patched_memory_instruction_count         = 0;
	uint64_t               stack_dependent_memory_instruction_count = 0;
	uint64_t               control_flow_memory_instruction_count    = 0;
	uint64_t               unrelocatable_memory_instruction_count   = 0;
	uint64_t               indirect_red_zone_function_count         = 0;
	InstructionPatchCounts reciprocal_sqrt;
	InstructionPatchCounts extrq;
	InstructionPatchCounts insertq;
	InstructionPatchCounts rdpid;
	InstructionPatchCounts clwb;
};

void RegisterGuestInstructionPatchModule(void* module_ptr, uint64_t module_size,
                                         void* trampoline_area_ptr, uint64_t trampoline_area_size);
void UnregisterGuestInstructionPatchModule(void* module_ptr);

// Opt-in diagnostic for the verified PPSA05666 eboot only. The loader installs
// before guest execution; the guest hook records without calling host code.
// KYTY_HORIZON_KISMET_TRACE=1 enables calibration/movement. Additionally setting
// KYTY_HORIZON_KISMET_GAMEPLAY_TRACE=1 includes the verified game-mode and river-intro
// graphs. GameMode entry 0x05ea and intro entries 0x02e1/0x03e3 record their first 8
// invocations, then every 64th; separate saturating poll counters keep advancing
// after the shared record capacity fills and are reported periodically by Drain.
// Other entries retain normal tracing; the record layout and capacity do not change.
// On Windows, KYTY_HORIZON_KISMET_TRACE_FILE optionally names a new file in an
// existing directory. CREATE_NEW preserves older captures; any file error rejects
// installation. Without a path, trace storage remains in memory only.
// File layout 1 is little-endian: u32 reservation count@0, records@8, 2048 slots
// of 64 bytes (131080 bytes total). Each slot has u32 committed@0, entry@4,
// script_num@8, reserved@12; u64 object@16, node@24, frame@32, script@40, code@48,
// native_frame@56. Read a slot only when committed == its zero-based index + 1;
// a process crash can leave an uncommitted gap before later completed slots.
bool HorizonKismetTraceEnabled();
bool InstallHorizonKismetTrace(uint64_t module_addr, uint64_t module_size);
// Call from a periodic host frame path. A disabled trace only checks an atomic.
void DrainHorizonKismetTrace();

// Apply enabled instruction fixes using native trampolines or safe trap fallbacks.
GuestInstructionPatchResult PatchGuestInstructions(
    uint64_t segment_addr, uint64_t segment_size, std::span<const uintptr_t> function_starts,
    bool protect_memory, bool emulate_amd,
    GuestInstructionHostFeatures host_features = GetGuestInstructionHostFeatures());

bool DecodeEhFrameFunctionStarts(uint64_t eh_frame_header_addr, uint64_t eh_frame_header_size,
                                 std::vector<uintptr_t>* function_starts);

} // namespace Loader

#endif /* KYTY_LOADER_GUEST_INSTRUCTION_PATCHER_H_ */
